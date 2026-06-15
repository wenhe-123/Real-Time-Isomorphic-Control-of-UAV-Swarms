"""Standalone Orbbec hand-tracking loop (backup demo). Library: ``shared.display_sim.orbbec_hand``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from pyk4a import Config, FPS, PyK4A

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from functions.display_sim.common_utils import draw_hud
from functions.mode_switch.hand_constants import MCP_IDS, WRIST_ID
from functions.mode_switch.modes_runtime import (
    ModeState,
    RightHandState,
    palm_plane_curl_metrics,
    process_left_mode,
    process_right_open,
    topology_label_from_morph_alpha,
)
from functions.open_close.morph_renderers import prompt_and_init_fixed_surface_points
from functions.mode_switch.morph_shape_control import LpShapePipelineState, advance_lp_shape_p, index_mcp_tip_segment_norm
from functions.dual_cam.mp_hand_utils import find_hand_index_by_side
from backup.runtime import demo_defaults as demo
from functions.display_sim.orbbec_hand import (
    DEPTH_MAX_DELTA_FROM_WRIST_MM,
    DEPTH_MEDIAN_MAX_DELTA_MM,
    DEPTH_MEDIAN_PATCH_RADIUS,
    OPEN_REMAP_HI,
    OPEN_REMAP_LO,
    POINT_EMA_ALPHA,
    analyze_hand_topology,
    draw_hand,
    load_depth_unproject_rigid_npy,
    update_3d_plot,
)
from backup.runtime.orbbec_live_steps import (
    compute_open_out as _shared_compute_open_out,
    print_periodic_topology_status as _shared_print_periodic_topology_status,
    refresh_hud_cache as _shared_refresh_hud_cache,
    update_snap_visual_state as _shared_update_snap_visual_state,
)
from functions.dual_cam.stream_runtime_utils import (
    capture_orbbec_frame,
    detect_for_video_safe,
    get_aligned_depth,
    make_mp_image_from_bgr,
    safe_get_capture,
)
from functions.mode_switch.topology_utils import remap_open_display
from functions.mode_switch.webcam_mode_defaults import MORPH_AXIS_LIM_MM, NORM_AXIS_HALFLIM

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _build_orbbec_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Orbbec/K4A hand tracking with depth-fused 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Femto Bolt (e.g. F00364-152): for metric RGB–depth alignment, obtain from Orbbec SDK / Viewer:\n"
            "  • Color intrinsics (fx,fy,cx,cy) + distortion; depth intrinsics; depth scale (mm).\n"
            "  • Extrinsic T_depth_to_color (or color_to_depth).\n"
            "This script uses pyk4a’s K4A-style calibration blob; if the wrapper reports placeholder calib,\n"
            "use --depth-fusion 0 for stable MediaPipe 3D until calibration is verified.\n"
            "When --depth-fusion is near 1, try --depth-patch 3 and lower --ema-alpha (e.g. 0.15) "
            "if the skeleton is jittery.\n"
            "Default: --hand-frame scaled (wrist-centered + palm-scale). "
            "Use --hand-frame palm_plane for wrist origin + palm XY plane + fingertip curl metrics.\n"
            "Default --hand-3d mp: 3D skeleton from MediaPipe (stable); use --hand-3d fused for depth blend.\n"
        ),
    )
    ap.add_argument("--model", type=str, default="hand_landmarker.task", help="hand_landmarker.task path")
    ap.add_argument(
        "--depth-fusion",
        type=float,
        default=demo.DEPTH_FUSION_WEIGHT,
        help="0=MediaPipe world only, 1=depth unproject only (per-joint linear blend in between)",
    )
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=POINT_EMA_ALPHA,
        help="EMA smoothing 0..1 on fused keypoints (higher = faster tracking)",
    )
    ap.add_argument(
        "--hand-frame",
        choices=(demo.HAND_FRAME_SCALED, demo.HAND_FRAME_PALM_PLANE, demo.HAND_FRAME_METRIC_MM),
        default=demo.HAND_FRAME_SCALED,
        help=(
            "3D skeleton + topology coords: scaled=legacy wrist+palm-scale; "
            "palm_plane=wrist origin, XY=plane(wrist,index MCP,middle MCP), /palm-scale (fingertip curl in HUD); "
            "metric_mm=no normalization (raw mm)."
        ),
    )
    ap.add_argument(
        "--no-shape-normalize",
        action="store_true",
        help="Same as --hand-frame metric_mm (ignored if --hand-frame is set to palm_plane or metric_mm).",
    )
    ap.add_argument(
        "--no-depth-outlier-filter",
        action="store_true",
        help="Keep all per-joint depth samples (default: drop joints with depth far from wrist / >1.8m).",
    )
    ap.add_argument(
        "--no-open-remap",
        action="store_true",
        help="Use raw morph_alpha for blanket/HUD (default with shape_norm: map ~0.22–0.72 → 0–1).",
    )
    ap.add_argument(
        "--open-remap-lo",
        type=float,
        default=None,
        metavar="X",
        help=f"Lower bound for open linear remap (default {OPEN_REMAP_LO}).",
    )
    ap.add_argument(
        "--open-remap-hi",
        type=float,
        default=None,
        metavar="X",
        help=f"Upper bound for open linear remap (default {OPEN_REMAP_HI}).",
    )
    ap.add_argument(
        "--depth-patch",
        type=int,
        default=DEPTH_MEDIAN_PATCH_RADIUS,
        metavar="R",
        help=(
            "Median depth over (2R+1)^2 pixels on depth image (0=single pixel). "
            "Larger R reduces speckle when --depth-fusion is high; try 2–4."
        ),
    )
    ap.add_argument(
        "--depth-max-delta-mm",
        type=float,
        default=None,
        metavar="D",
        help=f"Max |depth−wrist| per joint in mm (default {DEPTH_MAX_DELTA_FROM_WRIST_MM}).",
    )
    ap.add_argument(
        "--depth-median-max-delta-mm",
        type=float,
        default=None,
        metavar="D",
        help=(
            "Second pass: max |depth−median(wrist+MCPs)|; 0 disables (default "
            f"{DEPTH_MEDIAN_MAX_DELTA_MM})."
        ),
    )
    ap.add_argument(
        "--use-transformed-depth",
        action="store_true",
        help=(
            "Use SDK depth→color alignment (transformed_depth). Real Azure Kinect only; "
            "Orbbec Femto / K4A-wrapper often crashes on this path — leave off (default)."
        ),
    )
    ap.add_argument(
        "--hand-3d",
        choices=(demo.HAND_3D_SOURCE_MP, demo.HAND_3D_SOURCE_FUSED),
        default=demo.HAND_3D_SOURCE_MP,
        help=(
            "3D plot + topology: mp=MediaPipe world (stable hand shape); "
            "fused=depth+MP per joint (only if RGB–D alignment is good)."
        ),
    )
    ap.add_argument(
        "--depth-unproject-rigid-npy",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional 4×4 float64 .npy (mm, homogeneous): T@[x,y,z,1] applied to each depth-unprojected point "
            "before fusion. Use offline calibration (depth camera → frame comparable to MediaPipe). "
            "Ignored for pure 3D when --hand-3d mp and depth-fusion=0."
        ),
    )
    ap.add_argument(
        "--show-sample-ids",
        action="store_true",
        help="Draw sample point ID text in 3D (slower; off by default).",
    )
    return ap


def _normalize_main_args(args):
    model_path = args.model
    fusion_w = float(np.clip(args.depth_fusion, 0.0, 1.0))
    ema_a = float(np.clip(args.ema_alpha, 0.0, 1.0))
    depth_patch_r = int(np.clip(args.depth_patch, 0, 15))
    if args.hand_frame != demo.HAND_FRAME_SCALED:
        hand_frame = args.hand_frame
    elif args.no_shape_normalize:
        hand_frame = demo.HAND_FRAME_METRIC_MM
    else:
        hand_frame = demo.HAND_FRAME_SCALED
    shape_norm = hand_frame in (demo.HAND_FRAME_SCALED, demo.HAND_FRAME_PALM_PLANE)
    depth_outlier_filter = not args.no_depth_outlier_filter
    if shape_norm and not args.no_open_remap:
        lo = OPEN_REMAP_LO if args.open_remap_lo is None else args.open_remap_lo
        hi = OPEN_REMAP_HI if args.open_remap_hi is None else args.open_remap_hi
        open_remap = (lo, hi) if hi > lo + 1e-6 else None
    else:
        open_remap = None

    depth_max_delta_mm = (
        DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    )
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)
    depth_rigid_T = load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy)
    return (
        model_path,
        fusion_w,
        ema_a,
        depth_patch_r,
        hand_frame,
        shape_norm,
        depth_outlier_filter,
        open_remap,
        depth_max_delta_mm,
        depth_median_max_delta_mm,
        depth_rigid_T,
    )


def _run_frame_step(
    *,
    k4a,
    landmarker,
    fig,
    ax_hand,
    ax_topo,
    calib,
    args,
    fusion_w: float,
    ema_a: float,
    depth_patch_r: int,
    hand_frame: str,
    shape_norm: bool,
    depth_outlier_filter: bool,
    open_remap,
    depth_max_delta_mm: float,
    depth_median_max_delta_mm,
    depth_rigid_T,
    frame_idx: int,
    warned_fusion_linear_map: bool,
    ema_3d,
    open_free_ema,
    alpha_smooth: float,
    snap_state,
    hud_cache: dict,
    snap_vis_state,
    snap_stable_frames: int,
    snap_hold_frames: int,
    enable_3d: bool,
    mode_state: Optional[ModeState] = None,
    right_state: Optional[RightHandState] = None,
    lp_shape: Optional[LpShapePipelineState] = None,
):
    # One frame pipeline: capture -> detect -> draw -> topology/HUD -> UI events.
    capture = safe_get_capture(k4a, warn_prefix="get_capture")
    if capture is None:
        return (
            frame_idx,
            warned_fusion_linear_map,
            ema_3d,
            open_free_ema,
            snap_state,
            snap_vis_state,
            snap_stable_frames,
            snap_hold_frames,
            enable_3d,
            False,
        )
    got = capture_orbbec_frame(capture)
    if got is None:
        return (
            frame_idx,
            warned_fusion_linear_map,
            ema_3d,
            open_free_ema,
            snap_state,
            snap_vis_state,
            snap_stable_frames,
            snap_hold_frames,
            enable_3d,
            False,
        )

    frame, depth_raw, capture = got
    depth_aligned = get_aligned_depth(capture, frame, args.use_transformed_depth)
    mp_image = make_mp_image_from_bgr(frame)
    t_ms = int(frame_idx * (1000 / 30))
    result = detect_for_video_safe(landmarker, mp_image, t_ms, warn_prefix="mediapipe detect_for_video")
    if result is None:
        return (
            frame_idx,
            warned_fusion_linear_map,
            ema_3d,
            open_free_ema,
            snap_state,
            snap_vis_state,
            snap_stable_frames,
            snap_hold_frames,
            enable_3d,
            False,
        )
    if (
        fusion_w >= 0.99
        and not args.use_transformed_depth
        and args.hand_3d == demo.HAND_3D_SOURCE_FUSED
        and not warned_fusion_linear_map
    ):
        print(
            "[INFO] depth-fusion≈1 + --hand-3d fused: linear RGB→depth map; 3D can be wrong "
            "without good calibration. Try --hand-3d mp for stable skeleton, or --depth-patch 3–4."
        )
        warned_fusion_linear_map = True

    frame, hands_3d, ema_3d = draw_hand(
        frame,
        result,
        depth_raw=depth_raw,
        depth_aligned=depth_aligned,
        print_depth=(frame_idx % 30 == 0),
        calibration=calib,
        fusion_weight=fusion_w,
        ema_alpha=ema_a,
        ema_points=ema_3d,
        depth_patch_radius=depth_patch_r,
        hand_frame=hand_frame,
        filter_depth_outliers=depth_outlier_filter,
        depth_max_delta_mm=depth_max_delta_mm,
        depth_median_max_delta_mm=depth_median_max_delta_mm,
        hand_3d_source=args.hand_3d,
        depth_unproject_rigid_T=depth_rigid_T,
    )
    if mode_state is not None and right_state is not None and lp_shape is not None:
        idx_left = find_hand_index_by_side(result, "left")
        idx_right = find_hand_index_by_side(result, "right")
        _mode_raw, _tier = process_left_mode(hands_3d, idx_left, mode_state)
        hands_3d_topo, open_out = process_right_open(hands_3d, idx_right, right_state)
        pts_left = hands_3d[idx_left] if idx_left is not None and idx_left < len(hands_3d) else None
        dist_norm = (
            index_mcp_tip_segment_norm(pts_left, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
            if pts_left is not None
            else None
        )
        advance_lp_shape_p(dist_norm, int(mode_state.morph_mode), lp_shape)
        open_free_ema = right_state.open_free_ema
        snap_state = right_state.snap_state
    else:
        open_out, open_free_ema, snap_state = _shared_compute_open_out(
            hands_3d=hands_3d,
            open_free_ema=open_free_ema,
            snap_state=snap_state,
            alpha_smooth=alpha_smooth,
            analyze_topology_fn=analyze_hand_topology,
            plane_snap_on=demo.PLANE_SNAP_ON,
            plane_snap_off=demo.PLANE_SNAP_OFF,
            sphere_snap_on=demo.SPHERE_SNAP_ON,
            sphere_snap_off=demo.SPHERE_SNAP_OFF,
        )
        hands_3d_topo = hands_3d
    snap_vis_state, snap_stable_frames, snap_hold_frames = _shared_update_snap_visual_state(
        snap_state=snap_state,
        snap_vis_state=snap_vis_state,
        snap_stable_frames=snap_stable_frames,
        snap_hold_frames=snap_hold_frames,
        snap_show_after_frames=demo.SNAP_SHOW_AFTER_FRAMES,
        snap_hold_after_release_frames=demo.SNAP_HOLD_AFTER_RELEASE_FRAMES,
    )

    analyses = None
    if enable_3d and (frame_idx % demo.PLOT_EVERY_N_FRAMES) == 0 and hands_3d_topo:
        if lp_shape is not None and mode_state is not None:
            analyses = update_3d_plot(
                ax_hand,
                ax_topo,
                hands_3d_topo,
                morph_alpha_smoothed=open_out,
                morph_mode=mode_state.morph_mode,
                mode_shape_t=lp_shape.left_shape_t_ema,
                epsilon_pair_display=lp_shape.epsilon_pair_display,
                shape_normalized=shape_norm,
                hand_frame=hand_frame,
                hand_3d_source=args.hand_3d,
                show_sample_ids=bool(args.show_sample_ids),
            )
        else:
            analyses = update_3d_plot(
                ax_hand,
                ax_topo,
                hands_3d_topo,
                morph_alpha_smoothed=open_out,
                shape_normalized=shape_norm,
                hand_frame=hand_frame,
                hand_3d_source=args.hand_3d,
                show_sample_ids=bool(args.show_sample_ids),
            )
        plt.pause(0.0001)

    if analyses:
        a0 = analyses[0]
        _shared_refresh_hud_cache(
            hud_cache=hud_cache,
            frame_idx=frame_idx,
            a0=a0,
            hands_3d=hands_3d_topo,
            hand_frame=hand_frame,
            hand_frame_palm_plane=demo.HAND_FRAME_PALM_PLANE,
            open_out=open_out,
            open_free_ema=open_free_ema,
            open_remap=open_remap,
            snap_vis_state=snap_vis_state,
            hud_update_every_n_frames=demo.HUD_UPDATE_EVERY_N_FRAMES,
            hud_open_step=demo.HUD_OPEN_STEP,
            hud_metric_step=demo.HUD_METRIC_STEP,
            topology_label_fn=topology_label_from_morph_alpha,
            remap_open_display_fn=remap_open_display,
            palm_plane_curl_metrics_fn=palm_plane_curl_metrics,
        )
        _shared_print_periodic_topology_status(
            frame_idx=frame_idx,
            a0=a0,
            hands_3d=hands_3d_topo,
            hand_frame=hand_frame,
            hand_frame_palm_plane=demo.HAND_FRAME_PALM_PLANE,
            open_out=open_out,
            open_free_ema=open_free_ema,
            open_remap=open_remap,
            topology_label_fn=topology_label_from_morph_alpha,
            remap_open_display_fn=remap_open_display,
            palm_plane_curl_metrics_fn=palm_plane_curl_metrics,
        )

    cv2.imshow("Hand Tracking Orbbec", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        out_name = f"hand_3d_frame_{frame_idx:06d}.png"
        fig.savefig(out_name, dpi=150, bbox_inches="tight")
        print(f"Saved 3D plot: {out_name}")
    if key == ord("p"):
        enable_3d = not enable_3d
        print(f"3D plot enabled: {enable_3d}")
    if key == ord("q"):
        return (
            frame_idx,
            warned_fusion_linear_map,
            ema_3d,
            open_free_ema,
            snap_state,
            snap_vis_state,
            snap_stable_frames,
            snap_hold_frames,
            enable_3d,
            True,
        )

    # Draw HUD every frame from cache to avoid flicker.
    if hud_cache["text"] is not None:
        draw_hud(frame, hud_cache["text"], origin=(16, 16))
    frame_idx += 1
    return (
        frame_idx,
        warned_fusion_linear_map,
        ema_3d,
        open_free_ema,
        snap_state,
        snap_vis_state,
        snap_stable_frames,
        snap_hold_frames,
        enable_3d,
        False,
    )


def main():
    args = _build_orbbec_arg_parser().parse_args()
    prompt_and_init_fixed_surface_points()
    model_path, fusion_w, ema_a, depth_patch_r, hand_frame, shape_norm, depth_outlier_filter, open_remap, depth_max_delta_mm, depth_median_max_delta_mm, depth_rigid_T = _normalize_main_args(args)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    # same camera opening style as test_orbbec.py
    k4a = PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30,
        )
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        plt.ion()
        fig = plt.figure("Hand 3D and Topology")
        ax_hand = fig.add_subplot(121, projection="3d")
        ax_topo = fig.add_subplot(122, projection="3d")

        k4a.start()
        calib = k4a.calibration
        print(
            "Depth-fused 3D. "
            f"fusion={fusion_w:.2f} ema={ema_a:.2f} depth_patch={depth_patch_r}  "
            f"hand_frame={hand_frame}  shape_norm={shape_norm}  hand_3d={args.hand_3d}  "
            f"depth_outlier_filter={depth_outlier_filter}  "
            f"dΔ={depth_max_delta_mm:.0f}mm medΔ={depth_median_max_delta_mm}  "
            f"open_remap={open_remap}  aligned_depth={args.use_transformed_depth}  "
            f"depth_rigid_T={'on' if depth_rigid_T is not None else 'off'}  "
            "q=quit  p=3D  s=save  |  L=mode+Lp (index segment)  R=open  (use backup/pipelines/orbbec_main.py for full HUD)"
        )

        try:
            frame_idx = 0
            warned_fusion_linear_map = False
            ema_3d = None
            open_free_ema = None
            alpha_smooth = 0.18
            snap_state = None  # None / "plane" / "sphere"
            hud_cache = {
                "open": None,
                "free": None,
                "plan": None,
                "iso": None,
                "spread": None,
                "curl": None,
                "text": None,
            }
            snap_vis_state = None
            snap_stable_frames = 0
            snap_hold_frames = 0
            enable_3d = demo.ENABLE_3D_PLOT
            mode_state = ModeState()
            right_state = RightHandState()
            lp_shape = LpShapePipelineState()
            while True:
                (
                    frame_idx,
                    warned_fusion_linear_map,
                    ema_3d,
                    open_free_ema,
                    snap_state,
                    snap_vis_state,
                    snap_stable_frames,
                    snap_hold_frames,
                    enable_3d,
                    should_break,
                ) = _run_frame_step(
                    k4a=k4a,
                    landmarker=landmarker,
                    fig=fig,
                    ax_hand=ax_hand,
                    ax_topo=ax_topo,
                    calib=calib,
                    args=args,
                    fusion_w=fusion_w,
                    ema_a=ema_a,
                    depth_patch_r=depth_patch_r,
                    hand_frame=hand_frame,
                    shape_norm=shape_norm,
                    depth_outlier_filter=depth_outlier_filter,
                    open_remap=open_remap,
                    depth_max_delta_mm=depth_max_delta_mm,
                    depth_median_max_delta_mm=depth_median_max_delta_mm,
                    depth_rigid_T=depth_rigid_T,
                    frame_idx=frame_idx,
                    warned_fusion_linear_map=warned_fusion_linear_map,
                    ema_3d=ema_3d,
                    open_free_ema=open_free_ema,
                    alpha_smooth=alpha_smooth,
                    snap_state=snap_state,
                    hud_cache=hud_cache,
                    snap_vis_state=snap_vis_state,
                    snap_stable_frames=snap_stable_frames,
                    snap_hold_frames=snap_hold_frames,
                    enable_3d=enable_3d,
                    mode_state=mode_state,
                    right_state=right_state,
                    lp_shape=lp_shape,
                )
                if should_break:
                    break
        finally:
            k4a.stop()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


