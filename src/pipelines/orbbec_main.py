"""Main entrypoint for modular Orbbec modes pipeline (shared-based)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyk4a import Config, FPS, PyK4A

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import runtime.hand_tracking_orbbec as ob
import runtime.hand_tracking_webcam_modes as hm
from shared.common_utils import resolve_model_path
from shared.hand_constants import MCP_IDS, WRIST_ID
from shared.morph_lp_plot import update_3d_plot_lp
from shared.morph_shape_control import (
    LpShapePipelineState,
    advance_lp_shape_p,
    index_mcp_tip_segment_norm,
)
from shared.topology_utils import clamp01
from shared.mp_hand_utils import find_hand_index_by_side
from shared.modes_runtime import (
    ModeState,
    RightHandState,
    RuntimeState,
    SnapVisualState,
    draw_bottom_status,
    overlay_mode_open_wrist_labels,
    process_left_mode,
    process_right_open,
    update_hud_cache,
    update_snap_visual_state_for_modes,
)
from shared.stream_runtime_utils import (
    capture_orbbec_frame,
    detect_for_video_safe,
    get_aligned_depth,
    make_mp_image_from_bgr,
    safe_get_capture,
)


@dataclass
class OrbbecModesConfig:
    model_path: str
    fusion_w: float
    ema_a: float
    depth_patch_r: int
    hand_frame: str
    shape_norm: bool
    depth_outlier_filter: bool
    open_remap: Optional[Tuple[float, float]]
    depth_max_delta_mm: float
    depth_median_max_delta_mm: Optional[float]
    depth_rigid_T: Optional[np.ndarray]
    use_transformed_depth: bool
    hand_3d: str


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Orbbec/K4A: Left→mode 1/2/3; Right→open/morph (depth-fused 3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Same depth/RGB options as hand_tracking_orbbec.py. "
            "See that script's --help for calibration and fusion notes."
        ),
    )
    ap.add_argument("--model", type=str, default=None, help="hand_landmarker.task path")
    ap.add_argument("--depth-fusion", type=float, default=ob.DEPTH_FUSION_WEIGHT, help="0=MediaPipe world only, 1=depth unproject only")
    ap.add_argument("--ema-alpha", type=float, default=ob.POINT_EMA_ALPHA, help="EMA smoothing on fused keypoints (0..1)")
    ap.add_argument(
        "--hand-frame",
        choices=(ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE, ob.HAND_FRAME_METRIC_MM),
        default=ob.HAND_FRAME_SCALED,
        help="3D coords: scaled | palm_plane | metric_mm",
    )
    ap.add_argument("--no-shape-normalize", action="store_true", help="Same as --hand-frame metric_mm (unless palm_plane / metric_mm already set).")
    ap.add_argument("--no-depth-outlier-filter", action="store_true", help="Keep all per-joint depth samples.")
    ap.add_argument("--no-open-remap", action="store_true", help="Use raw morph_alpha for HUD (default: remap with shape_norm).")
    ap.add_argument("--open-remap-lo", type=float, default=None, metavar="X", help=f"Lower bound for open remap (default {ob.OPEN_REMAP_LO}).")
    ap.add_argument("--open-remap-hi", type=float, default=None, metavar="X", help=f"Upper bound for open remap (default {ob.OPEN_REMAP_HI}).")
    ap.add_argument("--depth-patch", type=int, default=ob.DEPTH_MEDIAN_PATCH_RADIUS, metavar="R", help="Median depth patch radius")
    ap.add_argument("--depth-max-delta-mm", type=float, default=None, metavar="D", help=f"Max |depth−wrist| per joint (default {ob.DEPTH_MAX_DELTA_FROM_WRIST_MM}).")
    ap.add_argument("--depth-median-max-delta-mm", type=float, default=None, metavar="D", help="Second pass vs palm median; 0 disables.")
    ap.add_argument("--use-transformed-depth", action="store_true", help="Use SDK depth→color alignment (often unsafe on Orbbec K4A wrapper).")
    ap.add_argument("--hand-3d", choices=(ob.HAND_3D_SOURCE_MP, ob.HAND_3D_SOURCE_FUSED), default=ob.HAND_3D_SOURCE_MP, help="3D plot source: mp | fused")
    ap.add_argument("--depth-unproject-rigid-npy", type=str, default=None, metavar="PATH", help="Optional 4×4 .npy rigid transform after depth unproject.")
    return ap


def parse_config(argv: Optional[list[str]] = None) -> OrbbecModesConfig:
    args = build_parser().parse_args(argv)
    model_path = resolve_model_path(args.model, __file__)
    fusion_w = float(np.clip(args.depth_fusion, 0.0, 1.0))
    ema_a = float(np.clip(args.ema_alpha, 0.0, 1.0))
    depth_patch_r = int(np.clip(args.depth_patch, 0, 15))
    if args.hand_frame != ob.HAND_FRAME_SCALED:
        hand_frame = args.hand_frame
    elif args.no_shape_normalize:
        hand_frame = ob.HAND_FRAME_METRIC_MM
    else:
        hand_frame = ob.HAND_FRAME_SCALED
    shape_norm = hand_frame in (ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE)
    depth_outlier_filter = not args.no_depth_outlier_filter
    if shape_norm and not args.no_open_remap:
        lo = ob.OPEN_REMAP_LO if args.open_remap_lo is None else args.open_remap_lo
        hi = ob.OPEN_REMAP_HI if args.open_remap_hi is None else args.open_remap_hi
        open_remap = (lo, hi) if hi > lo + 1e-6 else None
    else:
        open_remap = None
    depth_max_delta_mm = ob.DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = ob.DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)
    depth_rigid_T = ob.load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy)
    return OrbbecModesConfig(
        model_path=model_path,
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
        use_transformed_depth=bool(args.use_transformed_depth),
        hand_3d=str(args.hand_3d),
    )


def build_landmarker(model_path: str):
    options = hm.HandLandmarkerOptions(
        base_options=hm.BaseOptions(model_asset_path=model_path),
        running_mode=hm.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return hm.HandLandmarker.create_from_options(options)


def build_k4a() -> PyK4A:
    return PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30,
        )
    )


def detect_frame(landmarker, frame, frame_idx: int):
    mp_image = make_mp_image_from_bgr(frame)
    t_ms = int(frame_idx * (1000 / 30))
    return detect_for_video_safe(landmarker, mp_image, t_ms, warn_prefix="detect_for_video")


def _print_periodic_metrics(frame_idx: int, hand_frame: str, hands_3d, analyses, open_out, open_free_ema, open_remap, morph_mode: int, mode_raw: int):
    if frame_idx % 30 != 0 or not analyses:
        return
    a0 = analyses[0]
    out_v = open_out if open_out is not None else a0["morph_alpha"]
    free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
    if open_remap is not None:
        lo_r, hi_r = open_remap
        out_show = ob._remap_open_display(out_v, lo_r, hi_r)
        free_show = ob._remap_open_display(free_v, lo_r, hi_r)
        open_part = f"open={out_show:.3f} raw={out_v:.3f} free={free_show:.3f} raw_free={free_v:.3f}"
    else:
        open_part = f"open_out={out_v:.3f} free={free_v:.3f}"
    curl_part = ""
    if hand_frame == ob.HAND_FRAME_PALM_PLANE and hands_3d:
        cm = ob._palm_plane_curl_metrics(hands_3d[0])
        if cm and cm.get("mean_r_xy_four") is not None:
            tr = cm.get("thumb_r_xy")
            thumb_p = f" thumb_r={tr:.3f}" if tr is not None else ""
            curl_part = f" curl_rxy4={cm['mean_r_xy_four']:.3f} curl_z4={cm['mean_abs_z_four']:.3f}{thumb_p}"
    print(f"M{morph_mode} raw={mode_raw} {open_part} spread={a0['finger_spread']:.3f} plan={a0['planarity']:.3f} iso={a0['isotropy']:.3f}{curl_part}")


def run(config: OrbbecModesConfig):
    landmarker = build_landmarker(config.model_path)
    k4a = build_k4a()
    plt.ion()
    fig = plt.figure("Hand 3D Orbbec")
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")
    runtime = RuntimeState(enable_3d=hm.ENABLE_3D_PLOT)
    mode_state = ModeState()
    right_state = RightHandState()
    snap_visual = SnapVisualState()
    lp_shape = LpShapePipelineState()
    k4a.start()
    calib = k4a.calibration
    print(
        "Depth-fused modes. "
        f"fusion={config.fusion_w:.2f} ema={config.ema_a:.2f} depth_patch={config.depth_patch_r}  "
        f"hand_frame={config.hand_frame}  shape_norm={config.shape_norm}  hand_3d={config.hand_3d}  "
        f"open_remap={config.open_remap}  aligned_depth={config.use_transformed_depth}  "
        f"depth_rigid_T={'on' if config.depth_rigid_T is not None else 'off'}  "
        "L=mode 1/2/3  R=open  q=quit  p=3D  s=save"
    )
    try:
        while True:
            capture = safe_get_capture(k4a)
            if capture is None:
                continue
            got = capture_orbbec_frame(capture)
            if got is None:
                continue
            frame, depth_raw, capture = got
            depth_aligned = get_aligned_depth(capture, frame, config.use_transformed_depth)
            result = detect_frame(landmarker, frame, runtime.frame_idx)
            if result is None:
                continue
            if (
                config.fusion_w >= 0.99
                and not config.use_transformed_depth
                and config.hand_3d == ob.HAND_3D_SOURCE_FUSED
                and not runtime.warned_fusion_linear_map
            ):
                print("[INFO] depth-fusion≈1 + --hand-3d fused: linear RGB→depth map; try --hand-3d mp or --depth-patch 3–4.")
                runtime.warned_fusion_linear_map = True
            frame, keypoints_3d, runtime.ema_3d = ob.draw_hand(
                frame,
                result,
                depth_raw=depth_raw,
                depth_aligned=depth_aligned,
                print_depth=(runtime.frame_idx % 30 == 0),
                calibration=calib,
                fusion_weight=config.fusion_w,
                ema_alpha=config.ema_a,
                ema_points=runtime.ema_3d,
                depth_patch_radius=config.depth_patch_r,
                hand_frame=config.hand_frame,
                filter_depth_outliers=config.depth_outlier_filter,
                depth_max_delta_mm=config.depth_max_delta_mm,
                depth_median_max_delta_mm=config.depth_median_max_delta_mm,
                hand_3d_source=config.hand_3d,
                depth_unproject_rigid_T=config.depth_rigid_T,
                skip_wrist_labels=True,
            )
            idx_left = find_hand_index_by_side(result, "left")
            idx_right = find_hand_index_by_side(result, "right")
            mode_raw, tier_count = process_left_mode(keypoints_3d, idx_left, mode_state)
            hands_3d, open_out = process_right_open(keypoints_3d, idx_right, right_state)

            pts_left = None
            if idx_left is not None and idx_left < len(keypoints_3d):
                pts_left = keypoints_3d[idx_left]
            dist_norm = (
                index_mcp_tip_segment_norm(pts_left, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
                if pts_left is not None
                else None
            )
            advance_lp_shape_p(dist_norm, int(mode_state.morph_mode), lp_shape)
            overlay_mode_open_wrist_labels(
                frame=frame,
                result=result,
                idx_left=idx_left,
                idx_right=idx_right,
                morph_mode=mode_state.morph_mode,
                open_out=open_out,
                overlay_wrist_labels_fn=ob.overlay_wrist_labels,
            )
            draw_bottom_status(frame, mode_state.morph_mode, mode_raw, tier_count, idx_left, idx_right, open_out)
            update_snap_visual_state_for_modes(right_state.snap_state, snap_visual)
            analyses = None
            if runtime.enable_3d and (runtime.frame_idx % ob.PLOT_EVERY_N_FRAMES) == 0 and hands_3d:
                analyses = update_3d_plot_lp(
                    ax_hand,
                    ax_topo,
                    hands_3d,
                    morph_mode=mode_state.morph_mode,
                    morph_alpha_smoothed=open_out,
                    control_label="open+p",
                    analyze_hand_topology_fn=ob.analyze_hand_topology,
                    clamp01_fn=clamp01,
                    shape_normalized=config.shape_norm,
                    hand_frame=config.hand_frame,
                    hand_3d_source=config.hand_3d,
                    hand_frame_palm_plane=ob.HAND_FRAME_PALM_PLANE,
                    norm_axis_halflim=ob.NORM_AXIS_HALFLIM,
                    morph_axis_lim_mm=ob.MORPH_AXIS_LIM_MM,
                    mode_shape_t=lp_shape.left_shape_t_ema,
                    epsilon_pair_display=lp_shape.epsilon_pair_display,
                )
                plt.pause(0.0001)
            if analyses:
                update_hud_cache(
                    runtime=runtime,
                    frame_idx=runtime.frame_idx,
                    analyses=analyses,
                    hands_3d=hands_3d,
                    hand_frame=config.hand_frame,
                    morph_mode=mode_state.morph_mode,
                    open_out=open_out,
                    open_free_ema=right_state.open_free_ema,
                    open_remap=config.open_remap,
                    snap_vis_state=snap_visual.snap_vis_state,
                )
                _print_periodic_metrics(runtime.frame_idx, config.hand_frame, hands_3d, analyses, open_out, right_state.open_free_ema, config.open_remap, mode_state.morph_mode, mode_raw)
            cv2.imshow("Hand Tracking Orbbec Modes", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                out_name = f"hand_orbbec_mode_frame_{runtime.frame_idx:06d}.png"
                fig.savefig(out_name, dpi=150, bbox_inches="tight")
                print(f"Saved: {out_name}")
            if key == ord("p"):
                runtime.enable_3d = not runtime.enable_3d
                print(f"3D plot: {runtime.enable_3d}")
            if key == ord("q"):
                break
            if runtime.hud_cache["text"] is not None:
                ob.draw_hud(frame, runtime.hud_cache["text"], origin=(16, 16))
            runtime.frame_idx += 1
    finally:
        landmarker.close()
        k4a.stop()
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()


def main():
    run(parse_config())


if __name__ == "__main__":
    main()

