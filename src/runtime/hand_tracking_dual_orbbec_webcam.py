"""
Dual-view hand tracking: Orbbec RGB-D + laptop webcam.

Orbbec: ``hand_tracking_orbbec.draw_hand`` (depth + calibration). Webcam: MediaPipe world.

**Per-view confidence (what we draw as vis_μ / vis_min):**
  Each 2D landmark carries **visibility** (preferred) or **presence** from MediaPipe Hands.
  These are network-estimated scores in ``[0,1]`` for “how reliably this joint is localized
  in the image”.  **Occlusion / motion blur / self-occlusion** typically **lower** visibility on
  the affected joints, which pulls down **vis_min** (worst joint) and often **vis_mean**.
  They are **not** depth-based; depth fusion is separate (Orbbec pane).

**w_geom:** geometry weight from PCA shape (planarity / isotropy) on that view’s 21×3 world
points — used as a **per-view** multiplier together with per-joint visibility in fusion.

**Fusion (MP world mm):** ``c_{v,k} = visibility_{v,k} * w_geom_v``; weighted mean and
``--fuse-vis-low/high`` gates as before.  **open** from fused hand via ``analyze_hand_topology``.

**Two landmarkers:** ``VIDEO`` mode keeps per-stream temporal state. One shared landmarker
alternating two cameras confuses tracking — use **one HandLandmarker per camera** and separate
timestamps, then fuse the two world-space results.

3D plot: ``shared.morph_lp_plot.update_3d_plot_lp`` (same superellipsoid morph as webcam / Orbbec modes).  Controls: q / p / s.
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pyk4a import Config, FPS, PyK4A

from . import hand_tracking_orbbec as ob
from . import hand_tracking_webcam_modes as hm
from shared.common_utils import draw_hud, resolve_model_path
from shared.hand_constants import MCP_IDS, WRIST_ID
from shared.morph_lp_plot import update_3d_plot_lp
from shared.morph_renderers import prompt_and_init_fixed_surface_points
from shared.morph_shape_control import (
    LpShapePipelineState,
    advance_lp_shape_p,
    index_mcp_tip_segment_norm,
)
from shared.topology_utils import clamp01
from shared.dual_view_utils import draw_hand_webcam, open_webcam_capture, overlay_inset
from shared.fusion_utils import fuse_dual_views_weighted, geom_weight_from_eigen_analysis
from shared.mp_hand_utils import (
    confidence_color_bgr,
    extract_landmark_visibilities,
    extract_world_points_mm_result,
    find_left_right_indices,
    summarize_mp_visibility,
)
from shared.dual_state_utils import init_dual_runtime_state
from shared.modes_runtime import (
    ModeState,
    build_modes_hud_lines,
    overlay_mode_open_wrist_labels,
    update_mode_state,
)
from shared.stream_runtime_utils import (
    capture_orbbec_frame,
    detect_for_video_safe,
    get_aligned_depth,
    make_mp_image_from_bgr,
    normalize_webcam_bgr,
    safe_get_capture,
)

# ===== MediaPipe initialize =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

PLOT_EVERY_N_FRAMES = 5
ENABLE_3D_PLOT = True


def main():
    ap = argparse.ArgumentParser(
        description="Dual Orbbec (depth-fused overlay) + webcam; fused MP world for topology / 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Orbbec pane: hand_tracking_orbbec.draw_hand. Dual fusion: per-joint "
            "c = visibility * w_geom(view), weighted mean with --fuse-vis-low/high; "
            "see module docstring. Depth CLI matches hand_tracking_orbbec.py."
        ),
    )
    ap.add_argument("--model", type=str, default=None, help="Path to hand_landmarker.task")
    ap.add_argument(
        "--webcam-index",
        type=int,
        default=-1,
        help="OpenCV webcam index (-1=auto scan, default -1)",
    )
    ap.add_argument("--webcam-width", type=int, default=0, help="Optional capture width (0=default)")
    ap.add_argument("--webcam-height", type=int, default=0, help="Optional capture height (0=default)")
    ap.add_argument(
        "--webcam-max-index",
        type=int,
        default=8,
        help="When --webcam-index=-1, probe [0..N] (default 8)",
    )
    ap.add_argument(
        "--depth-fusion",
        type=float,
        default=ob.DEPTH_FUSION_WEIGHT,
        help="Orbbec pane: 0=MP world only, 1=depth unproject only",
    )
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=ob.POINT_EMA_ALPHA,
        help="EMA on Orbbec 3D keypoints (0..1)",
    )
    ap.add_argument(
        "--hand-frame",
        choices=(ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE, ob.HAND_FRAME_METRIC_MM),
        default=ob.HAND_FRAME_SCALED,
        help="Orbbec draw_hand coordinate normalization (3D plot axes follow this for Orbbec pane)",
    )
    ap.add_argument(
        "--no-shape-normalize",
        action="store_true",
        help="Same as --hand-frame metric_mm unless palm_plane/metric_mm set",
    )
    ap.add_argument(
        "--no-depth-outlier-filter",
        action="store_true",
        help="Orbbec: keep all depth samples",
    )
    ap.add_argument(
        "--no-open-remap",
        action="store_true",
        help="HUD: raw morph_alpha (default: remap when shape_norm)",
    )
    ap.add_argument("--open-remap-lo", type=float, default=None, metavar="X")
    ap.add_argument("--open-remap-hi", type=float, default=None, metavar="X")
    ap.add_argument(
        "--depth-patch",
        type=int,
        default=ob.DEPTH_MEDIAN_PATCH_RADIUS,
        metavar="R",
        help="Median depth patch radius on Orbbec depth image",
    )
    ap.add_argument("--depth-max-delta-mm", type=float, default=None, metavar="D")
    ap.add_argument("--depth-median-max-delta-mm", type=float, default=None, metavar="D")
    ap.add_argument(
        "--use-transformed-depth",
        action="store_true",
        help="Use SDK depth registered to color (often unsafe on Orbbec pyk4a)",
    )
    ap.add_argument(
        "--hand-3d",
        choices=(ob.HAND_3D_SOURCE_MP, ob.HAND_3D_SOURCE_FUSED),
        default=ob.HAND_3D_SOURCE_MP,
        help="Orbbec pane 3D source for draw_hand",
    )
    ap.add_argument(
        "--depth-unproject-rigid-npy",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional 4×4 .npy rigid after depth unproject",
    )
    ap.add_argument(
        "--fuse-vis-low",
        type=float,
        default=0.12,
        help="Drop joint contribution if c=vis*w_geom < this (default 0.12)",
    )
    ap.add_argument(
        "--fuse-vis-high",
        type=float,
        default=0.72,
        help="If c >= this for a joint, use only the strongest view (100%%) for that joint",
    )
    ap.add_argument(
        "--fuse-debug-every",
        type=int,
        default=0,
        metavar="FRAMES",
        help=(
            "Print fusion debug (eigenvalues, w_geom, mean weights) every FRAMES frames. "
            "Must be a positive integer, e.g. 30 (not the letter N). 0=off."
        ),
    )
    ap.add_argument(
        "--fuse-debug",
        action="store_true",
        help="Shortcut for --fuse-debug-every 30",
    )
    ap.add_argument(
        "--invert-orbbec-handedness",
        action="store_true",
        help="Treat Orbbec MediaPipe handedness as flipped (swap Left/Right). Use if Orbbec view is mirrored vs webcam.",
    )
    ap.add_argument(
        "--invert-webcam-handedness",
        action="store_true",
        help="Treat webcam MediaPipe handedness as flipped (swap Left/Right). Use if webcam view is mirrored vs Orbbec.",
    )
    ap.add_argument(
        "--flip-orbbec-input",
        action="store_true",
        help="Flip Orbbec BGR frame horizontally BEFORE sending to MediaPipe (fix mirrored driver/input).",
    )
    ap.add_argument(
        "--flip-webcam-input",
        action="store_true",
        help="Flip webcam BGR frame horizontally BEFORE sending to MediaPipe (fix mirrored driver/input).",
    )
    ap.add_argument(
        "--handedness-debug-every",
        type=int,
        default=0,
        metavar="FRAMES",
        help="Print and overlay MediaPipe handedness (Left/Right) every FRAMES frames. 0=off.",
    )
    args = ap.parse_args()
    prompt_and_init_fixed_surface_points()
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
    depth_max_delta_mm = (
        ob.DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    )
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = ob.DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)
    depth_rigid_T = ob.load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy)
    fuse_conf_low = float(np.clip(args.fuse_vis_low, 0.0, 0.99))
    fuse_conf_high = float(np.clip(args.fuse_vis_high, fuse_conf_low + 1e-4, 1.0))
    fuse_debug_every = max(0, int(args.fuse_debug_every))
    if args.fuse_debug and fuse_debug_every == 0:
        fuse_debug_every = 30

    # Keep these aligned with the single-view pipelines (e.g. pipelines/orbbec_main.py)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    k4a = PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30,
        )
    )

    cap, webcam_index_used, webcam_backend = open_webcam_capture(
        preferred_index=args.webcam_index,
        width=args.webcam_width,
        height=args.webcam_height,
        max_probe_index=args.webcam_max_index,
    )
    print(f"[INFO] Webcam opened: index={webcam_index_used} backend={webcam_backend}")

    landmarker_o = HandLandmarker.create_from_options(options)
    landmarker_l = HandLandmarker.create_from_options(options)
    plt.ion()
    fig = plt.figure("Hand 3D and Topology (dual)")
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")

    k4a.start()
    calib = k4a.calibration
    print(
        "Dual Orbbec + webcam. Orbbec draw: "
        f"fusion={fusion_w:.2f} ema={ema_a:.2f} depth_patch={depth_patch_r} "
        f"hand_frame={hand_frame} hand_3d={args.hand_3d}  "
        f"q=quit  p=toggle 3D  s=save plot"
    )

    try:
        _st = init_dual_runtime_state(ENABLE_3D_PLOT)
        frame_idx = _st["frame_idx"]
        mp_ts_o_ms = _st["mp_ts_o_ms"]
        mp_ts_l_ms = _st["mp_ts_l_ms"]
        warned_fusion_linear_map = _st["warned_fusion_linear_map"]
        ema_3d = _st["ema_3d"]
        open_free_ema = _st["open_free_ema"]
        alpha_smooth = _st["alpha_smooth"]
        snap_state = _st["snap_state"]
        hud_cache = _st["hud_cache"]
        snap_vis_state = _st["snap_vis_state"]
        snap_stable_frames = _st["snap_stable_frames"]
        snap_hold_frames = _st["snap_hold_frames"]
        enable_3d = _st["enable_3d"]
        mode_state = ModeState()
        lp_shape = LpShapePipelineState()

        while True:
            capture = safe_get_capture(k4a)
            if capture is None:
                continue
            ok_cam, frame_web = cap.read()
            if not ok_cam or frame_web is None:
                print("[WARN] webcam read failed")
                continue
            if args.flip_webcam_input:
                frame_web = cv2.flip(frame_web, 1)

            got = capture_orbbec_frame(capture)
            if got is None:
                continue
            frame, depth_raw, capture = got
            if args.flip_orbbec_input:
                frame = cv2.flip(frame, 1)
            depth_map = depth_raw
            if depth_raw is not None and (
                depth_raw.shape[0] != frame.shape[0] or depth_raw.shape[1] != frame.shape[1]
            ):
                depth_map = cv2.resize(
                    depth_raw,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            depth_aligned = get_aligned_depth(capture, frame, args.use_transformed_depth)
            mp_o = make_mp_image_from_bgr(frame)
            frame_web = normalize_webcam_bgr(frame_web)
            mp_w = make_mp_image_from_bgr(frame_web)

            # VIDEO mode timestamps should be in ms and reflect the expected frame rate.
            # Use the same scheme as other pipelines (frame_idx * 1000/30) to keep temporal filters stable.
            mp_ts_o_ms = int(frame_idx * (1000 / 30))
            result_o = detect_for_video_safe(
                landmarker_o, mp_o, mp_ts_o_ms, warn_prefix="orbbec detect_for_video"
            )
            mp_ts_l_ms = int(frame_idx * (1000 / 30))
            result_l = detect_for_video_safe(
                landmarker_l, mp_w, mp_ts_l_ms, warn_prefix="webcam detect_for_video"
            )
            if result_o is None or result_l is None:
                continue

            debug_every = int(getattr(args, "handedness_debug_every", 0) or 0)
            if debug_every > 0 and (frame_idx % debug_every) == 0:
                def _summarize_handedness(tag: str, result) -> str:
                    if not getattr(result, "hand_landmarks", None):
                        return f"{tag}: no hand"
                    parts = []
                    for i in range(len(result.hand_landmarks)):
                        lbl = "?"
                        if getattr(result, "handedness", None) and i < len(result.handedness):
                            lbl = str(result.handedness[i][0].category_name)
                        # Wrist x in normalized image coords (if available)
                        try:
                            wx = float(result.hand_landmarks[i][0].x)
                            parts.append(f"{i}:{lbl}@x={wx:.2f}")
                        except Exception:
                            parts.append(f"{i}:{lbl}")
                    return f"{tag}: " + ", ".join(parts)

                print(_summarize_handedness("Orbbec", result_o))
                print(_summarize_handedness("Webcam", result_l))

            if (
                fusion_w >= 0.99
                and not args.use_transformed_depth
                and args.hand_3d == ob.HAND_3D_SOURCE_FUSED
                and not warned_fusion_linear_map
            ):
                print(
                    "[INFO] depth-fusion≈1 + --hand-3d fused: linear RGB→depth map; "
                    "try --hand-3d mp or --depth-patch 3–4."
                )
                warned_fusion_linear_map = True

            has_o = bool(result_o.hand_landmarks)
            has_l = bool(result_l.hand_landmarks)

            # Mode selection (like orbbec_main): LEFT hand controls morph mode (1..4).
            # Prefer Orbbec view if available; fallback to webcam view.
            idx_left_o, idx_right_o = (
                find_left_right_indices(result_o, invert_handedness=bool(args.invert_orbbec_handedness))
                if has_o
                else (None, None)
            )
            idx_left_l, idx_right_l = (
                find_left_right_indices(result_l, invert_handedness=bool(args.invert_webcam_handedness))
                if has_l
                else (None, None)
            )

            # Mode selection (like orbbec_main): LEFT hand controls morph mode (1..4).
            # Prefer Orbbec view if available; fallback to webcam view.
            idx_mode_o = idx_left_o
            pts_mode_o = (
                extract_world_points_mm_result(result_o, int(idx_mode_o))
                if idx_mode_o is not None
                else None
            )

            idx_mode_l = idx_left_l
            pts_mode_l = (
                extract_world_points_mm_result(result_l, int(idx_mode_l))
                if idx_mode_l is not None
                else None
            )

            pts_mode = pts_mode_o if pts_mode_o is not None else pts_mode_l
            mode_raw, tier_count = update_mode_state(
                pts_mode,
                mode_state=mode_state,
                classify_mode_fn=hm.classify_mode_from_fingers,
                debounce_frames=hm.MODE_DEBOUNCE_FRAMES,
                mode_smooth=0.22,
            )
            active_morph_mode = int(mode_state.morph_mode)

            dist_norm = (
                index_mcp_tip_segment_norm(pts_mode, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
                if pts_mode is not None
                else None
            )
            advance_lp_shape_p(dist_norm, active_morph_mode, lp_shape)

            # Role assignment: use MediaPipe handedness labels (Left/Right).
            # If a view is mirrored relative to the other, handedness may need swapping (see flags).
            idx_open_o = idx_right_o
            idx_open_l = idx_right_l

            pts_o = extract_world_points_mm_result(result_o, int(idx_open_o)) if idx_open_o is not None else None
            pts_l = extract_world_points_mm_result(result_l, int(idx_open_l)) if idx_open_l is not None else None

            vis_o = extract_landmark_visibilities(result_o, int(idx_open_o)) if idx_open_o is not None else None
            vis_l = extract_landmark_visibilities(result_l, int(idx_open_l)) if idx_open_l is not None else None
            summ_o = summarize_mp_visibility(vis_o)
            summ_l = summarize_mp_visibility(vis_l)

            w_geom_o, ev_o = (
                geom_weight_from_eigen_analysis(pts_o, ob.analyze_hand_topology)
                if pts_o is not None
                else (float("nan"), np.full(3, np.nan))
            )
            w_geom_l, ev_l = (
                geom_weight_from_eigen_analysis(pts_l, ob.analyze_hand_topology)
                if pts_l is not None
                else (float("nan"), np.full(3, np.nan))
            )

            fuse_mode = "none"
            w_mean_o = 0.0
            w_mean_l = 0.0
            fusion_dbg: Dict[str, Any] = {}
            fused_pts = None

            if (
                has_o
                and has_l
                and pts_o is not None
                and pts_l is not None
                and vis_o is not None
                and vis_l is not None
            ):
                fuse_mode = "dual"
                fused_pts, fusion_dbg = fuse_dual_views_weighted(
                    pts_o,
                    pts_l,
                    vis_o,
                    vis_l,
                    w_geom_o,
                    w_geom_l,
                    conf_low=fuse_conf_low,
                    conf_high=fuse_conf_high,
                )
                w_mean_o = float(fusion_dbg["w_mean_o"])
                w_mean_l = float(fusion_dbg["w_mean_l"])
            elif has_o and pts_o is not None:
                fuse_mode = "orbbec"
                fused_pts = list(pts_o)
                w_mean_o, w_mean_l = 1.0, 0.0
            elif has_l and pts_l is not None:
                fuse_mode = "webcam"
                fused_pts = list(pts_l)
                w_mean_o, w_mean_l = 0.0, 1.0

            if fuse_debug_every > 0 and frame_idx % fuse_debug_every == 0:

                def _fuse_dbg_side(prefix: str, pts, wg: float, ev: np.ndarray) -> str:
                    if pts is None:
                        return f"{prefix}:no_hand"
                    evs = np.round(ev, 2) if np.any(np.isfinite(ev)) else None
                    lam = f"lam={evs}" if evs is not None else "lam=—"
                    ws = f"w={float(wg):.3f}" if np.isfinite(float(wg)) else "w=—"
                    return f"{prefix} {ws} {lam}"

                print(
                    f"[fuse] {_fuse_dbg_side('O', pts_o, w_geom_o, ev_o)} | "
                    f"{_fuse_dbg_side('L', pts_l, w_geom_l, ev_l)} | "
                    f"meanW_O={w_mean_o:.3f} meanW_L={w_mean_l:.3f} "
                    f"hi={fusion_dbg.get('n_high_exclusive', 0)} "
                    f"low<{fuse_conf_low:.2f} high>={fuse_conf_high:.2f}"
                )

            hands_3d = [fused_pts] if fused_pts is not None else []

            frame_disp = frame.copy()
            frame_disp, _kp_o, ema_3d = ob.draw_hand(
                frame_disp,
                result_o,
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
                skip_wrist_labels=True,
            )
            frame_web_vis = frame_web.copy()
            frame_web_vis, _ = draw_hand_webcam(frame_web_vis, result_l, depth_map=None, print_depth=False)

            footer_l: List[Tuple[str, Tuple[int, int, int]]] = []
            if summ_l is not None and has_l:
                col_lm = confidence_color_bgr(summ_l["mean"])
                footer_l.append(
                    (
                        f"Webcam MP: mean={summ_l['mean']:.2f} min={summ_l['min']:.2f}",
                        col_lm,
                    )
                )
                wgl = f"{w_geom_l:.2f}" if np.isfinite(float(w_geom_l)) else "--"
                footer_l.append((f"w_geom={wgl}", (200, 200, 200)))
            elif has_l:
                footer_l.append(("Webcam: hand (no vis)", (180, 180, 180)))
            else:
                footer_l.append(("Webcam: no hand", (100, 100, 255)))

            overlay_inset(frame_disp, frame_web_vis, footer_lines=footer_l)

            y_hud = 22
            if has_o and has_l:
                cv2.putText(
                    frame_disp,
                    "Both views: hand detected",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (60, 255, 100),
                    2,
                    cv2.LINE_AA,
                )
            elif has_o or has_l:
                cv2.putText(
                    frame_disp,
                    "Single view (fusion prefers both)",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (80, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_disp,
                    "No hand",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (100, 100, 255),
                    2,
                    cv2.LINE_AA,
                )
            y_hud += 26
            if summ_o is not None and has_o:
                col_o = confidence_color_bgr(summ_o["mean"])
                cv2.putText(
                    frame_disp,
                    f"Orbbec MP: mean={summ_o['mean']:.2f} min={summ_o['min']:.2f}",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    col_o,
                    2,
                    cv2.LINE_AA,
                )
                y_hud += 22
                wgo = f"{w_geom_o:.2f}" if np.isfinite(float(w_geom_o)) else "--"
                cv2.putText(
                    frame_disp,
                    f"w_geom={wgo} (shape PCA)",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (210, 210, 210),
                    2,
                    cv2.LINE_AA,
                )
            elif has_o:
                cv2.putText(
                    frame_disp,
                    "Orbbec: hand (no vis)",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_disp,
                    "Orbbec: no hand",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 100, 255),
                    2,
                    cv2.LINE_AA,
                )

            open_out = None
            if hands_3d and hands_3d[0] is not None:
                tmp = ob.analyze_hand_topology(hands_3d[0])
                if tmp is not None:
                    if open_free_ema is None:
                        open_free_ema = float(tmp["morph_alpha"])
                    else:
                        open_free_ema = (
                            alpha_smooth * float(tmp["morph_alpha"]) + (1.0 - alpha_smooth) * open_free_ema
                        )

                    open_free = float(open_free_ema)
                    if snap_state == "plane":
                        if open_free < ob.PLANE_SNAP_OFF:
                            snap_state = None
                    elif snap_state == "sphere":
                        if open_free > ob.SPHERE_SNAP_OFF:
                            snap_state = None
                    else:
                        if open_free > ob.PLANE_SNAP_ON:
                            snap_state = "plane"
                        elif open_free < ob.SPHERE_SNAP_ON:
                            snap_state = "sphere"

                    open_out = open_free
                    if snap_state == "plane":
                        open_out = 1.0
                    elif snap_state == "sphere":
                        open_out = 0.0
                else:
                    open_out = None
            else:
                open_out = None

            if has_o and result_o.hand_landmarks:
                overlay_mode_open_wrist_labels(
                    frame=frame_disp,
                    result=result_o,
                    idx_left=idx_mode_o,
                    idx_right=idx_open_o,
                    morph_mode=active_morph_mode,
                    open_out=open_out,
                    overlay_wrist_labels_fn=ob.overlay_wrist_labels,
                )

            _wgo = f"{w_geom_o:.2f}" if np.isfinite(float(w_geom_o)) else "--"
            _wgl = f"{w_geom_l:.2f}" if np.isfinite(float(w_geom_l)) else "--"
            cv2.putText(
                frame_disp,
                    f"fuse:{fuse_mode} mO={w_mean_o:.2f} mL={w_mean_l:.2f} wgO={_wgo} wgL={_wgl}",
                (16, frame_disp.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if snap_state is None:
                snap_stable_frames = 0
                if snap_vis_state is not None:
                    snap_hold_frames += 1
                    if snap_hold_frames >= ob.SNAP_HOLD_AFTER_RELEASE_FRAMES:
                        snap_vis_state = None
                        snap_hold_frames = 0
            else:
                snap_hold_frames = 0
                if snap_vis_state == snap_state:
                    snap_stable_frames = min(ob.SNAP_SHOW_AFTER_FRAMES, snap_stable_frames + 1)
                else:
                    snap_stable_frames += 1
                    if snap_stable_frames >= ob.SNAP_SHOW_AFTER_FRAMES:
                        snap_vis_state = snap_state
                        snap_stable_frames = 0

            analyses = None
            if enable_3d and (frame_idx % PLOT_EVERY_N_FRAMES) == 0 and hands_3d:
                analyses = update_3d_plot_lp(
                    ax_hand,
                    ax_topo,
                    hands_3d,
                    morph_mode=active_morph_mode,
                    morph_alpha_smoothed=open_out,
                    control_label="open+p",
                    analyze_hand_topology_fn=ob.analyze_hand_topology,
                    clamp01_fn=clamp01,
                    shape_normalized=False,
                    hand_frame=ob.HAND_FRAME_METRIC_MM,
                    hand_3d_source=ob.HAND_3D_SOURCE_MP,
                    hand_frame_palm_plane=ob.HAND_FRAME_PALM_PLANE,
                    norm_axis_halflim=ob.NORM_AXIS_HALFLIM,
                    morph_axis_lim_mm=ob.MORPH_AXIS_LIM_MM,
                    mode_shape_t=lp_shape.left_shape_t_ema,
                    epsilon_pair_display=lp_shape.epsilon_pair_display,
                )
                plt.pause(0.0001)

            if analyses:
                a0 = analyses[0]
                topo_lbl = ob._topology_label_from_alpha(
                    float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"])
                )
                open_disp = open_out if open_out is not None else a0["morph_alpha"]
                free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                if open_remap is not None:
                    lo_r, hi_r = open_remap
                    open_disp = ob._remap_open_display(open_disp, lo_r, hi_r)
                    free_disp = ob._remap_open_display(free_disp, lo_r, hi_r)

                need_refresh = (frame_idx % ob.HUD_UPDATE_EVERY_N_FRAMES) == 0 or hud_cache["open"] is None
                if not need_refresh:
                    if abs(float(open_disp) - float(hud_cache["open"])) > ob.HUD_OPEN_STEP:
                        need_refresh = True
                    if abs(float(free_disp) - float(hud_cache["free"])) > ob.HUD_OPEN_STEP:
                        need_refresh = True
                    if abs(float(a0["planarity"]) - float(hud_cache["plan"])) > ob.HUD_METRIC_STEP:
                        need_refresh = True
                    if abs(float(a0["isotropy"]) - float(hud_cache["iso"])) > ob.HUD_METRIC_STEP:
                        need_refresh = True
                    if abs(float(a0["finger_spread"]) - float(hud_cache["spread"])) > ob.HUD_METRIC_STEP:
                        need_refresh = True

                if need_refresh:
                    hud_cache["open"] = float(open_disp)
                    hud_cache["free"] = float(free_disp)
                    hud_cache["plan"] = float(a0["planarity"])
                    hud_cache["iso"] = float(a0["isotropy"])
                    hud_cache["spread"] = float(a0["finger_spread"])
                    snap_txt = f"  SNAP:{snap_vis_state.upper()}" if snap_vis_state is not None else ""
                    hud_cache["text"] = build_modes_hud_lines(
                        morph_mode=active_morph_mode,
                        topo_label=f"{topo_lbl}{snap_txt}  [{fuse_mode}]",
                        open_disp=open_disp,
                        free_disp=free_disp,
                        spread=float(a0["finger_spread"]),
                        planarity=float(a0["planarity"]),
                        isotropy=float(a0["isotropy"]),
                    ) + [f"mO={w_mean_o:.2f}  mL={w_mean_l:.2f}"]

                if frame_idx % 30 == 0:
                    out_v = open_out if open_out is not None else a0["morph_alpha"]
                    free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                    print(
                        f"mode={fuse_mode} meanW_O={w_mean_o:.3f} meanW_L={w_mean_l:.3f} "
                        f"topology={a0['topology']} open_out={out_v:.3f} free={free_v:.3f} "
                        f"spread={a0['finger_spread']:.3f} "
                        f"planarity={a0['planarity']:.3f} isotropy={a0['isotropy']:.3f}"
                    )

            cv2.imshow("Hand Tracking Dual (Orbbec + Webcam)", frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                out_name = f"hand_3d_dual_frame_{frame_idx:06d}.png"
                fig.savefig(out_name, dpi=150, bbox_inches="tight")
                print(f"Saved 3D plot: {out_name}")
            if key == ord("p"):
                enable_3d = not enable_3d
                print(f"3D plot enabled: {enable_3d}")
            if key == ord("q"):
                break

            if hud_cache["text"] is not None:
                draw_hud(frame_disp, hud_cache["text"], origin=(16, 16))

            frame_idx += 1
    finally:
        landmarker_o.close()
        landmarker_l.close()
        k4a.stop()
        cap.release()
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
