"""
Single-camera Orbbec RGB-D hand tracking core implementation.

This module provides the main Orbbec pipeline used by apps/pipelines:
- MediaPipe hand detection/tracking
- RGB + depth fusion into stable 3D hand points
- topology metrics (planarity/spread/isotropy) and morph alpha
- OpenCV HUD + optional matplotlib 3D visualization

Entry function: `main()`.
"""

import argparse
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pyk4a import Config, FPS, PyK4A
from shared.common_utils import draw_hud
from shared.depth_fusion_utils import (
    ema_point_triplet as _shared_ema_point_triplet,
    fuse_cam_and_mp as _shared_fuse_cam_and_mp,
    map_color_pixel_to_depth_pixel as _shared_map_color_pixel_to_depth_pixel,
    median_valid_depth_mm as _shared_median_valid_depth_mm,
    mp_world_to_mm as _shared_mp_world_to_mm,
    read_depth_mm_at_landmark as _shared_read_depth_mm_at_landmark,
    reject_depth_outliers as _shared_reject_depth_outliers,
    transform_point_rigid_4x4_mm as _shared_transform_point_rigid_4x4_mm,
    unproject_color_aligned_to_depth_camera_mm as _shared_unproject_color_aligned_to_depth_camera_mm,
    unproject_depth_pixel_to_depth_camera_mm as _shared_unproject_depth_pixel_to_depth_camera_mm,
    unproject_to_depth_cam_mm as _shared_unproject_to_depth_cam_mm,
)
from shared.hand_constants import FINGERTIP_IDS, HAND_CONNECTIONS, MCP_IDS, WRIST_ID
from shared.hand_frame_utils import (
    metric_hand_to_palm_plane_normalized as _shared_metric_hand_to_palm_plane_normalized,
    metric_hand_to_shape_normalized as _shared_metric_hand_to_shape_normalized,
    palm_plane_basis_from_world as _shared_palm_plane_basis_from_world,
    palm_plane_curl_metrics as _shared_palm_plane_curl_metrics,
)
from shared.orbbec_draw_steps import (
    build_mp_mm,
    compute_fused_raw,
    draw_2d_overlay,
    extract_points_and_depth,
    select_base_mm,
    select_viz_points,
    smooth_viz_points,
)
from shared.morph_lp_plot import (
    MORPH_LP_MESH_ETA,
    MORPH_LP_MESH_OMEGA,
    update_3d_plot_lp,
)
from shared.orbbec_live_steps import (
    compute_open_out as _shared_compute_open_out,
    print_periodic_topology_status as _shared_print_periodic_topology_status,
    refresh_hud_cache as _shared_refresh_hud_cache,
    update_snap_visual_state as _shared_update_snap_visual_state,
)
from shared.stream_runtime_utils import (
    capture_orbbec_frame,
    detect_for_video_safe,
    get_aligned_depth,
    make_mp_image_from_bgr,
    safe_get_capture,
)
from shared.mp_hand_utils import find_hand_index_by_side
from shared.modes_runtime import ModeState, RightHandState, process_left_mode, process_right_open
from shared.morph_renderers import prompt_and_init_fixed_surface_points
from shared.morph_shape_control import (
    LpShapePipelineState,
    advance_lp_shape_p,
    index_mcp_tip_segment_norm,
)
from shared.topology_utils import (
    analyze_hand_topology_common,
    clamp01,
    remap_open_display,
    safe_normalize,
    topology_label_from_alpha,
)

# ===== MediaPipe initialize =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Index/middle/ring/pinky tips only — thumb often stays extended in a fist and inflates “open” spread.
FINGERTIP_IDS_FOUR = [8, 12, 16, 20]

# Nonlinear gain for morph sensitivity:
# open in [0,1] (0=sphere/fist, 1=plane/open). open**GAMMA makes fist side "more spherical".
OPEN_GAMMA = 1.8
# Labels / SNAP: same banding as the original MP-world script (plane > 0.67, sphere < 0.33 on instant alpha).
TOPO_ALPHA_PLANE = 0.67
TOPO_ALPHA_SPHERE = 0.33

# Reject depth reads that hit background / wrong layer (linear RGB→depth map often mis-anchors palm edge).
DEPTH_ABS_MAX_MM = 1800.0
# Same-hand metric depth rarely differs by >~235 mm from wrist at desk distance (pinky mis-map ~230 mm).
DEPTH_MAX_DELTA_FROM_WRIST_MM = 235.0
# Second pass vs robust palm depth: only wrist + index + middle MCP (5,9). Ring/pinky MCP (13,17)
# often mis-hit on linear RGB→depth and must NOT define the reference median.
DEPTH_REF_ANCHOR_IDS = (WRIST_ID, 5, 9)
# Second pass: reject joints far from that reference (open hand + bad tips still caught).
DEPTH_MEDIAN_MAX_DELTA_MM = 175.0

# Empirical raw morph_alpha range with shape_norm (fist ~0.22, open ~0.72): map to [0,1] for blanket + HUD.
OPEN_REMAP_LO = 0.22
OPEN_REMAP_HI = 0.72

# 3D skeleton + topology: MP-only keeps a coherent hand; fused mixes depth frame + MP world → often skewed.
HAND_3D_SOURCE_MP = "mp"
HAND_3D_SOURCE_FUSED = "fused"
# Coordinate frame for 3D output (after metric base_mm from MP or fused).
HAND_FRAME_SCALED = "scaled"  # wrist origin + isotropic palm scale (legacy)
HAND_FRAME_PALM_PLANE = "palm_plane"  # wrist origin; XY = plane(wrist, index MCP, middle MCP); then / palm scale
HAND_FRAME_METRIC_MM = "metric_mm"  # raw mm in camera/world frame from base_mm

# Keep morph axis scale fixed (only the morph surface changes size).
MORPH_AXIS_LIM_MM = 200.0
# Wrist-centered + palm-scale normalization: 3D plot limits (unit hand size).
NORM_AXIS_HALFLIM = 1.35

# Performance / real-time controls
PLOT_EVERY_N_FRAMES = 5  # update matplotlib every N frames
ENABLE_3D_PLOT = True    # press 'p' to toggle at runtime

# Snap-to-canonical plane/sphere (EMA raw_free); hysteresis matches original standalone script.
PLANE_SNAP_ON = 0.88
PLANE_SNAP_OFF = 0.82
SPHERE_SNAP_ON = 0.12
SPHERE_SNAP_OFF = 0.18

# 2D HUD anti-flicker
HUD_UPDATE_EVERY_N_FRAMES = 10
HUD_OPEN_STEP = 0.03
HUD_METRIC_STEP = 0.05

# SNAP display debounce (avoid visible blinking)
SNAP_SHOW_AFTER_FRAMES = 6
SNAP_HOLD_AFTER_RELEASE_FRAMES = 10

# --- Depth camera metric 3D + fusion with MediaPipe ---
# Depth unproject → 3D in **depth camera** frame (mm). MediaPipe world → wrist-centric pseudo-metric (mm).
# IMPORTANT: Never do wrist_cam_depth + (p_mp - wrist_mp): those vectors live in different frames and
# will scramble the skeleton. Per-joint blend only: w * p_depth + (1-w) * p_mediapipe (see _fuse_cam_and_mp).
DEPTH_FUSION_WEIGHT = 0.55  # 1 = depth unproject only; 0 = MediaPipe world only
POINT_EMA_ALPHA = 0.28  # temporal smoothing on fused 3D (per keypoint, mm space)
# Median depth in a (2r+1)^2 patch on the depth image reduces single-pixel noise (critical for fusion≈1).
DEPTH_MEDIAN_PATCH_RADIUS = 2

# Femto Bolt is factory-calibrated; Orbbec SDK (and this K4A wrapper) can expose vision + IMU calibration
# programmatically. After PyK4A.open/start, `k4a.calibration` is the K4A-style blob used by
# `convert_2d_to_3d` for depth/color intrinsics and depth↔color extrinsics. IMU is separate from RGB-D
# unprojection in this script. If unprojection looks wrong, some wrappers still ship placeholder blobs —
# compare with OrbbecViewer / native SDK export or Orbbec docs for your firmware.


def _mp_world_to_mm(wlm):
    return _shared_mp_world_to_mm(wlm)


def _map_color_pixel_to_depth_pixel(x_c: int, y_c: int, w_c: int, h_c: int, w_d: int, h_d: int):
    return _shared_map_color_pixel_to_depth_pixel(x_c, y_c, w_c, h_c, w_d, h_d)


def _unproject_depth_pixel_to_depth_camera_mm(calibration, x_d: float, y_d: float, depth_mm: float):
    return _shared_unproject_depth_pixel_to_depth_camera_mm(calibration, x_d, y_d, depth_mm)


def _unproject_color_aligned_to_depth_camera_mm(calibration, x_c: float, y_c: float, depth_mm: float):
    return _shared_unproject_color_aligned_to_depth_camera_mm(calibration, x_c, y_c, depth_mm)


def _metric_hand_to_shape_normalized(points):
    return _shared_metric_hand_to_shape_normalized(points, wrist_id=WRIST_ID, mcp_ids=MCP_IDS, fingertip_ids=FINGERTIP_IDS)


def _palm_plane_basis_from_world(points):
    return _shared_palm_plane_basis_from_world(points, wrist_id=WRIST_ID)


def _metric_hand_to_palm_plane_normalized(points):
    return _shared_metric_hand_to_palm_plane_normalized(points, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)


def _palm_plane_curl_metrics(points_21):
    return _shared_palm_plane_curl_metrics(points_21, fingertip_ids_four=FINGERTIP_IDS_FOUR)


def load_depth_unproject_rigid_npy(path: str | None) -> np.ndarray | None:
    """Load optional 4×4 (float64) rigid transform in mm (homogeneous) for depth-unprojected points."""
    if not path:
        return None
    try:
        T = np.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"--depth-unproject-rigid-npy: file not found: {path!r}. "
            "Use a real 4×4 float64 .npy from calibration, or omit this flag (documentation used /path/to/ as placeholder)."
        ) from e
    if getattr(T, "shape", None) != (4, 4):
        raise ValueError(f"--depth-unproject-rigid-npy must be 4×4, got {getattr(T, 'shape', None)}")
    return np.asarray(T, dtype=np.float64)


def _transform_point_rigid_4x4_mm(p_xyz: tuple | None, T: np.ndarray | None) -> tuple | None:
    return _shared_transform_point_rigid_4x4_mm(p_xyz, T)


def _fuse_cam_and_mp(p_cam, p_mp, fusion_weight: float):
    return _shared_fuse_cam_and_mp(p_cam, p_mp, fusion_weight)


def _ema_point_triplet(prev, cur, alpha: float):
    return _shared_ema_point_triplet(prev, cur, alpha)


def _median_valid_depth_mm(depth_img: np.ndarray, u: int, v: int, patch_r: int):
    return _shared_median_valid_depth_mm(depth_img, u, v, patch_r)


def _read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, patch_r: int = DEPTH_MEDIAN_PATCH_RADIUS):
    return _shared_read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, patch_r)


def _unproject_to_depth_cam_mm(
    calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw
):
    return _shared_unproject_to_depth_cam_mm(calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw)


# ===== draw keypoints =====
def draw_hand(
    frame,
    result,
    depth_raw=None,
    depth_aligned=None,
    print_depth=False,
    *,
    calibration=None,
    fusion_weight: float = DEPTH_FUSION_WEIGHT,
    ema_alpha: float = POINT_EMA_ALPHA,
    ema_points=None,
    depth_patch_radius: int = DEPTH_MEDIAN_PATCH_RADIUS,
    hand_frame: str = HAND_FRAME_SCALED,
    filter_depth_outliers: bool = True,
    depth_max_delta_mm: float = DEPTH_MAX_DELTA_FROM_WRIST_MM,
    depth_median_max_delta_mm: float | None = DEPTH_MEDIAN_MAX_DELTA_MM,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
    depth_unproject_rigid_T: np.ndarray | None = None,
    skip_wrist_labels: bool = False,
):
    """
    depth_raw: native depth image (H_d,W_d), uint16 mm — used on Orbbec/Femto (no transformed_depth).
    depth_aligned: optional depth warped to color (same H,W as frame); only if SDK supports it.
    calibration: pyk4a Calibration for metric unproject.

    Output 3D is fused (depth camera mm + MediaPipe) then EMA-smoothed per keypoint.
    depth_patch_radius: median depth over (2r+1)^2 in depth image (reduces noise when fusion≈1).
    hand_frame: scaled | palm_plane | metric_mm — see --hand-frame.
    filter_depth_outliers: drop per-joint depths far from wrist / absurd mm → fusion uses MP for those joints.
    hand_3d_source: "mp" = 3D from MediaPipe world only (recommended); "fused" = depth+MP per joint (fragile).
    depth_unproject_rigid_T: optional 4×4 (mm) applied to each depth-unprojected point before fusion (see --depth-unproject-rigid-npy).
    skip_wrist_labels: if True, do not draw Left/Right at the wrist (use overlay_wrist_labels after computing role strings).
    """
    keypoints_3d = []
    all_ema_out = []

    if not result.hand_landmarks:
        return frame, keypoints_3d, ema_points

    h, w, _ = frame.shape

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        points = []
        points_3d = []
        world_landmarks = None
        if hasattr(result, "hand_world_landmarks") and len(result.hand_world_landmarks) > idx:
            world_landmarks = result.hand_world_landmarks[idx]

        mp_mm = build_mp_mm(world_landmarks, 21, _mp_world_to_mm)
        points, depth_vals = extract_points_and_depth(
            hand_landmarks,
            h,
            w,
            lambda x, y, hh, ww: _read_depth_mm_at_landmark(
                x, y, hh, ww, depth_aligned, depth_raw, depth_patch_radius
            ),
        )

        if filter_depth_outliers:
            depth_vals = _reject_depth_outliers(
                depth_vals,
                max_delta_mm=depth_max_delta_mm,
                median_max_delta_mm=depth_median_max_delta_mm,
            )

        hand_ema_in = ema_points[idx] if ema_points is not None and idx < len(ema_points) else None

        fused_raw = compute_fused_raw(
            points=points,
            depth_vals=depth_vals,
            mp_mm=mp_mm,
            calibration=calibration,
            h=h,
            w=w,
            depth_aligned=depth_aligned,
            depth_raw=depth_raw,
            depth_unproject_rigid_T=depth_unproject_rigid_T,
            fusion_weight=fusion_weight,
            unproject_to_depth_cam_mm=_unproject_to_depth_cam_mm,
            transform_point_rigid_4x4_mm=_transform_point_rigid_4x4_mm,
            fuse_cam_and_mp=_fuse_cam_and_mp,
        )

        base_mm = select_base_mm(
            hand_3d_source=hand_3d_source,
            hand_3d_source_fused=HAND_3D_SOURCE_FUSED,
            world_landmarks=world_landmarks,
            mp_mm=mp_mm,
            fused_raw=fused_raw,
        )
        viz_pts = select_viz_points(
            hand_frame=hand_frame,
            hand_frame_palm_plane=HAND_FRAME_PALM_PLANE,
            hand_frame_scaled=HAND_FRAME_SCALED,
            base_mm=base_mm,
            metric_hand_to_palm_plane_normalized=_metric_hand_to_palm_plane_normalized,
            metric_hand_to_shape_normalized=_metric_hand_to_shape_normalized,
        )

        norm_depth_label = hand_frame in (HAND_FRAME_SCALED, HAND_FRAME_PALM_PLANE)

        hand_ema_out = smooth_viz_points(viz_pts, hand_ema_in, ema_alpha, _ema_point_triplet)
        points_3d = list(hand_ema_out)

        all_ema_out.append(hand_ema_out)

        handed_label = result.handedness[idx][0].category_name if result.handedness else None
        draw_2d_overlay(
            frame,
            idx=idx,
            hand_landmarks=hand_landmarks,
            points=points,
            depth_vals=depth_vals,
            norm_depth_label=norm_depth_label,
            print_depth=print_depth,
            draw_wrist_label=not skip_wrist_labels,
            handed_label=handed_label,
            hand_connections=HAND_CONNECTIONS,
        )
        keypoints_3d.append(points_3d)

    return frame, keypoints_3d, all_ema_out


def overlay_wrist_labels(frame, result, labels_by_idx: dict, *, font_scale: float = 0.72):
    """
    Draw short strings near wrist (landmark 0). ``labels_by_idx`` maps hand index
    (same order as ``result.hand_landmarks``) to text, e.g. ``M2`` / ``open 0.73``.
    """
    if not result.hand_landmarks or not labels_by_idx:
        return frame
    h, w, _ = frame.shape
    for idx, hand_lms in enumerate(result.hand_landmarks):
        if idx not in labels_by_idx:
            continue
        lm0 = hand_lms[0]
        px = int(np.clip(int(lm0.x * w), 0, w - 1))
        py = int(np.clip(int(lm0.y * h), 0, h - 1))
        cv2.putText(
            frame,
            labels_by_idx[idx],
            (px, max(24, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def _safe_normalize(v):
    return safe_normalize(v)


def _clamp01(x):
    return clamp01(x)


def _topology_label_from_alpha(alpha: float) -> str:
    """Coarse label from continuous morph_alpha (use EMA for HUD so it matches raw_free / SNAP)."""
    return topology_label_from_alpha(alpha, plane_thr=TOPO_ALPHA_PLANE, sphere_thr=TOPO_ALPHA_SPHERE)


def _remap_open_display(alpha: float, lo: float, hi: float) -> float:
    """Linear stretch of raw morph_alpha to [0, 1] for visualization (empirical fist/open range)."""
    return remap_open_display(alpha, lo, hi)


def _reject_depth_outliers(
    depth_vals,
    *,
    max_delta_mm: float = DEPTH_MAX_DELTA_FROM_WRIST_MM,
    median_max_delta_mm: float | None = DEPTH_MEDIAN_MAX_DELTA_MM,
):
    return _shared_reject_depth_outliers(
        depth_vals,
        depth_abs_max_mm=DEPTH_ABS_MAX_MM,
        max_delta_mm=max_delta_mm,
        median_max_delta_mm=median_max_delta_mm,
        depth_ref_anchor_ids=DEPTH_REF_ANCHOR_IDS,
        wrist_id=WRIST_ID,
    )


def analyze_hand_topology(hand_points):
    return analyze_hand_topology_common(
        hand_points,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
        fingertip_ids=FINGERTIP_IDS,
        open_gamma=OPEN_GAMMA,
        label_fn=lambda a: _topology_label_from_alpha(a),
    )


def update_3d_plot(
    ax_hand,
    ax_topo,
    hands_3d,
    morph_alpha_smoothed=None,
    *,
    morph_mode: int = 1,
    mode_shape_t: Optional[float] = None,
    epsilon_pair_display: Optional[Tuple[float, float]] = None,
    lp_show_refs: bool = True,
    mesh_n_eta: int = MORPH_LP_MESH_ETA,
    mesh_n_omega: int = MORPH_LP_MESH_OMEGA,
    shape_normalized: bool = False,
    hand_frame: str = HAND_FRAME_SCALED,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
):
    return update_3d_plot_lp(
        ax_hand,
        ax_topo,
        hands_3d,
        morph_mode=morph_mode,
        morph_alpha_smoothed=morph_alpha_smoothed,
        control_label="",
        analyze_hand_topology_fn=analyze_hand_topology,
        clamp01_fn=_clamp01,
        shape_normalized=shape_normalized,
        hand_frame=hand_frame,
        hand_3d_source=hand_3d_source,
        hand_frame_palm_plane=HAND_FRAME_PALM_PLANE,
        norm_axis_halflim=NORM_AXIS_HALFLIM,
        morph_axis_lim_mm=MORPH_AXIS_LIM_MM,
        hand_connections=HAND_CONNECTIONS,
        mode_shape_t=mode_shape_t,
        epsilon_pair_display=epsilon_pair_display,
        lp_show_refs=lp_show_refs,
        mesh_n_eta=mesh_n_eta,
        mesh_n_omega=mesh_n_omega,
    )


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
        default=DEPTH_FUSION_WEIGHT,
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
        choices=(HAND_FRAME_SCALED, HAND_FRAME_PALM_PLANE, HAND_FRAME_METRIC_MM),
        default=HAND_FRAME_SCALED,
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
        choices=(HAND_3D_SOURCE_MP, HAND_3D_SOURCE_FUSED),
        default=HAND_3D_SOURCE_MP,
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
    return ap


def _normalize_main_args(args):
    model_path = args.model
    fusion_w = float(np.clip(args.depth_fusion, 0.0, 1.0))
    ema_a = float(np.clip(args.ema_alpha, 0.0, 1.0))
    depth_patch_r = int(np.clip(args.depth_patch, 0, 15))
    if args.hand_frame != HAND_FRAME_SCALED:
        hand_frame = args.hand_frame
    elif args.no_shape_normalize:
        hand_frame = HAND_FRAME_METRIC_MM
    else:
        hand_frame = HAND_FRAME_SCALED
    shape_norm = hand_frame in (HAND_FRAME_SCALED, HAND_FRAME_PALM_PLANE)
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
        and args.hand_3d == HAND_3D_SOURCE_FUSED
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
            plane_snap_on=PLANE_SNAP_ON,
            plane_snap_off=PLANE_SNAP_OFF,
            sphere_snap_on=SPHERE_SNAP_ON,
            sphere_snap_off=SPHERE_SNAP_OFF,
        )
        hands_3d_topo = hands_3d
    snap_vis_state, snap_stable_frames, snap_hold_frames = _shared_update_snap_visual_state(
        snap_state=snap_state,
        snap_vis_state=snap_vis_state,
        snap_stable_frames=snap_stable_frames,
        snap_hold_frames=snap_hold_frames,
        snap_show_after_frames=SNAP_SHOW_AFTER_FRAMES,
        snap_hold_after_release_frames=SNAP_HOLD_AFTER_RELEASE_FRAMES,
    )

    analyses = None
    if enable_3d and (frame_idx % PLOT_EVERY_N_FRAMES) == 0 and hands_3d_topo:
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
            hand_frame_palm_plane=HAND_FRAME_PALM_PLANE,
            open_out=open_out,
            open_free_ema=open_free_ema,
            open_remap=open_remap,
            snap_vis_state=snap_vis_state,
            hud_update_every_n_frames=HUD_UPDATE_EVERY_N_FRAMES,
            hud_open_step=HUD_OPEN_STEP,
            hud_metric_step=HUD_METRIC_STEP,
            topology_label_fn=_topology_label_from_alpha,
            remap_open_display_fn=_remap_open_display,
            palm_plane_curl_metrics_fn=_palm_plane_curl_metrics,
        )
        _shared_print_periodic_topology_status(
            frame_idx=frame_idx,
            a0=a0,
            hands_3d=hands_3d_topo,
            hand_frame=hand_frame,
            hand_frame_palm_plane=HAND_FRAME_PALM_PLANE,
            open_out=open_out,
            open_free_ema=open_free_ema,
            open_remap=open_remap,
            topology_label_fn=_topology_label_from_alpha,
            remap_open_display_fn=_remap_open_display,
            palm_plane_curl_metrics_fn=_palm_plane_curl_metrics,
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
            "q=quit  p=3D  s=save  |  L=mode+Lp (index segment)  R=open  (use pipelines/orbbec_main.py for full HUD)"
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
            enable_3d = ENABLE_3D_PLOT
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


if __name__ == "__main__":
    main()
