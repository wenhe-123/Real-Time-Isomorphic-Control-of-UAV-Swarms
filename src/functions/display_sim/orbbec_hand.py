"""Orbbec RGB-D hand pipeline library for ``online_control.py`` / ``online_control_dual.py``.

MediaPipe landmarker helpers, depth fusion (:func:`fuse_hand_landmarks`), and 3D plot wrapper.
Standalone demo: ``backup/runtime/hand_tracking_orbbec_demo.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mediapipe as mp
import numpy as np

try:
    from pyk4a import Config, FPS, PyK4A
except ImportError as exc:
    raise ImportError(
        "pyk4a is missing (Linux: reinstall with pixi install / pip install -e '.'). "
        "Also run once: pixi run setup-orbbec  (Orbbec native libs under third_party/orbbec/)."
    ) from exc

from functions.display_sim.depth_fusion_utils import (
    ema_point_triplet as _shared_ema_point_triplet,
    fuse_cam_and_mp as _shared_fuse_cam_and_mp,
    mp_world_to_mm as _shared_mp_world_to_mm,
    read_depth_mm_at_landmark as _shared_read_depth_mm_at_landmark,
    reject_depth_outliers as _shared_reject_depth_outliers,
    transform_point_rigid_4x4_mm as _shared_transform_point_rigid_4x4_mm,
    unproject_to_depth_cam_mm as _shared_unproject_to_depth_cam_mm,
)
from functions.mode_switch.hand_constants import (
    FINGERTIP_IDS,
    HAND_CONNECTIONS,
    INDEX_MCP_ID,
    MCP_IDS,
    MIDDLE_MCP_ID,
    WRIST_ID,
)
from functions.mode_switch.hand_frame_utils import (
    metric_hand_to_palm_plane_normalized as _shared_metric_hand_to_palm_plane_normalized,
    metric_hand_to_shape_normalized as _shared_metric_hand_to_shape_normalized,
)
from functions.display_sim.orbbec_draw_steps import (
    build_mp_mm,
    compute_fused_raw,
    extract_points_and_depth,
    select_base_mm,
    select_viz_points,
    smooth_viz_points,
)
from functions.open_close.morph_lp_plot import (
    MORPH_LP_MESH_ETA,
    MORPH_LP_MESH_OMEGA,
    update_3d_plot_lp,
)
from functions.mode_switch.topology_utils import clamp01
from functions.mode_switch.webcam_mode_defaults import (
    HAND_3D_SOURCE_FUSED,
    HAND_3D_SOURCE_MP,
    HAND_FRAME_PALM_PLANE,
    HAND_FRAME_SCALED,
    MORPH_AXIS_LIM_MM,
    NORM_AXIS_HALFLIM,
    analyze_hand_topology,
)

# ===== MediaPipe initialize =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def resolve_mp_delegate(delegate: str) -> int:
    """Map a CLI delegate string to a MediaPipe ``BaseOptions.Delegate`` value.

    Args:
        delegate: ``"cpu"``, ``"gpu"``, ``"gl"``, or ``"opengl"``.

    Returns:
        ``BaseOptions.Delegate.CPU`` or ``BaseOptions.Delegate.GPU``.

    Raises:
        ValueError: If ``delegate`` is not recognized.
    """
    key = str(delegate).strip().lower()
    if key in ("gpu", "gl", "opengl"):
        return BaseOptions.Delegate.GPU
    if key in ("cpu", "default", ""):
        return BaseOptions.Delegate.CPU
    raise ValueError(f"unknown MediaPipe delegate {delegate!r}; use 'cpu' or 'gpu'")


def make_hand_landmarker_options(
    model_asset_path: str,
    *,
    delegate: str = "cpu",
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.55,
    min_hand_presence_confidence: float = 0.55,
    min_tracking_confidence: float = 0.55,
    running_mode: VisionRunningMode = VisionRunningMode.VIDEO,
) -> HandLandmarkerOptions:
    """Build MediaPipe ``HandLandmarkerOptions`` for video-mode hand tracking.

    Args:
        model_asset_path: Path to the ``hand_landmarker.task`` model file.
        delegate: MediaPipe delegate string (``"cpu"`` or ``"gpu"``).
        num_hands: Maximum number of hands to detect per frame.
        min_hand_detection_confidence: Minimum confidence to start a hand detection.
        min_hand_presence_confidence: Minimum confidence that a hand is present.
        min_tracking_confidence: Minimum confidence to keep tracking an existing hand.
        running_mode: MediaPipe vision running mode (default ``VIDEO``).

    Returns:
        Configured ``HandLandmarkerOptions`` instance.
    """
    return HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_asset_path),
            delegate=resolve_mp_delegate(delegate),
        ),
        running_mode=running_mode,
        num_hands=int(num_hands),
        min_hand_detection_confidence=float(min_hand_detection_confidence),
        min_hand_presence_confidence=float(min_hand_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )


def create_hand_landmarker(
    model_asset_path: str,
    *,
    delegate: str = "cpu",
    **kwargs,
) -> HandLandmarker:
    """Create a MediaPipe hand landmarker, falling back to CPU if GPU fails.

    Args:
        model_asset_path: Path to the ``hand_landmarker.task`` model file.
        delegate: Preferred delegate (``"cpu"`` or ``"gpu"``).
        **kwargs: Forwarded to :func:`make_hand_landmarker_options`.

    Returns:
        Initialized ``HandLandmarker`` instance.
    """
    del_key = str(delegate).strip().lower()
    try:
        return HandLandmarker.create_from_options(
            make_hand_landmarker_options(model_asset_path, delegate=del_key, **kwargs)
        )
    except Exception as exc:
        if del_key not in ("gpu", "gl", "opengl"):
            raise
        print(
            f"[WARN] MediaPipe GPU delegate failed ({exc}); falling back to CPU."
        )
        return HandLandmarker.create_from_options(
            make_hand_landmarker_options(model_asset_path, delegate="cpu", **kwargs)
        )

# Reject depth reads that hit background / wrong layer (linear RGB→depth map often mis-anchors palm edge).
DEPTH_ABS_MAX_MM = 1800.0
# Same-hand metric depth rarely differs by >~235 mm from wrist at desk distance (pinky mis-map ~230 mm).
DEPTH_MAX_DELTA_FROM_WRIST_MM = 235.0
# Second pass vs robust palm depth: only wrist + index + middle MCP (5,9). Ring/pinky MCP (13,17)
# often mis-hit on linear RGB→depth and must NOT define the reference median.
DEPTH_REF_ANCHOR_IDS = (WRIST_ID, INDEX_MCP_ID, MIDDLE_MCP_ID)
# Second pass: reject joints far from that reference (open hand + bad tips still caught).
DEPTH_MEDIAN_MAX_DELTA_MM = 175.0

# --- Depth camera metric 3D + fusion with MediaPipe ---
DEPTH_FUSION_WEIGHT = 0.55  # 1 = depth unproject only; 0 = MediaPipe world only
POINT_EMA_ALPHA = 0.28  # temporal smoothing on fused 3D (per keypoint, mm space)
DEPTH_MEDIAN_PATCH_RADIUS = 2


def _mp_world_to_mm(wlm):
    return _shared_mp_world_to_mm(wlm)


def _metric_hand_to_shape_normalized(points):
    return _shared_metric_hand_to_shape_normalized(points, wrist_id=WRIST_ID, mcp_ids=MCP_IDS, fingertip_ids=FINGERTIP_IDS)


def _metric_hand_to_palm_plane_normalized(points):
    return _shared_metric_hand_to_palm_plane_normalized(points, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)


def _transform_point_rigid_4x4_mm(p_xyz: tuple | None, T: np.ndarray | None) -> tuple | None:
    return _shared_transform_point_rigid_4x4_mm(p_xyz, T)


def _fuse_cam_and_mp(p_cam, p_mp, fusion_weight: float):
    return _shared_fuse_cam_and_mp(p_cam, p_mp, fusion_weight)


def _ema_point_triplet(prev, cur, alpha: float):
    return _shared_ema_point_triplet(prev, cur, alpha)


def _read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, patch_r: int = DEPTH_MEDIAN_PATCH_RADIUS):
    return _shared_read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, patch_r)


def _unproject_to_depth_cam_mm(
    calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw
):
    return _shared_unproject_to_depth_cam_mm(calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw)


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


def fuse_hand_landmarks(
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
    draw_skeleton: bool = False,
):
    """Fuse Orbbec depth and MediaPipe landmarks into per-hand 21×3 mm keypoints.

    Args:
        frame: BGR image; modified in place only when ``draw_skeleton`` is ``True``.
        result: MediaPipe ``HandLandmarkerResult`` for the current frame.
        depth_raw: Native depth image in mm, or ``None``.
        depth_aligned: Depth registered to the color frame, or ``None``.
        print_depth: When ``True``, print depth values to stdout (debug overlay only).
        calibration: PyK4A calibration for unprojection, or ``None``.
        fusion_weight: Blend weight between depth and MediaPipe world coords.
        ema_alpha: Temporal EMA smoothing factor for 3D keypoints.
        ema_points: Per-hand previous EMA state, or ``None``.
        depth_patch_radius: Median patch radius for depth reads.
        hand_frame: Coordinate frame for visualization (scaled, palm-plane, or raw mm).
        filter_depth_outliers: When ``True``, apply wrist/anchor outlier rejection.
        depth_max_delta_mm: Max deviation from wrist depth during outlier rejection.
        depth_median_max_delta_mm: Second-pass median threshold, or ``None`` to disable.
        hand_3d_source: ``"mp"`` or ``"fused"`` source selection for 3D points.
        depth_unproject_rigid_T: Optional 4×4 rigid correction after unprojection.
        skip_wrist_labels: When ``True``, omit handedness labels at the wrist (debug overlay).
        draw_skeleton: When ``True``, draw debug 2D skeleton via :mod:`debug.orbbec_hand_draw`.

    Returns:
        ``(frame, keypoints_3d, ema_out)`` where ``keypoints_3d`` is a list of per-hand
        21×3 mm point lists and ``ema_out`` holds updated EMA state per hand.
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
        mp_only_3d = (
            str(hand_3d_source).strip().lower() != HAND_3D_SOURCE_FUSED
            and depth_raw is None
            and depth_aligned is None
        )
        if mp_only_3d:
            points = [
                (
                    int(np.clip(int(lm.x * w), 0, w - 1)),
                    int(np.clip(int(lm.y * h), 0, h - 1)),
                )
                for lm in hand_landmarks
            ]
            depth_vals = [None] * len(points)
            fused_raw = mp_mm
        else:
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

        hand_ema_in = ema_points[idx] if ema_points is not None and idx < len(ema_points) else None

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

        if draw_skeleton:
            from debug.orbbec_hand_draw import draw_hand_2d_overlay

            handed_label = result.handedness[idx][0].category_name if result.handedness else None
            draw_hand_2d_overlay(
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
    show_sample_ids: bool = False,
    mesh_n_eta: int = MORPH_LP_MESH_ETA,
    mesh_n_omega: int = MORPH_LP_MESH_OMEGA,
    shape_normalized: bool = False,
    hand_frame: str = HAND_FRAME_SCALED,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
    topo_radius_override_mm: Optional[float] = None,
    control_label: str = "",
):
    """Update the dual 3D hand/topology matplotlib plot for online control.

    Thin wrapper around :func:`functions.open_close.morph_lp_plot.update_3d_plot_lp`
    with Orbbec runtime defaults.

    Args:
        ax_hand: Matplotlib 3D axis for the hand skeleton, or ``None``.
        ax_topo: Matplotlib 3D axis for morph topology.
        hands_3d: List of per-hand 21×3 mm keypoint arrays.
        morph_alpha_smoothed: Smoothed open-hand scalar for morph visualization.
        morph_mode: Active morph mode index (1–5).
        mode_shape_t: Left-hand shape parameter for LP morph display.
        epsilon_pair_display: Optional epsilon pair for debug overlay.
        lp_show_refs: When ``True``, draw LP reference geometry.
        show_sample_ids: When ``True``, label morph sample indices.
        mesh_n_eta: LP mesh resolution along eta.
        mesh_n_omega: LP mesh resolution along omega.
        shape_normalized: When ``True``, hand points are shape-normalized.
        hand_frame: Hand coordinate frame name (scaled, palm-plane, or raw).
        hand_3d_source: 3D source tag (``"mp"`` or ``"fused"``).
        topo_radius_override_mm: Optional fixed topology axis radius in mm.
        control_label: Short label for the control source in the plot title.

    Returns:
        Return value from :func:`update_3d_plot_lp` (typically ``None``).
    """
    return update_3d_plot_lp(
        ax_hand,
        ax_topo,
        hands_3d,
        morph_mode=morph_mode,
        morph_alpha_smoothed=morph_alpha_smoothed,
        control_label=control_label,
        analyze_hand_topology_fn=analyze_hand_topology,
        clamp01_fn=clamp01,
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
        show_sample_ids=show_sample_ids,
        mesh_n_eta=mesh_n_eta,
        mesh_n_omega=mesh_n_omega,
        topo_radius_override_mm=topo_radius_override_mm,
    )
