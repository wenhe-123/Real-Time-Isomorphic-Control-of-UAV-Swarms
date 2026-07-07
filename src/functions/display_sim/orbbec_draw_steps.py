"""Orbbec per-frame hand pipeline: depth reads, MP mm coords, fused raw, base selection, smoothed viz points."""

from __future__ import annotations

import numpy as np


def extract_points_and_depth(hand_landmarks, h: int, w: int, depth_reader):
    """Convert MediaPipe landmarks to color pixels and read depth at each joint.

    Args:
        hand_landmarks: Iterable of MediaPipe normalized hand landmarks.
        h: Color image height in pixels.
        w: Color image width in pixels.
        depth_reader: Callable ``(x, y, h, w) -> depth_mm | None`` for each pixel.

    Returns:
        ``(points, depth_vals)`` where ``points`` is a list of ``(x, y)`` tuples and
        ``depth_vals`` holds the corresponding depth in mm (or ``None``).
    """
    points = []
    depth_vals = []
    for lm in hand_landmarks:
        x = int(np.clip(int(lm.x * w), 0, w - 1))
        y = int(np.clip(int(lm.y * h), 0, h - 1))
        points.append((x, y))
        depth_vals.append(depth_reader(x, y, h, w))
    return points, depth_vals


def build_mp_mm(world_landmarks, n_kp: int, mp_world_to_mm):
    """Build per-keypoint MediaPipe world coordinates in millimeters.

    Args:
        world_landmarks: MediaPipe world landmark list, or ``None``.
        n_kp: Number of keypoints to produce (typically 21).
        mp_world_to_mm: Callable converting one world landmark to an ``(x, y, z)`` mm tuple.

    Returns:
        List of ``n_kp`` ``(x, y, z)`` tuples; missing landmarks are ``(nan, nan, nan)``.
    """
    if world_landmarks is None:
        return [(np.nan, np.nan, np.nan)] * n_kp
    out = []
    for kp_id in range(n_kp):
        if kp_id < len(world_landmarks):
            out.append(mp_world_to_mm(world_landmarks[kp_id]))
        else:
            out.append((np.nan, np.nan, np.nan))
    return out


def compute_fused_raw(
    *,
    points,
    depth_vals,
    mp_mm,
    calibration,
    h: int,
    w: int,
    depth_aligned,
    depth_raw,
    depth_unproject_rigid_T,
    fusion_weight: float,
    unproject_to_depth_cam_mm,
    transform_point_rigid_4x4_mm,
    fuse_cam_and_mp,
):
    """Fuse depth-unprojected and MediaPipe mm coordinates for each keypoint.

    Args:
        points: Color pixel ``(x, y)`` list per keypoint.
        depth_vals: Depth in mm per keypoint (parallel to ``points``).
        mp_mm: MediaPipe world coordinates in mm per keypoint.
        calibration: PyK4A calibration for unprojection, or ``None``.
        h: Color image height in pixels.
        w: Color image width in pixels.
        depth_aligned: Depth registered to the color frame, or ``None``.
        depth_raw: Native depth image, or ``None``.
        depth_unproject_rigid_T: Optional 4×4 rigid correction after unprojection.
        fusion_weight: Blend weight between depth and MediaPipe (see :func:`fuse_cam_and_mp`).
        unproject_to_depth_cam_mm: Unprojection callable for each landmark.
        transform_point_rigid_4x4_mm: Rigid transform callable applied after unprojection.
        fuse_cam_and_mp: Fusion callable blending camera and MP points.

    Returns:
        List of fused ``(x, y, z)`` tuples in mm, one per keypoint.
    """
    fused_raw = []
    for kp_id in range(len(points)):
        x, y = points[kp_id]
        depth_mm = depth_vals[kp_id]
        p_cam = None
        if depth_mm is not None and calibration is not None:
            p_cam = unproject_to_depth_cam_mm(calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw)
            p_cam = transform_point_rigid_4x4_mm(p_cam, depth_unproject_rigid_T)
        p_mp = mp_mm[kp_id]
        fused_raw.append(fuse_cam_and_mp(p_cam, p_mp, fusion_weight))
    return fused_raw


def select_base_mm(*, hand_3d_source, hand_3d_source_fused, world_landmarks, mp_mm, fused_raw):
    """Choose the raw mm keypoint source before frame normalization.

    Args:
        hand_3d_source: Active source tag (``"mp"`` or ``"fused"``).
        hand_3d_source_fused: Constant for the fused source tag.
        world_landmarks: MediaPipe world landmarks, or ``None``.
        mp_mm: MediaPipe coordinates in mm per keypoint.
        fused_raw: Depth-fused coordinates in mm per keypoint.

    Returns:
        List of 21 ``(x, y, z)`` mm tuples from the selected source; falls back to
        ``fused_raw`` when MP data is incomplete.
    """
    if hand_3d_source == hand_3d_source_fused:
        return fused_raw
    if world_landmarks is not None:
        base_mm = [tuple(mp_mm[i]) for i in range(21)]
        if not np.all(np.isfinite(np.array(base_mm, dtype=float))):
            return fused_raw
        return base_mm
    return fused_raw


def select_viz_points(
    *,
    hand_frame,
    hand_frame_palm_plane,
    hand_frame_scaled,
    base_mm,
    metric_hand_to_palm_plane_normalized,
    metric_hand_to_shape_normalized,
):
    """Apply the selected hand coordinate frame for visualization.

    Args:
        hand_frame: Frame mode string (palm-plane, scaled, or raw metric).
        hand_frame_palm_plane: Constant for palm-plane normalization.
        hand_frame_scaled: Constant for shape normalization.
        base_mm: Raw mm keypoints before normalization.
        metric_hand_to_palm_plane_normalized: Callable for palm-plane frame.
        metric_hand_to_shape_normalized: Callable for shape-normalized frame.

    Returns:
        List of 21 visualization points in the chosen coordinate frame.
    """
    if hand_frame == hand_frame_palm_plane:
        return metric_hand_to_palm_plane_normalized(base_mm)
    if hand_frame == hand_frame_scaled:
        return metric_hand_to_shape_normalized(base_mm)
    return list(base_mm)


def smooth_viz_points(viz_pts, hand_ema_in, ema_alpha: float, ema_point_triplet):
    """Apply per-keypoint EMA smoothing to visualization points.

    Args:
        viz_pts: Current visualization points (21 entries).
        hand_ema_in: Previous EMA state for this hand, or ``None``.
        ema_alpha: EMA smoothing factor passed to ``ema_point_triplet``.
        ema_point_triplet: Callable ``(prev, cur, alpha) -> smoothed triplet``.

    Returns:
        List of smoothed ``(x, y, z)`` tuples, one per keypoint.
    """
    out = []
    for kp_id in range(len(viz_pts)):
        prev_k = hand_ema_in[kp_id] if hand_ema_in is not None and kp_id < len(hand_ema_in) else None
        out.append(ema_point_triplet(prev_k, viz_pts[kp_id], ema_alpha))
    return out

