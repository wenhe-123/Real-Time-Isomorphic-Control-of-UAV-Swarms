"""Orbbec per-frame draw pipeline: depth reads, MP mm coords, fused raw, base selection, smoothed viz points."""

from __future__ import annotations

import cv2
import numpy as np


def extract_points_and_depth(hand_landmarks, h: int, w: int, depth_reader):
    points = []
    depth_vals = []
    for lm in hand_landmarks:
        x = int(np.clip(int(lm.x * w), 0, w - 1))
        y = int(np.clip(int(lm.y * h), 0, h - 1))
        points.append((x, y))
        depth_vals.append(depth_reader(x, y, h, w))
    return points, depth_vals


def build_mp_mm(world_landmarks, n_kp: int, mp_world_to_mm):
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
    if hand_frame == hand_frame_palm_plane:
        return metric_hand_to_palm_plane_normalized(base_mm)
    if hand_frame == hand_frame_scaled:
        return metric_hand_to_shape_normalized(base_mm)
    return list(base_mm)


def smooth_viz_points(viz_pts, hand_ema_in, ema_alpha: float, ema_point_triplet):
    out = []
    for kp_id in range(len(viz_pts)):
        prev_k = hand_ema_in[kp_id] if hand_ema_in is not None and kp_id < len(hand_ema_in) else None
        out.append(ema_point_triplet(prev_k, viz_pts[kp_id], ema_alpha))
    return out


def draw_2d_overlay(
    frame,
    *,
    idx: int,
    hand_landmarks,
    points,
    depth_vals,
    norm_depth_label: bool,
    print_depth: bool,
    draw_wrist_label: bool,
    handed_label: str | None,
    hand_connections,
):
    for kp_id, _lm in enumerate(hand_landmarks):
        x, y = points[kp_id]
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        depth_mm = depth_vals[kp_id]
        if depth_mm is not None and depth_mm > 0:
            dw = depth_vals[0]
            if norm_depth_label and dw is not None and dw > 0:
                label = f"{depth_mm - dw:+d}"
            else:
                label = f"{depth_mm}"
            cv2.putText(
                frame,
                label,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
            if print_depth:
                print(f"hand:{idx} kp:{kp_id:02d} x:{x:4d} y:{y:4d} depth_mm:{depth_mm:5d}")

    for a, b in hand_connections:
        p1 = points[a]
        p2 = points[b]
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

    if draw_wrist_label and handed_label:
        cv2.putText(
            frame,
            handed_label,
            points[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

