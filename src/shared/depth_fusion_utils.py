"""Depth fusion with MediaPipe: coordinate conversion, RGB/depth pixel mapping, unprojection, joint EMA.

Used by Orbbec/K4A pipelines to align 2D depth with MP world coordinates into stable metric 3D (mm).
"""

from __future__ import annotations

import numpy as np
from pyk4a.calibration import CalibrationType


def mp_world_to_mm(wlm):
    return (
        float(wlm.x * 1000.0),
        float(-wlm.y * 1000.0),
        float(-wlm.z * 1000.0),
    )


def map_color_pixel_to_depth_pixel(x_c: int, y_c: int, w_c: int, h_c: int, w_d: int, h_d: int):
    xd = int(np.clip(round(x_c * (w_d / max(w_c, 1))), 0, w_d - 1))
    yd = int(np.clip(round(y_c * (h_d / max(h_c, 1))), 0, h_d - 1))
    return xd, yd


def unproject_depth_pixel_to_depth_camera_mm(calibration, x_d: float, y_d: float, depth_mm: float):
    if calibration is None or depth_mm is None or depth_mm <= 0:
        return None
    try:
        p = calibration.convert_2d_to_3d(
            (float(x_d), float(y_d)),
            float(depth_mm),
            CalibrationType.DEPTH,
            CalibrationType.DEPTH,
        )
        return (float(p[0]), float(p[1]), float(p[2]))
    except (ValueError, Exception):
        return None


def unproject_color_aligned_to_depth_camera_mm(calibration, x_c: float, y_c: float, depth_mm: float):
    if calibration is None or depth_mm is None or depth_mm <= 0:
        return None
    try:
        p = calibration.convert_2d_to_3d(
            (float(x_c), float(y_c)),
            float(depth_mm),
            CalibrationType.COLOR,
            CalibrationType.DEPTH,
        )
        return (float(p[0]), float(p[1]), float(p[2]))
    except (ValueError, Exception):
        return None


def transform_point_rigid_4x4_mm(p_xyz: tuple | None, T: np.ndarray | None) -> tuple | None:
    if T is None or p_xyz is None:
        return p_xyz
    if np.any(np.isnan(np.asarray(p_xyz, dtype=float))):
        return p_xyz
    v = np.array([float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2]), 1.0], dtype=np.float64)
    o = T @ v
    return (float(o[0]), float(o[1]), float(o[2]))


def fuse_cam_and_mp(p_cam, p_mp, fusion_weight: float):
    w = float(np.clip(fusion_weight, 0.0, 1.0))
    p_mp = np.array(p_mp, dtype=float)
    if p_cam is None or np.any(np.isnan(p_cam)):
        return (float(p_mp[0]), float(p_mp[1]), float(p_mp[2]))
    p_cam = np.array(p_cam, dtype=float)
    out = w * p_cam + (1.0 - w) * p_mp
    return (float(out[0]), float(out[1]), float(out[2]))


def ema_point_triplet(prev, cur, alpha: float):
    if prev is None:
        return cur
    a = np.array(prev, dtype=float)
    b = np.array(cur, dtype=float)
    if np.any(np.isnan(b)):
        return tuple(a.tolist())
    return tuple((float(v) for v in ((1.0 - alpha) * a + alpha * b)))


def median_valid_depth_mm(depth_img: np.ndarray, u: int, v: int, patch_r: int):
    if depth_img is None or patch_r < 0:
        return None
    h, w = int(depth_img.shape[0]), int(depth_img.shape[1])
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    if patch_r == 0:
        d = int(depth_img[v, u])
        return d if d > 0 else None
    u0, u1 = max(0, u - patch_r), min(w, u + patch_r + 1)
    v0, v1 = max(0, v - patch_r), min(h, v + patch_r + 1)
    patch = depth_img[v0:v1, u0:u1].astype(np.float64).ravel()
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return int(np.median(patch))


def read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, patch_r: int):
    if depth_aligned is not None and depth_aligned.shape[0] == h and depth_aligned.shape[1] == w:
        return median_valid_depth_mm(depth_aligned, x, y, patch_r)
    if depth_raw is not None:
        xd, yd = map_color_pixel_to_depth_pixel(x, y, w, h, depth_raw.shape[1], depth_raw.shape[0])
        return median_valid_depth_mm(depth_raw, xd, yd, patch_r)
    return None


def unproject_to_depth_cam_mm(calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw):
    if calibration is None or depth_mm is None or depth_mm <= 0:
        return None
    if depth_aligned is not None and depth_aligned.shape[0] == h and depth_aligned.shape[1] == w:
        return unproject_color_aligned_to_depth_camera_mm(calibration, float(x), float(y), float(depth_mm))
    if depth_raw is not None:
        xd, yd = map_color_pixel_to_depth_pixel(x, y, w, h, depth_raw.shape[1], depth_raw.shape[0])
        return unproject_depth_pixel_to_depth_camera_mm(calibration, float(xd), float(yd), float(depth_mm))
    return None


def reject_depth_outliers(
    depth_vals,
    *,
    depth_abs_max_mm: float,
    max_delta_mm: float,
    median_max_delta_mm: float | None,
    depth_ref_anchor_ids,
    wrist_id: int,
):
    if not depth_vals or len(depth_vals) < 21:
        return depth_vals
    dw = depth_vals[wrist_id]
    if dw is None or dw <= 0:
        return depth_vals
    dw = float(dw)
    out = list(depth_vals)
    for i in range(len(out)):
        d = out[i]
        if d is None or d <= 0:
            continue
        df = float(d)
        if df > depth_abs_max_mm or abs(df - dw) > max_delta_mm:
            out[i] = None

    if median_max_delta_mm is None or median_max_delta_mm <= 0:
        return out

    ref_vals = []
    for i in depth_ref_anchor_ids:
        if i < len(out) and out[i] is not None and out[i] > 0:
            ref_vals.append(float(out[i]))
    if len(ref_vals) < 2:
        return out
    mref = float(np.median(ref_vals))
    for i in range(len(out)):
        if i == wrist_id:
            continue
        d = out[i]
        if d is None or d <= 0:
            continue
        if abs(float(d) - mref) > float(median_max_delta_mm):
            out[i] = None
    return out

