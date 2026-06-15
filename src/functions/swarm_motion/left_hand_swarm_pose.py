"""Left-hand 6DoF rigid control for the swarm.

**Translation:** palm-center 3D from depth unprojection (plane-fit centroid, not wrist).
**Rotation:** palm orthonormal basis from the same depth landmarks (not webcam 2D).

MediaPipe ``hand_world_landmarks`` are wrist-centric and must not drive global pan.

**Press 0:** freeze palm-center translation origin + palm basis and per-drone rotation pivots at
arm time. Each frame: rigid ``T_now · T_arm⁻¹`` is applied on **live** morph/open targets.
Translation-dominant motion suppresses spurious palm-basis flips via ``sanitize_palm_rotvec_apply``
and axis-locked trans/rot blend weights.
Bad basis / jump frames use partial blend toward the rigid target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from functions.mode_switch.hand_constants import (
    HAND_SPAN_LANDMARK_IDS,
    INDEX_MCP_ID,
    INDEX_TIP_ID,
    MCP_IDS,
    MIDDLE_MCP_ID,
    MIDDLE_TIP_ID,
    PALM_CENTER_IDS,
    RING_MCP_ID,
    RING_TIP_ID,
    THUMB_MCP_ID,
    THUMB_TIP_ID,
    WRIST_ID,
)

_MIN_FINGERTIP_REL_MM = 8.0

# Palm local rotation axes → simulation world axes.
# x=thumb/lateral, y=fingertip, z=palm normal. Map palm-normal twist to world Z yaw.
PALM_AXIS_TO_WORLD_PERM = (0, 1, 2)

# Palm frame: Gram–Schmidt on (p_first − wrist), (p_second − wrist). Wider span reduces
# spurious rotation when index/middle MCPs are close in depth noise.
LEFT_PALM_BASIS_PRESETS: dict[str, tuple[int, int]] = {
    # Default: X/Y in screen plane; Y is wrist→middle fingertip, X thumb side; Z flips for palm/back.
    "middle_thumb": (MIDDLE_MCP_ID, THUMB_MCP_ID),
    "index_middle": (MIDDLE_MCP_ID, INDEX_MCP_ID),
    "index_ring": (RING_MCP_ID, INDEX_MCP_ID),
    "middle_ring": (RING_MCP_ID, MIDDLE_MCP_ID),
}
DEFAULT_LEFT_PALM_BASIS = "middle_thumb"


def palm_basis_pair_indices(preset: str) -> tuple[int, int]:
    key = str(preset).strip().lower()
    if key not in LEFT_PALM_BASIS_PRESETS:
        keys = ", ".join(sorted(LEFT_PALM_BASIS_PRESETS))
        raise ValueError(f"unknown left palm basis {preset!r}; choose one of: {keys}")
    a, b = LEFT_PALM_BASIS_PRESETS[key]
    return int(a), int(b)


# Prefer tip−wrist if segment length exceeds this (mm); else MCP−wrist.
_MIN_TIP_EDGE_MM = 12.0

# --- Depth camera (mm) → simulation world (m) for left-swarm rigid -----------------
# Femto Bolt / K4A depth camera: +X right, +Y down, +Z forward (Orbbec documentation).
# Each preset is an orthonormal 3×3 ``M`` with det(M)=+1 used for translation (after row scaling)
# and for rotation ``R_world = M @ R_cam @ M.T``.
#
# Camera mm from ``unproject_to_depth_cam_mm``: +X right, +Y down, +Z forward (Femto / K4A).
# World: Z-up. See module docstring for fwd_y row meanings.

LEFT_CAM_PRESET_ROT: dict[str, np.ndarray] = {
    # Depth-camera axes = simulation axes (X right, Y down, Z forward). Use with ``camera_at_arm``.
    "camera": np.eye(3, dtype=np.float64),
    # Older iso_swarm mapping: −optical Z → world X, +cam X → world Y, −cam Y → world Z.
    "legacy": np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    # Optical Z (depth) → world Y “into scene”; optical X (image right) → world X; optical Y → −world Z.
    "fwd_y": np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    # Negate depth column vs ``legacy`` (try if near/far feels inverted).
    "flip_depth": np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
}

def build_sim_from_cam_matrices(
    preset: str,
    *,
    image_y_to_world_z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M_rot, M_trans)`` embedding depth-camera mm into simulation coordinates."""
    M_rot = left_cam_preset_rotation(preset)
    M_trans = make_cam_translation_matrix(M_rot, image_y_to_world_z=float(image_y_to_world_z))
    return M_rot, M_trans


def left_cam_preset_rotation(preset: str) -> np.ndarray:
    """Return a copy of the orthonormal rotation for ``preset`` (``camera`` | ``legacy`` | ``fwd_y`` | ``flip_depth``)."""
    key = str(preset).strip().lower()
    if key not in LEFT_CAM_PRESET_ROT:
        keys = ", ".join(sorted(LEFT_CAM_PRESET_ROT))
        raise ValueError(f"unknown left-cam preset {preset!r}; choose one of: {keys}")
    return np.asarray(LEFT_CAM_PRESET_ROT[key], dtype=np.float64).copy()


def make_cam_translation_matrix(M_rot: np.ndarray, *, image_y_to_world_z: float = 0.0) -> np.ndarray:
    """World mm delta from camera mm delta: ``M_trans @ v`` with same XY rows as ``M_rot``.

    World-Z row is ``M_rot[2,:] * image_y_to_world_z`` so ``0`` disables altitude from camera Y
    (avoids near/far coupling into ``wz``). ``1`` restores full third row like rotation.
    """
    M = np.asarray(M_rot, dtype=np.float64).reshape(3, 3).copy()
    s = float(np.clip(image_y_to_world_z, 0.0, 1.0))
    M[2, :] *= s
    return M




_PALM_ROOT_IDS = (WRIST_ID, THUMB_MCP_ID, *MCP_IDS)


def _project_point_onto_plane(p: np.ndarray, n: np.ndarray, h_plane: float) -> np.ndarray:
    d = float(np.dot(n, p)) - float(h_plane)
    return np.asarray(p, dtype=np.float64).reshape(3) - d * np.asarray(n, dtype=np.float64).reshape(3)


def palm_roots_centroid_mm(h: np.ndarray, *, z_outlier_mm: float = 90.0) -> np.ndarray | None:
    """Equal-weight mean of wrist + five finger MCPs in depth camera mm."""
    pts: list[np.ndarray] = []
    for jidx in _PALM_ROOT_IDS:
        p = _landmark_mm_if_valid(h, int(jidx))
        if p is not None:
            pts.append(p)
    if len(pts) < 4:
        return None
    if len(pts) >= 5:
        z_vals = np.array([float(p[2]) for p in pts], dtype=np.float64)
        z_med = float(np.median(z_vals))
        kept = [p for p in pts if abs(float(p[2]) - z_med) <= float(z_outlier_mm)]
        if len(kept) >= 4:
            pts = kept
    return np.mean(np.stack(pts, axis=0), axis=0)


def palm_frame_origin_mm(h: np.ndarray) -> np.ndarray | None:
    """Palm frame origin: wrist + five MCP centroid projected onto the fitted palm plane."""
    fit = palm_plane_fit_mm(h)
    if fit is None:
        return palm_roots_centroid_mm(h)
    n, hp, _ = fit
    roots: list[np.ndarray] = []
    for jidx in _PALM_ROOT_IDS:
        p = _landmark_mm_if_valid(h, int(jidx))
        if p is None:
            continue
        if abs(float(np.dot(n, p)) - hp) <= float(_PALM_PLANE_OUTLIER_MM):
            roots.append(p)
    c = np.mean(np.stack(roots, axis=0), axis=0) if len(roots) >= 4 else palm_roots_centroid_mm(h)
    if c is None:
        return None
    return _project_point_onto_plane(c, n, hp)


def palm_center_mm(h: np.ndarray) -> np.ndarray | None:
    """Palm frame origin in depth camera mm (wrist + five MCPs on palm plane)."""
    return palm_frame_origin_mm(h)


def _is_depth_measurement_reliable(
    depth_mm: float | None,
    prev_z_mm: float,
    *,
    lateral_delta_mm: float,
    z_outlier_mm: float,
    z_outlier_lateral_ratio: float,
) -> bool:
    """True when a fresh depth sample at the palm centroid is usable."""
    if depth_mm is None or not np.isfinite(depth_mm) or float(depth_mm) <= 0.0:
        return False
    dz = abs(float(depth_mm) - float(prev_z_mm))
    z_thr = float(max(z_outlier_mm, 1.0))
    lat_thr = float(max(z_outlier_lateral_ratio, 0.5))
    if dz >= z_thr and dz >= lat_thr * max(float(lateral_delta_mm), 8.0):
        return False
    return True


def _unproject_palm_color_px_mm(
    color_px: tuple[int, int],
    depth_mm: float,
    *,
    calibration,
    frame_h: int,
    frame_w: int,
    depth_aligned,
    depth_raw,
) -> np.ndarray | None:
    from functions.display_sim.depth_fusion_utils import unproject_to_depth_cam_mm

    if calibration is None or depth_mm <= 0.0 or not np.isfinite(depth_mm):
        return None
    xc, yc = int(color_px[0]), int(color_px[1])
    p = unproject_to_depth_cam_mm(
        calibration,
        xc,
        yc,
        float(depth_mm),
        int(frame_h),
        int(frame_w),
        depth_aligned,
        depth_raw,
    )
    if p is None:
        return None
    return np.asarray(p, dtype=np.float64).reshape(3)


def filter_palm_center_depth_mm(
    palm_center: np.ndarray,
    state: "LeftSwarmPoseState",
    *,
    color_px: tuple[int, int] | None = None,
    calibration=None,
    frame_h: int = 0,
    frame_w: int = 0,
    depth_aligned=None,
    depth_raw=None,
    depth_patch_r: int = 2,
    measured_depth_mm: float | None = None,
    z_outlier_mm: float = 95.0,
    z_outlier_lateral_ratio: float = 2.2,
    ema_alpha: float = 0.42,
) -> tuple[np.ndarray, bool]:
    """Track palm centroid in depth-camera mm via 2D centroid + depth unprojection.

    Measurement is always from ``color_px`` + depth (not landmark 3D averaging). When depth
    is reliable the filtered point snaps to the measurement; when not, Z is held and XY
    still follows the current 2D centroid at the held depth.
    """
    pc_raw = np.asarray(palm_center, dtype=np.float64).reshape(3)
    prev = getattr(state, "filtered_palm_mm", None)
    if prev is None or not bool(state.initialized):
        state.filtered_palm_mm = pc_raw.copy()
        state.last_depth_outlier = False
        return pc_raw, True

    prev = np.asarray(prev, dtype=np.float64).reshape(3)
    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)

    d_meas = measured_depth_mm
    if d_meas is None and color_px is not None:
        xc, yc = int(color_px[0]), int(color_px[1])
        if calibration is not None:
            from functions.display_sim.depth_fusion_utils import read_depth_mm_at_landmark

            d_meas = read_depth_mm_at_landmark(
                xc,
                yc,
                fh,
                fw,
                depth_aligned,
                depth_raw,
                int(depth_patch_r),
            )

    dl = float(np.hypot(float(pc_raw[0] - prev[0]), float(pc_raw[1] - prev[1])))
    if color_px is not None and prev is not None:
        lu = getattr(state, "last_palm_color_u", None)
        lv = getattr(state, "last_palm_color_v", None)
        if lu is not None and lv is not None:
            dl = max(
                dl,
                float(np.hypot(float(color_px[0]) - float(lu), float(color_px[1]) - float(lv))),
            )
    depth_ok = _is_depth_measurement_reliable(
        d_meas,
        float(prev[2]),
        lateral_delta_mm=dl,
        z_outlier_mm=float(z_outlier_mm),
        z_outlier_lateral_ratio=float(z_outlier_lateral_ratio),
    )
    if not depth_ok and d_meas is not None and bool(state.initialized):
        ref = np.asarray(getattr(state, "ref_palm_center", prev), dtype=np.float64).reshape(3)
        prev_err = abs(float(prev[2]) - float(ref[2]))
        meas_err = abs(float(d_meas) - float(ref[2]))
        if meas_err <= max(140.0, 0.55 * prev_err) and meas_err + 45.0 < prev_err:
            depth_ok = True

    z_use = float(d_meas) if depth_ok and d_meas is not None else float(prev[2])
    pc_meas = pc_raw
    if color_px is not None and calibration is not None and z_use > 0.0:
        p_px = _unproject_palm_color_px_mm(
            color_px,
            z_use,
            calibration=calibration,
            frame_h=fh,
            frame_w=fw,
            depth_aligned=depth_aligned,
            depth_raw=depth_raw,
        )
        if p_px is not None:
            pc_meas = p_px

    alpha = float(np.clip(ema_alpha, 0.12, 1.0))
    if depth_ok:
        pc_use = alpha * np.asarray(pc_meas, dtype=np.float64).reshape(3) + (1.0 - alpha) * prev
    else:
        pc_use = prev.copy()
        pc_use[0] = float(pc_meas[0])
        pc_use[1] = float(pc_meas[1])
        pc_use[2] = float(prev[2])

    if color_px is not None:
        state.last_palm_color_u = float(color_px[0])
        state.last_palm_color_v = float(color_px[1])

    state.filtered_palm_mm = np.asarray(pc_use, dtype=np.float64).reshape(3).copy()
    state.last_depth_outlier = not bool(depth_ok)
    return pc_use, bool(depth_ok)


def _landmark_color_px(
    hlm,
    jidx: int,
    *,
    frame_h: int,
    frame_w: int,
    mp_h: int,
    mp_w: int,
) -> tuple[float, float] | None:
    if jidx < 0 or jidx >= len(hlm):
        return None
    lm = hlm[jidx]
    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    mh = max(int(mp_h), 1)
    mw = max(int(mp_w), 1)
    u = float(lm.x) * float(mw) * (float(fw) / float(mw))
    v = float(lm.y) * float(mh) * (float(fh) / float(mh))
    return u, v


def palm_center_color_px_from_landmarks(
    result,
    hand_idx: int,
    frame_h: int,
    frame_w: int,
    mp_h: int,
    mp_w: int,
) -> tuple[int, int] | None:
    """2D palm origin on the color frame: mean pixel of wrist + five finger MCPs."""
    if result is None or not getattr(result, "hand_landmarks", None):
        return None
    if hand_idx < 0 or hand_idx >= len(result.hand_landmarks):
        return None
    hlm = result.hand_landmarks[hand_idx]
    xs: list[float] = []
    ys: list[float] = []
    for jidx in PALM_CENTER_IDS:
        uv = _landmark_color_px(hlm, int(jidx), frame_h=frame_h, frame_w=frame_w, mp_h=mp_h, mp_w=mp_w)
        if uv is None:
            continue
        xs.append(float(uv[0]))
        ys.append(float(uv[1]))
    if len(xs) < 3:
        return None
    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    xc = int(np.clip(round(float(np.mean(xs))), 0, fw - 1))
    yc = int(np.clip(round(float(np.mean(ys))), 0, fh - 1))
    return xc, yc


def robust_palm_depth_mm_from_landmarks(
    result,
    hand_idx: int,
    *,
    frame_h: int,
    frame_w: int,
    mp_h: int,
    mp_w: int,
    depth_aligned,
    depth_raw,
    patch_r: int,
    fallback_px: tuple[int, int] | None = None,
) -> float | None:
    """Robust palm depth (mm): median of valid depth patches at wrist + MCP color pixels."""
    from functions.display_sim.depth_fusion_utils import read_depth_mm_at_landmark

    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    depths: list[float] = []
    if (
        result is not None
        and hand_idx is not None
        and getattr(result, "hand_landmarks", None)
        and 0 <= int(hand_idx) < len(result.hand_landmarks)
    ):
        hlm = result.hand_landmarks[int(hand_idx)]
        for jidx in PALM_CENTER_IDS:
            uv = _landmark_color_px(
                hlm,
                int(jidx),
                frame_h=fh,
                frame_w=fw,
                mp_h=int(mp_h),
                mp_w=int(mp_w),
            )
            if uv is None:
                continue
            d = read_depth_mm_at_landmark(
                int(round(uv[0])),
                int(round(uv[1])),
                fh,
                fw,
                depth_aligned,
                depth_raw,
                int(patch_r),
            )
            if d is not None and float(d) > 0.0:
                depths.append(float(d))
    if not depths and fallback_px is not None:
        d = read_depth_mm_at_landmark(
            int(fallback_px[0]),
            int(fallback_px[1]),
            fh,
            fw,
            depth_aligned,
            depth_raw,
            int(patch_r),
        )
        if d is not None and float(d) > 0.0:
            return float(d)
    if not depths:
        return None
    med = float(np.median(np.asarray(depths, dtype=np.float64)))
    kept = [d for d in depths if abs(d - med) <= 50.0]
    pool = kept if len(kept) >= 2 else depths
    return float(np.median(np.asarray(pool, dtype=np.float64)))


_PALM_PLANE_RAY_Z0_MM = 350.0
_PALM_PLANE_RAY_Z1_MM = 950.0
_PALM_PLANE_MIN_POINTS = 4
_PALM_PLANE_OUTLIER_MM = 40.0
_PALM_PLANE_MAX_MEDIAN_RESID_MM = 38.0
_PALM_PLANE_Z_MIN_MM = 180.0
_PALM_PLANE_Z_MAX_MM = 1600.0
# Wrist + MCPs first; finger tips added only if they agree with the plane.
_PALM_PLANE_CORE_IDS = (WRIST_ID, THUMB_MCP_ID, *MCP_IDS)
_PALM_PLANE_OPTIONAL_TIP_IDS = (MIDDLE_TIP_ID, INDEX_TIP_ID, RING_TIP_ID)


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Unit normal ``n`` and scalar ``h`` with ``n @ p = h`` for points on the plane."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < _PALM_PLANE_MIN_POINTS or not np.all(np.isfinite(P)):
        return None
    c = np.mean(P, axis=0)
    _, _, vh = np.linalg.svd(P - c, full_matrices=False)
    n = np.asarray(vh[-1, :], dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n /= nn
    if float(n[2]) < 0.0:
        n = -n
    h = float(np.dot(n, c))
    return n, h


def _landmark_mm_if_valid(h: np.ndarray, jidx: int) -> np.ndarray | None:
    p = np.asarray(h[int(jidx), :3], dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)) or float(p[2]) <= 0.0:
        return None
    return p


def _palm_plane_reject_outliers_mm(
    P: np.ndarray,
    *,
    outlier_mm: float = _PALM_PLANE_OUTLIER_MM,
    min_points: int = _PALM_PLANE_MIN_POINTS,
    iterations: int = 3,
) -> np.ndarray | None:
    """Iterative plane-distance rejection on 3D samples (depth-camera mm)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < int(min_points) or not np.all(np.isfinite(P)):
        return None
    for _ in range(int(max(1, iterations))):
        fit = _fit_plane_svd(P)
        if fit is None:
            return None
        n, hp = fit
        dist = np.abs(P @ n - hp)
        med = float(np.median(dist))
        thr = float(max(18.0, min(float(outlier_mm), med * 2.5 + 6.0)))
        keep = dist <= thr
        if int(np.sum(keep)) < int(min_points):
            break
        P = P[keep]
    return P if P.shape[0] >= int(min_points) else None


def palm_plane_fit_mm(
    h: np.ndarray,
    *,
    outlier_mm: float = _PALM_PLANE_OUTLIER_MM,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Robust palm plane: wrist + thumb/MCP core, optional tips if inliers.

    Returns ``(unit_normal, h_plane, inlier_points_Nx3)`` or ``None``.
    """
    core: list[np.ndarray] = []
    for jidx in _PALM_PLANE_CORE_IDS:
        p = _landmark_mm_if_valid(h, int(jidx))
        if p is not None:
            core.append(p)
    if len(core) < _PALM_PLANE_MIN_POINTS:
        return None
    P = _palm_plane_reject_outliers_mm(np.stack(core, axis=0), outlier_mm=float(outlier_mm))
    if P is None:
        return None
    fit = _fit_plane_svd(P)
    if fit is None:
        return None
    n, hp = fit
    tip_pts: list[np.ndarray] = []
    for jidx in _PALM_PLANE_OPTIONAL_TIP_IDS:
        p = _landmark_mm_if_valid(h, int(jidx))
        if p is None:
            continue
        if float(abs(float(np.dot(n, p)) - hp)) <= float(outlier_mm) * 0.85:
            tip_pts.append(p)
    if tip_pts:
        P = _palm_plane_reject_outliers_mm(
            np.vstack([P, np.stack(tip_pts, axis=0)]),
            outlier_mm=float(outlier_mm),
        )
        if P is None:
            return None
        fit = _fit_plane_svd(P)
        if fit is None:
            return None
        n, hp = fit
    if _palm_plane_median_residual_mm(P, n, hp) > _PALM_PLANE_MAX_MEDIAN_RESID_MM:
        return None
    return n, hp, P


def _palm_plane_inlier_points_mm(
    h: np.ndarray,
    *,
    outlier_mm: float = _PALM_PLANE_OUTLIER_MM,
) -> np.ndarray | None:
    """Wrist + palm landmarks in depth-camera mm after plane outlier rejection."""
    fit = palm_plane_fit_mm(h, outlier_mm=float(outlier_mm))
    if fit is None:
        return None
    _, _, P = fit
    return P


def _palm_plane_median_residual_mm(P: np.ndarray, n: np.ndarray, h_plane: float) -> float:
    return float(np.median(np.abs(P @ n - h_plane)))


def _ray_plane_intersect_at_color_px_mm(
    color_px: tuple[int, int],
    n: np.ndarray,
    h_plane: float,
    *,
    calibration,
    frame_h: int,
    frame_w: int,
    depth_aligned,
    depth_raw,
) -> np.ndarray | None:
    """Intersect the color-pixel viewing ray with the palm plane (depth-camera mm)."""
    from functions.display_sim.depth_fusion_utils import unproject_to_depth_cam_mm

    xc, yc = int(color_px[0]), int(color_px[1])
    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    p0 = unproject_to_depth_cam_mm(
        calibration, xc, yc, _PALM_PLANE_RAY_Z0_MM, fh, fw, depth_aligned, depth_raw
    )
    p1 = unproject_to_depth_cam_mm(
        calibration, xc, yc, _PALM_PLANE_RAY_Z1_MM, fh, fw, depth_aligned, depth_raw
    )
    if p0 is None or p1 is None:
        return None
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    ray = p1 - p0
    denom = float(np.dot(n, ray))
    if abs(denom) < 1e-6:
        return None
    t = (float(h_plane) - float(np.dot(n, p0))) / denom
    p = p0 + t * ray
    if not np.all(np.isfinite(p)):
        return None
    z = float(p[2])
    if z < _PALM_PLANE_Z_MIN_MM or z > _PALM_PLANE_Z_MAX_MM:
        return None
    return p


def palm_center_mm_from_palm_plane(
    h: np.ndarray,
    color_px: tuple[int, int],
    *,
    calibration,
    frame_h: int,
    frame_w: int,
    depth_aligned,
    depth_raw,
) -> np.ndarray | None:
    """Palm center: 2D centroid pixel ray intersects plane fit to wrist + MCP depth samples."""
    P = _palm_plane_inlier_points_mm(h)
    if P is None:
        return None
    fit = _fit_plane_svd(P)
    if fit is None:
        return None
    n, h_plane = fit
    if _palm_plane_median_residual_mm(P, n, h_plane) > _PALM_PLANE_MAX_MEDIAN_RESID_MM:
        return None
    p = _ray_plane_intersect_at_color_px_mm(
        color_px,
        n,
        h_plane,
        calibration=calibration,
        frame_h=int(frame_h),
        frame_w=int(frame_w),
        depth_aligned=depth_aligned,
        depth_raw=depth_raw,
    )
    if p is None:
        return None
    return np.asarray(p, dtype=np.float64).reshape(3)


def palm_center_mm_from_landmarks_2d(
    result,
    hand_idx: int,
    frame_h: int,
    frame_w: int,
    mp_h: int,
    mp_w: int,
    *,
    calibration,
    depth_aligned,
    depth_raw,
    patch_r: int,
) -> np.ndarray | None:
    """Palm center in depth-camera mm: 2D centroid + robust multi-landmark depth + unproject."""
    from functions.display_sim.depth_fusion_utils import unproject_to_depth_cam_mm

    px = palm_center_color_px_from_landmarks(result, hand_idx, frame_h, frame_w, mp_h, mp_w)
    if px is None:
        return None
    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    xc, yc = px
    d = robust_palm_depth_mm_from_landmarks(
        result,
        hand_idx,
        frame_h=fh,
        frame_w=fw,
        mp_h=int(mp_h),
        mp_w=int(mp_w),
        depth_aligned=depth_aligned,
        depth_raw=depth_raw,
        patch_r=int(patch_r),
        fallback_px=(xc, yc),
    )
    p = unproject_to_depth_cam_mm(calibration, xc, yc, d, fh, fw, depth_aligned, depth_raw)
    if p is None:
        return None
    return np.asarray(p, dtype=np.float64).reshape(3)


def palm_center_components_mm(h: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Return (wrist, roots_mean_ex_wrist, valid_root_count) in depth camera mm."""
    wrist = _landmark_mm_if_valid(h, WRIST_ID)
    if wrist is None:
        return None, None, 0
    root_pts: list[np.ndarray] = []
    for jidx in (THUMB_MCP_ID, *MCP_IDS):
        p = _landmark_mm_if_valid(h, int(jidx))
        if p is not None:
            root_pts.append(p)
    if not root_pts:
        return wrist, None, 0
    roots_mean = np.mean(np.stack(root_pts, axis=0), axis=0)
    return wrist, roots_mean, len(root_pts)


def hand_palm_span_mm(h: np.ndarray, wrist: np.ndarray | None = None) -> float:
    """Mean wrist→landmark distance (mm); grows when the hand moves closer (appears larger)."""
    w = np.asarray(h[WRIST_ID, :3], dtype=np.float64).reshape(3) if wrist is None else np.asarray(wrist).reshape(3)
    dists: list[float] = []
    for idx in HAND_SPAN_LANDMARK_IDS:
        v = np.asarray(h[int(idx), :3], dtype=np.float64).reshape(3) - w
        n = float(np.linalg.norm(v))
        if n >= 8.0:
            dists.append(n)
    if not dists:
        return 0.0
    return float(np.mean(dists))








def world_rotvec_from_palm_delta(
    Mc_rot: np.ndarray,
    B_current: np.ndarray,
    B_arm: np.ndarray,
) -> np.ndarray:
    """World rotation from palm basis change: ``ω = R_to_rotvec(M @ (B B_arm^T) @ M.T)``."""
    M = np.asarray(Mc_rot, dtype=np.float64).reshape(3, 3)
    R_cam = np.asarray(B_current, dtype=np.float64).reshape(3, 3) @ np.asarray(
        B_arm, dtype=np.float64
    ).reshape(3, 3).T
    R_world = M @ R_cam @ M.T
    return np.asarray(R_to_rotvec(R_world), dtype=np.float64).reshape(3)


def palm_world_rotvec_from_basis_delta(
    Mc_rot: np.ndarray | None,
    B_current: np.ndarray,
    B_arm: np.ndarray,
) -> np.ndarray:
    """Palm ΔR → world rotvec via palm-local axes (normal twist → world yaw)."""
    del Mc_rot
    rv_local = palm_local_rotvec_from_basis_delta(B_current, B_arm)
    return palm_world_rotvec_from_local_intrinsic(rv_local)


def stabilize_palm_basis_continuity(B: np.ndarray, B_ref: np.ndarray) -> np.ndarray:
    """Keep palm basis near the arm reference — removes 180° palm/back Z flips from depth noise."""
    out = np.asarray(B, dtype=np.float64).reshape(3, 3).copy()
    ref = np.asarray(B_ref, dtype=np.float64).reshape(3, 3)
    if float(np.dot(out[:, 2], ref[:, 2])) < 0.0:
        out[:, 0] *= -1.0
        out[:, 2] *= -1.0
    if float(np.dot(out[:, 0], ref[:, 0])) < 0.0:
        out[:, 0] *= -1.0
        out[:, 2] *= -1.0
    return out


def palm_cam_rotvec_from_basis_delta(B_current: np.ndarray, B_arm: np.ndarray) -> np.ndarray:
    """Intrinsic palm rotation in camera frame (for classify; less false rot on 3D translation)."""
    R = np.asarray(B_current, dtype=np.float64).reshape(3, 3) @ np.asarray(B_arm, dtype=np.float64).reshape(
        3, 3
    ).T
    rv = np.asarray(R_to_rotvec(R), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv)) > np.deg2rad(150.0):
        return np.zeros(3, dtype=np.float64)
    return rv


def palm_local_rotvec_from_basis_delta(B_current: np.ndarray, B_arm: np.ndarray) -> np.ndarray:
    """Palm-frame relative rotvec; local z is palm-normal twist."""
    R_local = np.asarray(B_arm, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        B_current, dtype=np.float64
    ).reshape(3, 3)
    rv = np.asarray(R_to_rotvec(R_local), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv)) > np.deg2rad(150.0):
        return np.zeros(3, dtype=np.float64)
    return rv


def palm_world_rotvec_from_local_intrinsic(rv_local: np.ndarray) -> np.ndarray:
    """Map palm-local axis-angle components to sim/world axis-angle components."""
    rv = np.asarray(rv_local, dtype=np.float64).reshape(3)
    out = np.zeros(3, dtype=np.float64)
    for palm_axis, world_axis in enumerate(PALM_AXIS_TO_WORLD_PERM):
        out[int(world_axis)] = float(rv[palm_axis])
    return out


def palm_world_rotvec_from_cam_intrinsic(
    Mc_rot: np.ndarray | None,
    rv_cam: np.ndarray,
) -> np.ndarray:
    """Map camera-frame palm twist to world; identity when ``rv_cam`` is zero."""
    rv_c = np.asarray(rv_cam, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv_c)) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    R_cam = rotvec_to_R(rv_c)
    if Mc_rot is not None:
        M = np.asarray(Mc_rot, dtype=np.float64).reshape(3, 3)
        R_world = M @ R_cam @ M.T
    else:
        R_world = R_cam
    return np.asarray(R_to_rotvec(R_world), dtype=np.float64).reshape(3)


def sanitize_palm_rotvec_apply(
    rv_world: np.ndarray,
    rv_cam: np.ndarray,
    *,
    prev_basis: np.ndarray | None,
    B_current: np.ndarray,
    Mc_rot: np.ndarray | None = None,
    max_step_rad: float | None = None,
) -> np.ndarray:
    """Drop basis-flip spikes before commanding swarm rotation (keeps intentional twist 1:1)."""
    rv_w = np.asarray(rv_world, dtype=np.float64).reshape(3)
    rv_c = np.asarray(rv_cam, dtype=np.float64).reshape(3)
    wn = float(np.linalg.norm(rv_w))
    cn = float(np.linalg.norm(rv_c))
    step_cap = float(max_step_rad if max_step_rad is not None else np.deg2rad(32.0))

    # Camera classifier zeroed a palm/back flip; do not apply inflated world pose.
    if cn < 1e-9 and wn > np.deg2rad(18.0):
        return np.zeros(3, dtype=np.float64)
    if wn > np.deg2rad(118.0):
        return np.zeros(3, dtype=np.float64)

    if prev_basis is not None:
        rv_step = palm_cam_rotvec_from_basis_delta(B_current, prev_basis)
        if Mc_rot is not None:
            rv_step = palm_world_rotvec_from_basis_delta(Mc_rot, B_current, prev_basis)
        if float(np.linalg.norm(rv_step)) > step_cap:
            return np.zeros(3, dtype=np.float64)

    return rv_w


def rate_limit_rotvec_toward(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_step_rad: float,
) -> np.ndarray:
    """Slew accumulated palm orientation toward absolute target (≤ ``max_step_rad`` per frame)."""
    cur = np.asarray(current, dtype=np.float64).reshape(3)
    tgt = np.asarray(target, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(tgt)) < 1e-12:
        return cur.copy()
    cap = float(max(max_step_rad, 1e-9))
    R_cur = rotvec_to_R(cur)
    R_tgt = rotvec_to_R(tgt)
    rv_step = R_to_rotvec(R_tgt @ R_cur.T)
    sn = float(np.linalg.norm(rv_step))
    if sn > cap:
        rv_step = rv_step * (cap / sn)
    return R_to_rotvec(rotvec_to_R(rv_step) @ R_cur)



def axis_locked_trans_rot_blend_weights(
    delta_world_m: np.ndarray,
    rv_world_rad: np.ndarray,
    *,
    trans_on_m: float,
    rot_on_rad: float,
    rv_cam_rad: np.ndarray | None = None,
    delta_cam_mm: np.ndarray | None = None,
    delta_trans_mm: np.ndarray | None = None,
    secondary_frac: float = 0.50,
    none_below: float = 0.12,
    rot_noise_rad: float = 0.10,
) -> tuple[str, float, float]:
    """Compare normalized translation vs rotation strength; return primary motion + blend weights.

    Rotation score uses **camera-frame** palm twist (``rv_cam``), not ``M @ R @ M.T`` (inflates on
  pans). Translation score uses world m **and** raw palm-center mm so metric scale is not lost.
    """
    rv_w = np.asarray(rv_world_rad, dtype=np.float64).reshape(3)
    rv_c = (
        np.asarray(rv_cam_rad, dtype=np.float64).reshape(3)
        if rv_cam_rad is not None
        else rv_w
    )
    tw = axis_locked_trans_metric_world_m(
        delta_world_m,
        rv_c,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(rot_on_rad),
        ignore_world_y_when_rotating=False,
    )
    t_on = float(max(trans_on_m, 1e-6))
    t_on_mm = t_on * 1000.0
    ts = float(tw / t_on)
    pan_mm = 0.0
    pan_frame_mm = 0.0
    if delta_cam_mm is not None:
        dc = np.asarray(delta_cam_mm, dtype=np.float64).reshape(3)
        pan_frame_mm = float(np.linalg.norm(dc))
        pan_mm = pan_frame_mm
        ts = max(ts, pan_mm / max(t_on_mm, 1e-6))
    if delta_trans_mm is not None:
        dt = np.asarray(delta_trans_mm, dtype=np.float64).reshape(3)
        gt = float(np.linalg.norm(dt))
        if gt >= 2.5:
            ts = max(ts, gt / max(t_on_mm, 1e-6))
    r_on = float(max(rot_on_rad, 1e-6))
    rv_n = float(np.linalg.norm(rv_c))
    rs = float(rv_n / r_on)
    rn = float(max(rot_noise_rad, 1e-6))
    if rv_n < rn:
        rs = 0.0
    elif rv_n < np.deg2rad(22.0):
        rs *= (rv_n / np.deg2rad(22.0)) ** 2
    if delta_cam_mm is not None:
        if pan_frame_mm < 4.0 and rv_n >= np.deg2rad(6.0):
            rs *= max(0.04, (pan_frame_mm / 4.0) ** 2)
        if pan_mm >= 18.0:
            rs *= max(0.15, 18.0 / pan_mm)
    if delta_trans_mm is not None:
        gt = float(np.linalg.norm(np.asarray(delta_trans_mm, dtype=np.float64).reshape(3)))
        if gt >= 4.0 and rv_n < np.deg2rad(20.0):
            rs *= max(0.12, 4.0 / max(gt, 4.0))
    nb = float(max(0.08, none_below))
    tw_n = float(np.linalg.norm(np.asarray(delta_world_m, dtype=np.float64).reshape(3)))
    if tw_n < trans_on_m * 0.22 and rv_n < rn * 1.2:
        return "none", 0.0, 0.0
    if ts < nb and rs < nb:
        return "none", 0.0, 0.0
    # Only freeze translation on a clear in-place twist: large rotation, tiny pan, no metric trans.
    if (
        rv_n >= np.deg2rad(70.0)
        and pan_mm <= 10.0
        and rs >= nb * 1.15
        and ts < nb * 0.85
    ):
        return "rotate", 1.0, 0.0
    sec = float(np.clip(secondary_frac, 0.0, 0.75))
    st = max(ts, 1e-9)
    sr = max(rs, 1e-9)
    if delta_trans_mm is not None and float(np.linalg.norm(np.asarray(delta_trans_mm).reshape(3))) >= 5.0:
        if rv_n < np.deg2rad(12.0) and st >= sr * 0.65:
            sr = sr * 0.45
    if sr >= st:
        w_rot = 1.0
        w_trans = 1.0 if ts >= nb else sec * min(1.0, ts / sr)
        return "rotate", w_rot, w_trans
    w_trans = 1.0
    w_rot = min(1.0, sr / st) if rs >= nb else sec * min(1.0, sr / st)
    return "translate", w_rot, w_trans


def axis_locked_trans_metric_world_m(
    delta_world_m: np.ndarray,
    rv_intrinsic_rad: np.ndarray,
    *,
    trans_on_m: float = 0.009,
    rot_on_rad: float = 0.011,
    ignore_world_y_when_rotating: bool = True,
) -> float:
    """Translation score for classify — drop world Y when palm is already twisting (stops rot→fwd/back leak)."""
    d = np.asarray(delta_world_m, dtype=np.float64).reshape(3).copy()
    r = float(np.linalg.norm(np.asarray(rv_intrinsic_rad, dtype=np.float64).reshape(3)))
    if bool(ignore_world_y_when_rotating) and r >= 0.72 * float(max(rot_on_rad, 1e-9)):
        d[1] = 0.0
    tw = float(np.linalg.norm(d))
    ay, by = abs(float(d[1])), max(abs(float(d[0])), abs(float(d[2])))
    if ay >= float(max(trans_on_m, 1e-6)) * 0.65 and ay >= 1.15 * by:
        tw = max(tw, ay)
    return tw



def rotvec_to_R(v: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector (axis * angle) -> 3x3."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    x, y, z = float(k[0]), float(k[1]), float(k[2])
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), angle in [0, pi]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = float(np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(c))
    rx = float(R[2, 1] - R[1, 2])
    ry = float(R[0, 2] - R[2, 0])
    rz = float(R[1, 0] - R[0, 1])
    v = np.array([rx, ry, rz], dtype=np.float64)
    s = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if s < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-2:
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]], dtype=np.float64)
        k = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        axis[k] = np.sqrt(max(diag[k] + 1.0, 0.0) * 0.5)
        j, m = (k + 1) % 3, (k + 2) % 3
        axis[j] = R[j, k] / max(2.0 * axis[k], 1e-9)
        axis[m] = R[m, k] / max(2.0 * axis[k], 1e-9)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return axis * theta
    axis = v / (2.0 * np.sin(theta))
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    return axis * theta


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion ``(w, x, y, z)``."""
    M = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(M))
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0) * 2.0)
        w = 0.25 * s
        x = (M[2, 1] - M[1, 2]) / s
        y = (M[0, 2] - M[2, 0]) / s
        z = (M[1, 0] - M[0, 1]) / s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = float(np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0)
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = float(np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0)
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0)
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternion ``(w,x,y,z)`` → 3×3 rotation matrix."""
    w, x, y, z = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation; ``t=0`` → ``q0``, ``t=1`` → ``q1``."""
    a = np.asarray(q0, dtype=np.float64).reshape(4)
    b = np.asarray(q1, dtype=np.float64).reshape(4)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.clip(float(np.dot(a, b)), -1.0, 1.0))
    if dot < 0.0:
        b = -b
        dot = -dot
    u = float(np.clip(t, 0.0, 1.0))
    if dot > 0.9995:
        out = a + u * (b - a)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta = float(np.arccos(dot))
    s = float(np.sin(theta))
    if s < 1e-12:
        return a.copy()
    w0 = float(np.sin((1.0 - u) * theta) / s)
    w1 = float(np.sin(u * theta) / s)
    return w0 * a + w1 * b


def scale_rotation_matrix(R: np.ndarray, *, scale: float, gain: float = 1.0) -> np.ndarray:
    """Scale a proper rotation about identity: ``slerp(I, R, scale*gain)``."""
    s = float(np.clip(scale * max(0.0, gain), 0.0, 1.0))
    if s <= 1e-12:
        return np.eye(3, dtype=np.float64)
    if s >= 1.0 - 1e-12:
        return np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = R_to_quat(R)
    if float(np.dot(q_id, q)) < 0.0:
        q = -q
    return quat_to_R(quat_slerp(q_id, q, s))


def sync_left_swarm_pose_output(state: "LeftSwarmPoseState", off: np.ndarray, R: np.ndarray) -> None:
    """Keep internal pose state aligned with the rigid transform actually applied."""
    state.ema_offset = np.asarray(off, dtype=np.float64).reshape(3).copy()
    state.ema_rotvec = np.asarray(R_to_rotvec(R), dtype=np.float64).reshape(3).copy()


def _resolve_cam_world_mats(
    state: "LeftSwarmPoseState",
    *,
    cam_delta_to_world: np.ndarray | None,
    cam_translation_to_world: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if state.frozen_M_rot is not None:
        Mc_rot = np.asarray(state.frozen_M_rot, dtype=np.float64).reshape(3, 3)
        Mc_trans = (
            np.asarray(state.frozen_M_trans, dtype=np.float64).reshape(3, 3)
            if state.frozen_M_trans is not None
            else Mc_rot
        )
        return Mc_rot, Mc_trans
    Mc_rot = (
        np.asarray(cam_delta_to_world, dtype=np.float64).reshape(3, 3)
        if cam_delta_to_world is not None
        else None
    )
    if cam_translation_to_world is not None:
        Mc_trans = np.asarray(cam_translation_to_world, dtype=np.float64).reshape(3, 3)
    else:
        Mc_trans = Mc_rot
    return Mc_rot, Mc_trans


def _rigid_target_from_hand(
    state: "LeftSwarmPoseState",
    *,
    delta_cam_arm: np.ndarray,
    B: np.ndarray,
    ref_b_rot: np.ndarray,
    rv_world_override: np.ndarray | None = None,
    Mc_rot: np.ndarray | None,
    Mc_trans: np.ndarray | None,
    sign: np.ndarray,
    trans_scale: float,
    rot_scale: float,
    rot_gain: float,
    rot_world_z_scale: float,
    trans_deadzone_m: float,
    rot_deadzone_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Absolute rigid target ``(off, R, off_raw, rv_world)`` from arm-relative hand pose."""
    if Mc_trans is not None:
        off_raw = float(trans_scale) * (Mc_trans @ delta_cam_arm) * sign
    else:
        off_raw = delta_cam_arm * sign * float(trans_scale)
    off = np.asarray(off_raw, dtype=np.float64).reshape(3).copy()
    if float(np.linalg.norm(off)) < float(trans_deadzone_m):
        off[:] = 0.0

    del Mc_rot
    if rv_world_override is not None:
        rv_world = np.asarray(rv_world_override, dtype=np.float64).reshape(3).copy()
    else:
        rv_local = palm_local_rotvec_from_basis_delta(B, ref_b_rot)
        rv_world = palm_world_rotvec_from_local_intrinsic(rv_local)
    R_world = rotvec_to_R(rv_world)
    zsc = float(rot_world_z_scale)
    if zsc != 1.0 and float(np.linalg.norm(rv_world)) >= 1e-9:
        rv_world = rv_world.copy()
        rv_world[2] *= zsc
        R_world = rotvec_to_R(rv_world)
    R_out = scale_rotation_matrix(R_world, scale=float(rot_scale), gain=float(rot_gain))
    rv_out = np.asarray(R_to_rotvec(R_out), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv_out)) < float(rot_deadzone_rad):
        R_out = np.eye(3, dtype=np.float64)
        rv_out = np.zeros(3, dtype=np.float64)
    return off, R_out, np.asarray(off_raw, dtype=np.float64).reshape(3), rv_world


def _smooth_rigid_pose(
    off_hold: np.ndarray,
    R_hold: np.ndarray,
    off_tgt: np.ndarray,
    R_tgt: np.ndarray,
    *,
    trans_blend: float,
    rot_blend: float,
    max_step_rad: float,
    max_offset_m: float,
    max_trans_step_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    tb = float(np.clip(trans_blend, 0.0, 1.0))
    rb = float(np.clip(rot_blend, 0.0, 1.0))
    off0 = np.asarray(off_hold, dtype=np.float64).reshape(3)
    off1 = np.asarray(off_tgt, dtype=np.float64).reshape(3)
    step = off1 - off0
    cap_m = float(max(max_trans_step_m, 0.0))
    sn = float(np.linalg.norm(step))
    if cap_m > 0.0 and sn > cap_m:
        off1 = off0 + step * (cap_m / sn)
    off = (1.0 - tb) * off0 + tb * off1
    on = float(np.linalg.norm(off))
    if max_offset_m > 0.0 and on > max_offset_m:
        off *= max_offset_m / max(on, 1e-9)

    q0 = R_to_quat(R_hold)
    q1 = R_to_quat(R_tgt)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = quat_slerp(q0, q1, rb)
    R = quat_to_R(q)
    cap = float(max(max_step_rad, 0.0))
    if cap > 0.0:
        rv_step = R_to_rotvec(R @ np.asarray(R_hold, dtype=np.float64).reshape(3, 3).T)
        sn = float(np.linalg.norm(rv_step))
        if sn > cap:
            R = rotvec_to_R(rv_step * (cap / sn)) @ np.asarray(R_hold, dtype=np.float64).reshape(3, 3)
    return off, R


def _reject_noisy_pose_frame(
    state: "LeftSwarmPoseState",
    *,
    palm_center: np.ndarray,
    B_depth: np.ndarray,
    delta_cam: np.ndarray,
    delta_cam_arm: np.ndarray,
    mcp_valid: int,
    wrist_mm: np.ndarray | None,
    depth_hold: bool = False,
) -> tuple[bool, str]:
    """True → soft-reject (partial blend). Only obvious tracking loss (too few MCPs)."""
    if int(mcp_valid) < 3:
        return True, "mcp"
    dc = np.asarray(delta_cam, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(dc))
    dn_xy = float(np.hypot(float(dc[0]), float(dc[1])))
    depth_recover = bool(depth_hold or getattr(state, "last_depth_outlier_prev", False))
    if depth_recover:
        return False, ""
    if dn_xy > 120.0 or dn > 220.0:
        return True, "jump"
    return False, ""


def left_hand_pose_matrix_depth_mm(
    result,
    idx_l: int,
    frame_h: int,
    frame_w: int,
    mp_h: int,
    mp_w: int,
    *,
    calibration,
    depth_aligned,
    depth_raw,
    patch_r: int,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
):
    """Wrist + palm MCP landmarks in **depth-camera mm** (real translation in space).

    MediaPipe ``hand_world_landmarks`` are wrist-centric (wrist does not move when the hand
    translates), so global shift must come from depth unprojection at 2D landmarks mapped
    from the MediaPipe input size to the color frame size.
    """
    if (
        result is None
        or calibration is None
        or idx_l is None
        or not getattr(result, "hand_landmarks", None)
        or idx_l >= len(result.hand_landmarks)
    ):
        return None

    from functions.display_sim.depth_fusion_utils import read_depth_mm_at_landmark, unproject_to_depth_cam_mm

    hlm = result.hand_landmarks[idx_l]
    if len(hlm) < 21:
        return None

    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    mh = max(int(mp_h), 1)
    mw = max(int(mp_w), 1)

    def joint_mm(jidx: int) -> np.ndarray | None:
        lm = hlm[jidx]
        u_mp = float(lm.x) * float(mw)
        v_mp = float(lm.y) * float(mh)
        xc = int(np.clip(round(u_mp * (fw / float(mw))), 0, fw - 1))
        yc = int(np.clip(round(v_mp * (fh / float(mh))), 0, fh - 1))
        d = read_depth_mm_at_landmark(xc, yc, fh, fw, depth_aligned, depth_raw, int(patch_r))
        p = unproject_to_depth_cam_mm(calibration, xc, yc, d, fh, fw, depth_aligned, depth_raw)
        if p is None:
            return None
        return np.asarray(p, dtype=np.float64)

    ia, ib = palm_basis_pair_indices(palm_basis)
    wrist = joint_mm(WRIST_ID)
    p_ia = joint_mm(ia)
    p_ib = joint_mm(ib)
    t_m = joint_mm(MIDDLE_TIP_ID)
    if wrist is None or p_ia is None or p_ib is None:
        return None
    # Keep missing joints as NaN so centroid/basis code can ignore them safely.
    out = np.full((21, 3), np.nan, dtype=np.float64)
    out[WRIST_ID] = wrist
    out[ia] = p_ia
    out[ib] = p_ib
    for j in (*MCP_IDS, THUMB_MCP_ID):
        if j == ia or j == ib:
            continue
        pj = joint_mm(j)
        if pj is not None:
            out[j] = pj
    if t_m is not None:
        out[MIDDLE_TIP_ID] = t_m
    px = palm_center_color_px_from_landmarks(result, idx_l, fh, fw, mh, mw)
    pc: np.ndarray | None = None
    if px is not None:
        pc = palm_center_mm_from_palm_plane(
            out,
            px,
            calibration=calibration,
            frame_h=fh,
            frame_w=fw,
            depth_aligned=depth_aligned,
            depth_raw=depth_raw,
        )
    if pc is None:
        pc = palm_center_mm_from_landmarks_2d(
            result,
            idx_l,
            fh,
            fw,
            mh,
            mw,
            calibration=calibration,
            depth_aligned=depth_aligned,
            depth_raw=depth_raw,
            patch_r=int(patch_r),
        )
    return out, pc


def hand_points_to_matrix(pts) -> np.ndarray | None:
    if pts is None:
        return None
    if isinstance(pts, np.ndarray) and pts.dtype == object:
        rows = []
        for p in pts:
            v = np.asarray(p, dtype=np.float64).ravel()
            if v.size < 3:
                return None
            rows.append(v[:3])
        h = np.stack(rows, axis=0)
    else:
        h = np.asarray(pts, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 21 or h.shape[1] < 3:
        return None
    return h


def mp_hand_visibility_scores(result, hand_idx: int) -> tuple[float, float]:
    """Return (mean, min) per-joint visibility/presence in [0,1] for a hand index."""
    from functions.dual_cam.mp_hand_utils import extract_landmark_visibilities

    vis = extract_landmark_visibilities(result, hand_idx)
    if vis is None:
        return 0.0, 0.0
    return float(np.mean(vis)), float(np.min(vis))


def palm_basis_from_mp_image_plane(
    result,
    hand_idx: int,
    *,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
) -> np.ndarray | None:
    """Palm frame from 2D landmarks in depth-camera axes (+X right, +Y down, +Z forward).

    Used when depth MCPs are occluded but the hand is still visible in a webcam / color view.
    Wrist and two MCPs lie in the image plane (z=0); ``e3 = e1×e2`` points along +Z_cam.
    """
    if (
        result is None
        or not getattr(result, "hand_landmarks", None)
        or hand_idx is None
        or hand_idx >= len(result.hand_landmarks)
    ):
        return None
    hlm = result.hand_landmarks[hand_idx]
    if len(hlm) < 21:
        return None
    def xy(j: int) -> np.ndarray:
        lm = hlm[j]
        return np.array([float(lm.x), float(lm.y), 0.0], dtype=np.float64)

    key = str(palm_basis).strip().lower()
    if key in ("middle_thumb", "middle_y", "middle"):
        h2 = np.zeros((21, 3), dtype=np.float64)
        for j in range(21):
            h2[j] = xy(j)
        out = palm_orthonormal_basis_middle_y_thumb_x(h2)
        return None if out is None else out[1]

    ia, ib = palm_basis_pair_indices(palm_basis)
    wrist = xy(WRIST_ID)
    v1 = xy(ia) - wrist
    v2 = xy(ib) - wrist
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    e1 = v1 / n1
    v2o = v2 - float(np.dot(v2, e1)) * e1
    ne2 = float(np.linalg.norm(v2o))
    if ne2 < 1e-6:
        return None
    e2 = v2o / ne2
    e3 = np.cross(e1, e2)
    ne3 = float(np.linalg.norm(e3))
    if ne3 < 1e-6:
        return None
    e3 = e3 / ne3
    B = np.stack([e1, e2, e3], axis=1)
    h2 = np.zeros((21, 3), dtype=np.float64)
    for j in range(21):
        h2[j] = xy(j)
    return _enforce_thumb_positive_x(B, h2, wrist)


def orthonormal_basis_from_landmark_pair(
    h: np.ndarray, ia: int, ib: int, *, min_edge_mm: float = 1e-9
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (wrist, B) from two landmarks minus wrist (camera mm)."""
    wrist = h[WRIST_ID, :3]
    v1 = h[int(ia), :3] - wrist
    v2 = h[int(ib), :3] - wrist
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < float(min_edge_mm) or n2 < float(min_edge_mm):
        return None
    e1 = v1 / n1
    e2 = v2 - float(np.dot(v2, e1)) * e1
    ne2 = float(np.linalg.norm(e2))
    if ne2 < float(min_edge_mm):
        return None
    e2 = e2 / ne2
    e3 = np.cross(e1, e2)
    ne3 = float(np.linalg.norm(e3))
    if ne3 < float(min_edge_mm):
        return None
    e3 = e3 / ne3
    B = np.stack([e1, e2, e3], axis=1)
    return wrist, B


def _segment_axis(h: np.ndarray, wrist: np.ndarray, mcp_idx: int, tip_idx: int) -> np.ndarray | None:
    """Unit vector wrist → landmark; prefer tip when both segments are well observed."""
    v_mcp = np.asarray(h[int(mcp_idx), :3], dtype=np.float64).reshape(3) - wrist
    v_tip = np.asarray(h[int(tip_idx), :3], dtype=np.float64).reshape(3) - wrist
    n_mcp, n_tip = float(np.linalg.norm(v_mcp)), float(np.linalg.norm(v_tip))
    v = v_mcp
    if n_tip >= 15.0 and n_mcp >= 15.0 and 0.45 < (n_tip / max(n_mcp, 1e-9)) < 2.2:
        v = v_tip
    nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return None
    return v / nv


def _middle_finger_axis(h: np.ndarray, wrist: np.ndarray) -> np.ndarray | None:
    """+Y palm: wrist → middle fingertip (tip when reliable)."""
    return _segment_axis(h, wrist, MIDDLE_MCP_ID, MIDDLE_TIP_ID)


def _thumb_lateral_axis(h: np.ndarray, wrist: np.ndarray) -> np.ndarray | None:
    """+X palm: thumb direction (tip when reliable)."""
    return _segment_axis(h, wrist, THUMB_MCP_ID, THUMB_TIP_ID)


def _enforce_thumb_positive_x(B: np.ndarray, h: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Palm +X toward thumb; keep +Y fixed and recompute +Z = X×Y (right-handed)."""
    out = np.asarray(B, dtype=np.float64).reshape(3, 3).copy()
    thumb = _thumb_lateral_axis(h, wrist)
    if thumb is None:
        thumb = np.asarray(h[THUMB_MCP_ID, :3], dtype=np.float64).reshape(3) - wrist
    if float(np.dot(out[:, 0], thumb)) < 0.0:
        out[:, 0] *= -1.0
    ez = np.cross(out[:, 0], out[:, 1])
    nez = float(np.linalg.norm(ez))
    if nez > 1e-9:
        out[:, 2] = ez / nez
    return out


def _project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Remove component along ``plane_normal``."""
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return np.asarray(v, dtype=np.float64).reshape(3).copy()
    n = n / nn
    return np.asarray(v, dtype=np.float64).reshape(3) - float(np.dot(v, n)) * n


def _build_palm_basis_middle_y_thumb_x(
    ey: np.ndarray,
    h: np.ndarray,
    wrist: np.ndarray,
    *,
    plane_n: np.ndarray | None = None,
) -> np.ndarray | None:
    """Orthonormal palm basis (camera mm): **+Y** wrist→middle fingertip, **+X** thumb lateral, **+Z** = X×Y.

    When ``plane_n`` is set, +X/+Y are built in the fitted palm plane (reduces depth-noise tilt).
    ``+Y`` is never flipped to chase camera/ref continuity; palm vs back is represented by
    the resulting ``+Z`` sign. ``+X`` is chosen on the thumb side.
    """
    ey_u = np.asarray(ey, dtype=np.float64).reshape(3)
    if plane_n is not None:
        pn = np.asarray(plane_n, dtype=np.float64).reshape(3)
        pn = pn / max(float(np.linalg.norm(pn)), 1e-9)
        ey_u = _project_onto_plane(ey_u, pn)
    ney = float(np.linalg.norm(ey_u))
    if ney < 1e-9:
        return None
    ey_u = ey_u / ney
    thumb = _thumb_lateral_axis(h, wrist)
    if thumb is None:
        thumb = np.asarray(h[THUMB_MCP_ID, :3], dtype=np.float64).reshape(3) - wrist
    if plane_n is not None:
        pn = np.asarray(plane_n, dtype=np.float64).reshape(3)
        pn = pn / max(float(np.linalg.norm(pn)), 1e-9)
        thumb = _project_onto_plane(thumb, pn)
    ex = _project_onto_plane(thumb, ey_u)
    nex = float(np.linalg.norm(ex))
    if nex < 1e-6:
        index = np.asarray(h[INDEX_MCP_ID, :3], dtype=np.float64).reshape(3) - wrist
        ex = _project_onto_plane(index, ey_u)
        nex = float(np.linalg.norm(ex))
    if nex < 1e-9:
        return None
    ex = ex / nex
    ez = np.cross(ex, ey_u)
    nez = float(np.linalg.norm(ez))
    if nez < 1e-9:
        return None
    ez = ez / nez
    B = np.stack([ex, ey_u, ez], axis=1)
    return _enforce_thumb_positive_x(B, h, wrist)


def align_palm_basis_to_reference(
    B: np.ndarray,
    B_ref: np.ndarray,
    h: np.ndarray,
    wrist: np.ndarray,
    *,
    plane_n: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild with physical +Y preserved; palm/back changes appear as +Z changes."""
    del B_ref
    ey = np.asarray(B[:, 1], dtype=np.float64).reshape(3).copy()
    rebuilt = _build_palm_basis_middle_y_thumb_x(ey, h, wrist, plane_n=plane_n)
    if rebuilt is not None:
        return rebuilt
    return _enforce_thumb_positive_x(B, h, wrist)


def palm_orthonormal_basis_middle_y_thumb_x(
    h: np.ndarray,
    *,
    ref_basis: np.ndarray | None = None,
    palm_center_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Palm frame (camera mm): **+Y** wrist→middle fingertip, **+X** thumb lateral, **+Z** = X×Y."""
    wrist = np.asarray(h[WRIST_ID, :3], dtype=np.float64).reshape(3)
    ey = _middle_finger_axis(h, wrist)
    if ey is None:
        return None
    plane_n: np.ndarray | None = None
    fit = palm_plane_fit_mm(h)
    if fit is not None:
        plane_n, _, _ = fit
    B = _build_palm_basis_middle_y_thumb_x(ey, h, wrist, plane_n=plane_n)
    if B is None:
        return None
    if ref_basis is not None:
        B = align_palm_basis_to_reference(B, ref_basis, h, wrist, plane_n=plane_n)
    pc = palm_frame_origin_mm(h)
    if palm_center_override is not None:
        pc_ov = np.asarray(palm_center_override, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(pc_ov)):
            if plane_n is not None and fit is not None:
                _, hp, _ = fit
                pc = _project_point_onto_plane(pc_ov, plane_n, hp)
            else:
                pc = pc_ov
    if pc is None:
        pc = wrist
    return np.asarray(pc, dtype=np.float64).reshape(3), B


def palm_orthonormal_basis(
    h: np.ndarray,
    *,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
    ref_basis: np.ndarray | None = None,
    palm_center_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (palm_center_mm, B) with B columns = palm X (thumb), Y (fingertip), Z."""
    key = str(palm_basis).strip().lower()
    if key in ("middle_thumb", "middle_y", "middle"):
        return palm_orthonormal_basis_middle_y_thumb_x(
            h, ref_basis=ref_basis, palm_center_override=palm_center_override
        )
    ia, ib = palm_basis_pair_indices(palm_basis)
    out = orthonormal_basis_from_landmark_pair(h, ia, ib)
    if out is None:
        return None
    origin, B = out
    if ref_basis is not None:
        wrist = np.asarray(h[WRIST_ID, :3], dtype=np.float64).reshape(3)
        B = align_palm_basis_to_reference(B, ref_basis, h, wrist)
    pc = palm_frame_origin_mm(h)
    if palm_center_override is not None:
        pc_ov = np.asarray(palm_center_override, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(pc_ov)):
            pc = pc_ov
    if pc is None:
        pc = origin
    return np.asarray(pc, dtype=np.float64).reshape(3), B


@dataclass
class LeftSwarmPoseState:
    """Tracks reference palm frame and outputs smoothed world offset + full rotation."""

    enabled: bool = True
    initialized: bool = False
    ref_wrist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ref_basis: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    ema_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ema_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Smooth return-to-morph-frame after disarm (seconds); 0 = not unwinding
    unwind_end_t: float = 0.0
    unwind_duration: float = 0.0
    unwind_off0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    unwind_rv0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Previous wrist (camera mm), used for debug / glitch checks.
    prev_wrist: np.ndarray | None = None
    #: Previous palm center (camera mm), used for frame-to-frame translation delta.
    prev_palm_mm: np.ndarray | None = None
    #: Palm basis at previous frame (3×3), kept for legacy/full incremental helpers.
    prev_basis: np.ndarray | None = None
    #: Hand span (mm) at previous frame for incremental push/pull scale.
    prev_hand_span_mm: float = 0.0
    #: When set (``camera_at_arm``), maps gated camera mm → sim m; frozen at press-0, not updated per frame.
    frozen_M_rot: np.ndarray | None = None
    frozen_M_trans: np.ndarray | None = None
    frozen_cam_preset: str = ""
    #: Swarm XYZ at arm (sim m); rigid motion is applied to this snapshot, not live morph.
    ref_swarm_targets: np.ndarray | None = None
    #: Per-drone morph XYZ at arm (sim m) for ``per_drone`` rotation pivot (self-rotation).
    ref_drone_xyz: np.ndarray | None = None
    #: 2D/webcam palm basis at arm when dual-rotation fallback is enabled.
    ref_basis_image: np.ndarray | None = None
    #: Mean wrist→landmark span (mm) at arm; detects push/pull via apparent hand size.
    ref_hand_span_mm: float = 0.0
    #: Wrist (camera mm) at arm — debug / overlay only.
    ref_wrist_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Palm centroid (camera mm) at arm — translation origin.
    ref_palm_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Palm basis ΔR this frame (world rad); for HUD — may differ from applied ``rv_cmd``.
    last_rv_pose_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_rv_cmd_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_h_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_cam_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_cam_arm_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_h_raw_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_palm_center_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: EMA-smoothed palm center (camera mm) after depth outlier rejection.
    filtered_palm_mm: np.ndarray | None = None
    last_depth_outlier: bool = False
    last_depth_outlier_prev: bool = False
    last_wrist_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_mcp_center_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_mcp_valid_count: int = 0
    last_palm_center_color_px: tuple[int, int] | None = None
    last_palm_color_u: float | None = None
    last_palm_color_v: float | None = None
    last_delta_trans_palm_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_rv_cam_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_trans_cam_gated: bool = False
    #: Previous Orbbec MP min visibility (dual-rot); used to hold morph mode when occluded.
    last_orbbec_vis_min: float = 1.0
    last_dual_rot_source: str = "depth"
    last_dual_vis_min: float = 1.0
    last_dual_vis_thresh: float = 0.0
    last_pose_rejected: bool = False
    last_reject_reason: str = ""
    #: Previous palm basis (always updated) for frame-to-frame rot_jump checks only.
    basis_step_prev: np.ndarray | None = None
    #: Previous basis from the rotation source actually used (depth vs image/webcam).
    prev_rot_basis: np.ndarray | None = None
    prev_rot_source: str = "depth"
    #: Last classified motion: ``translate`` | ``rotate`` | ``none`` (mode-switch freeze).
    last_axis_motion: str = "none"
    last_rot_source: str = "depth"
    last_rot_blend_w: float = 0.0
    last_trans_blend_w: float = 0.0

    def reset_to_current(
        self,
        h: np.ndarray,
        *,
        palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
        sim_from_cam: np.ndarray | None = None,
        sim_trans_from_cam: np.ndarray | None = None,
        cam_preset_label: str = "",
        ref_drone_xyz: np.ndarray | None = None,
        ref_swarm_targets: np.ndarray | None = None,
        ref_basis_image: np.ndarray | None = None,
        palm_center_override: np.ndarray | None = None,
    ) -> bool:
        out = palm_orthonormal_basis(
            h, palm_basis=palm_basis, palm_center_override=palm_center_override
        )
        if out is None:
            return False
        origin, B = out
        wrist_mm = np.asarray(h[WRIST_ID, :3], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(wrist_mm)):
            return False
        B = _enforce_thumb_positive_x(B, h, wrist_mm)
        pc = np.asarray(origin, dtype=np.float64).reshape(3)
        if palm_center_override is not None:
            pc_ov = np.asarray(palm_center_override, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(pc_ov)):
                pc = pc_ov
        self.ref_wrist_mm = wrist_mm.copy()
        self.ref_wrist = wrist_mm.copy()
        self.ref_palm_center = np.asarray(pc, dtype=np.float64).reshape(3).copy()
        self.ref_basis = B.copy()
        self.initialized = True
        self.ema_offset[:] = 0.0
        self.ema_rotvec[:] = 0.0
        self.prev_wrist = wrist_mm.copy()
        self.prev_palm_mm = pc.copy()
        self.prev_basis = B.copy()
        self.basis_step_prev = B.copy()
        self.prev_rot_basis = B.copy()
        self.prev_rot_source = "depth"
        self.last_rot_source = "depth"
        self.prev_hand_span_mm = float(self.ref_hand_span_mm)
        self.last_palm_center_mm = pc.copy()
        self.last_delta_cam_mm[:] = 0.0
        self.last_delta_cam_arm_mm[:] = 0.0
        self.last_delta_trans_palm_mm[:] = 0.0
        self.last_delta_h_world[:] = 0.0
        self.last_delta_h_raw_m[:] = 0.0
        self.last_rv_pose_world[:] = 0.0
        self.last_rv_cmd_world[:] = 0.0
        if sim_from_cam is not None:
            self.frozen_M_rot = np.asarray(sim_from_cam, dtype=np.float64).reshape(3, 3).copy()
            if sim_trans_from_cam is not None:
                self.frozen_M_trans = np.asarray(sim_trans_from_cam, dtype=np.float64).reshape(3, 3).copy()
            else:
                self.frozen_M_trans = self.frozen_M_rot.copy()
            self.frozen_cam_preset = str(cam_preset_label)
        else:
            self.frozen_M_rot = None
            self.frozen_M_trans = None
            self.frozen_cam_preset = ""
        if ref_drone_xyz is not None:
            rd = np.asarray(ref_drone_xyz, dtype=np.float64)
            if rd.ndim == 2 and rd.shape[1] >= 3:
                self.ref_drone_xyz = rd[:, :3].copy()
        else:
            self.ref_drone_xyz = None
        if ref_swarm_targets is not None:
            rs = np.asarray(ref_swarm_targets, dtype=np.float64)
            if rs.ndim == 2 and rs.shape[1] >= 3:
                self.ref_swarm_targets = rs.astype(np.float32, copy=True)
        else:
            self.ref_swarm_targets = None
        if ref_basis_image is not None:
            self.ref_basis_image = np.asarray(ref_basis_image, dtype=np.float64).reshape(3, 3).copy()
        else:
            self.ref_basis_image = None
        self.ref_hand_span_mm = float(hand_palm_span_mm(h, wrist_mm))
        self.filtered_palm_mm = np.asarray(pc, dtype=np.float64).reshape(3).copy()
        self.last_depth_outlier = False
        return True

    def is_unwinding(self) -> bool:
        return float(self.unwind_end_t) > 0.0 and time.monotonic() < float(self.unwind_end_t)

    def begin_unwind(self, duration_s: float) -> None:
        """Fade rigid offset/rotation to identity over ``duration_s`` (smoothstep)."""
        d = float(max(duration_s, 1e-3))
        self.unwind_off0 = np.asarray(self.ema_offset, dtype=np.float64).copy()
        self.unwind_rv0 = np.asarray(self.ema_rotvec, dtype=np.float64).copy()
        self.unwind_duration = d
        self.unwind_end_t = time.monotonic() + d

    def cancel_unwind(self) -> None:
        """Abort smooth restore (e.g. user re-arms); clears offset like disarm."""
        self.unwind_end_t = 0.0
        self.unwind_duration = 0.0
        self.unwind_off0[:] = 0.0
        self.unwind_rv0[:] = 0.0
        self.initialized = False
        self.ema_offset[:] = 0.0
        self.ema_rotvec[:] = 0.0
        self.prev_wrist = None
        self.prev_palm_mm = None
        self.prev_basis = None
        self.basis_step_prev = None
        self.prev_rot_basis = None
        self.prev_rot_source = "depth"
        self.prev_hand_span_mm = 0.0
        self.last_axis_motion = "none"
        self.last_rot_source = "depth"
        self.last_dual_rot_source = "depth"
        self.last_dual_vis_min = 1.0
        self.last_dual_vis_thresh = 0.0
        self.last_rot_blend_w = 0.0
        self.last_trans_blend_w = 0.0
        self.frozen_M_rot = None
        self.frozen_M_trans = None
        self.frozen_cam_preset = ""
        self.ref_drone_xyz = None
        self.ref_swarm_targets = None
        self.ref_basis_image = None


def _clear_frozen_cam_to_sim(state: LeftSwarmPoseState) -> None:
    state.frozen_M_rot = None
    state.frozen_M_trans = None
    state.frozen_cam_preset = ""
    state.ref_drone_xyz = None
    state.ref_swarm_targets = None
    state.ref_basis_image = None


def swarm_base_targets(state: LeftSwarmPoseState, morph_fallback: np.ndarray) -> np.ndarray:
    """Live morph/open targets each frame; rigid transform applies on top when armed."""
    return np.asarray(morph_fallback, dtype=np.float32)









def disarm_left_swarm_pose(state: LeftSwarmPoseState) -> None:
    """Exit move mode: drop reference so the next arm captures a fresh baseline."""
    state.unwind_end_t = 0.0
    state.unwind_duration = 0.0
    state.unwind_off0[:] = 0.0
    state.unwind_rv0[:] = 0.0
    state.initialized = False
    state.ema_offset[:] = 0.0
    state.ema_rotvec[:] = 0.0
    state.prev_wrist = None
    state.prev_palm_mm = None
    state.prev_basis = None
    state.basis_step_prev = None
    state.prev_rot_basis = None
    state.prev_rot_source = "depth"
    state.prev_hand_span_mm = 0.0
    state.last_axis_motion = "none"
    state.last_rot_source = "depth"
    state.last_dual_rot_source = "depth"
    state.last_dual_vis_min = 1.0
    state.last_dual_vis_thresh = 0.0
    state.last_rot_blend_w = 0.0
    state.last_trans_blend_w = 0.0
    _clear_frozen_cam_to_sim(state)


def update_left_swarm_pose(
    pts_l,
    state: "LeftSwarmPoseState",
    *,
    trans_scale: float,
    rot_scale: float,
    trans_ema: float,
    rot_ema: float,
    max_offset_m: float,
    max_rot_rad: float,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hand_lost_decay: float = 0.92,
    force_reset: bool = False,
    cam_delta_to_world: np.ndarray | None = None,
    cam_translation_to_world: np.ndarray | None = None,
    rot_gate_rad: float = 0.11,
    rot_gain: float = 1.0,
    rot_trans_tau_mm: float = 0.0,
    max_trans_step_m: float = 0.055,
    rot_world_z_scale: float = 1.0,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
    arm_sim_from_cam: np.ndarray | None = None,
    arm_sim_trans_from_cam: np.ndarray | None = None,
    arm_cam_preset_label: str = "",
    ref_drone_xyz: np.ndarray | None = None,
    ref_swarm_xyz: np.ndarray | None = None,
    ref_basis_image: np.ndarray | None = None,
    B_rot: np.ndarray | None = None,
    rot_ref_basis: np.ndarray | None = None,
    trans_deadzone_m: float = 0.018,
    rot_deadzone_rad: float = 0.055,
    trans_on_m: float = 0.004,
    rot_on_rad: float = 0.020,
    trans_rot_coupling: float = 0.50,
    palm_center_mm: np.ndarray | None = None,
    palm_center_color_px: tuple[int, int] | None = None,
    palm_depth_outlier_z_mm: float = 95.0,
    palm_depth_outlier_lateral_ratio: float = 2.2,
    palm_center_depth_ema: float = 0.42,
    palm_depth_patch_r: int = 2,
    palm_calib: object | None = None,
    palm_frame_h: int = 0,
    palm_frame_w: int = 0,
    palm_depth_aligned: object | None = None,
    palm_depth_raw: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid follow: direct arm-relative (offset, R) with frame reject + slerp smoothing."""
    if not state.enabled:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    now = time.monotonic()
    if float(state.unwind_end_t) > 0.0:
        if now >= float(state.unwind_end_t):
            state.unwind_end_t = 0.0
            state.unwind_duration = 0.0
            state.initialized = False
            state.ema_offset[:] = 0.0
            state.ema_rotvec[:] = 0.0
            state.prev_wrist = None
            state.prev_palm_mm = None
            state.prev_basis = None
            state.basis_step_prev = None
            state.prev_rot_basis = None
            state.prev_rot_source = "depth"
            state.prev_hand_span_mm = 0.0
            _clear_frozen_cam_to_sim(state)
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        u = float(
            np.clip(
                (now - (state.unwind_end_t - state.unwind_duration))
                / max(state.unwind_duration, 1e-9),
                0.0,
                1.0,
            )
        )
        s = u * u * (3.0 - 2.0 * u)
        a = 1.0 - s
        state.ema_offset[:] = state.unwind_off0 * a
        state.ema_rotvec[:] = state.unwind_rv0 * a
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)

    h = hand_points_to_matrix(pts_l)
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)

    if h is None:
        state.prev_wrist = None
        state.prev_palm_mm = None
        state.prev_basis = None
        state.basis_step_prev = None
        state.prev_rot_basis = None
        state.prev_rot_source = "depth"
        state.prev_hand_span_mm = 0.0
        ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
        if state.initialized:
            ld = 1.0
        if ld < 1.0:
            state.ema_offset *= ld
            state.ema_rotvec *= ld
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)

    if force_reset:
        state.filtered_palm_mm = None
        state.last_depth_outlier = False
        state.last_depth_outlier_prev = False

    ref_b = state.ref_basis if state.initialized and not force_reset else None
    basis_palm_center_override = palm_center_mm
    out = palm_orthonormal_basis(
        h,
        palm_basis=palm_basis,
        ref_basis=ref_b,
        palm_center_override=basis_palm_center_override,
    )
    if palm_center_color_px is not None:
        state.last_palm_center_color_px = palm_center_color_px
    if out is None:
        state.prev_wrist = None
        state.prev_palm_mm = None
        state.prev_basis = None
        state.basis_step_prev = None
        state.prev_rot_basis = None
        state.prev_rot_source = "depth"
        state.prev_hand_span_mm = 0.0
        ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
        if state.initialized:
            ld = 1.0
        if ld < 1.0:
            state.ema_offset *= ld
            state.ema_rotvec *= ld
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)

    palm_center, B_depth = out
    palm_center = np.asarray(palm_center, dtype=np.float64).reshape(3)
    B_depth = np.asarray(B_depth, dtype=np.float64).reshape(3, 3)
    wrist_mm = np.asarray(h[WRIST_ID, :3], dtype=np.float64).reshape(3)
    _meas_depth_mm: float | None = None
    if palm_center_mm is not None:
        _pc_ext = np.asarray(palm_center_mm, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(_pc_ext)) and float(_pc_ext[2]) > 0.0:
            _meas_depth_mm = float(_pc_ext[2])
    palm_center, _depth_reliable = filter_palm_center_depth_mm(
        palm_center,
        state,
        color_px=palm_center_color_px,
        calibration=palm_calib,
        frame_h=int(palm_frame_h),
        frame_w=int(palm_frame_w),
        depth_aligned=palm_depth_aligned,
        depth_raw=palm_depth_raw,
        depth_patch_r=int(palm_depth_patch_r),
        measured_depth_mm=_meas_depth_mm,
        z_outlier_mm=float(palm_depth_outlier_z_mm),
        z_outlier_lateral_ratio=float(palm_depth_outlier_lateral_ratio),
        ema_alpha=float(palm_center_depth_ema),
    )
    if not np.all(np.isfinite(wrist_mm)):
        state.prev_wrist = None
        state.prev_palm_mm = None
        state.prev_basis = None
        state.basis_step_prev = None
        state.prev_rot_basis = None
        state.prev_rot_source = "depth"
        state.prev_hand_span_mm = 0.0
        ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
        if state.initialized:
            ld = 1.0
        if ld < 1.0:
            state.ema_offset *= ld
            state.ema_rotvec *= ld
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)

    if force_reset or not state.initialized:
        if not state.reset_to_current(
            h,
            palm_basis=palm_basis,
            sim_from_cam=arm_sim_from_cam if force_reset else None,
            sim_trans_from_cam=arm_sim_trans_from_cam if force_reset else None,
            cam_preset_label=str(arm_cam_preset_label) if force_reset else "",
            ref_drone_xyz=ref_drone_xyz if force_reset else None,
            ref_swarm_targets=ref_swarm_xyz if force_reset else None,
            ref_basis_image=ref_basis_image if force_reset else None,
            palm_center_override=palm_center,
        ):
            state.prev_wrist = None
            state.prev_palm_mm = None
            state.prev_basis = None
            state.basis_step_prev = None
            state.prev_rot_basis = None
            state.prev_rot_source = "depth"
            state.prev_hand_span_mm = 0.0
            ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
            if state.initialized:
                ld = 1.0
            if ld < 1.0:
                state.ema_offset *= ld
                state.ema_rotvec *= ld
            return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)
        state.last_palm_center_color_px = palm_center_color_px
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    ref_wrist = np.asarray(state.ref_wrist_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(ref_wrist)):
        ref_wrist = np.asarray(state.ref_wrist, dtype=np.float64).reshape(3)
    ref_pc = np.asarray(state.ref_palm_center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(ref_pc)):
        ref_pc = np.asarray(palm_center, dtype=np.float64).reshape(3)
    delta_cam_arm = np.asarray(palm_center, dtype=np.float64).reshape(3) - ref_pc
    if state.prev_palm_mm is not None:
        delta_cam = np.asarray(palm_center, dtype=np.float64).reshape(3) - np.asarray(
            state.prev_palm_mm, dtype=np.float64
        ).reshape(3)
    else:
        delta_cam = np.zeros(3, dtype=np.float64)

    w_now, mcp_now, mcp_n = palm_center_components_mm(h)
    state.last_palm_center_mm = np.asarray(palm_center, dtype=np.float64).reshape(3).copy()
    if w_now is not None:
        state.last_wrist_mm = np.asarray(w_now, dtype=np.float64).reshape(3).copy()
    if mcp_now is not None:
        state.last_mcp_center_mm = np.asarray(mcp_now, dtype=np.float64).reshape(3).copy()
    state.last_mcp_valid_count = int(mcp_n)
    state.last_delta_cam_mm = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    state.last_delta_cam_arm_mm = np.asarray(delta_cam_arm, dtype=np.float64).reshape(3).copy()
    state.last_delta_trans_palm_mm = np.asarray(delta_cam_arm, dtype=np.float64).reshape(3).copy()
    state.last_trans_cam_gated = True

    rejected, _reject_reason = _reject_noisy_pose_frame(
        state,
        palm_center=palm_center,
        B_depth=B_depth,
        delta_cam=delta_cam,
        delta_cam_arm=delta_cam_arm,
        mcp_valid=int(mcp_n),
        wrist_mm=w_now,
        depth_hold=bool(getattr(state, "last_depth_outlier", False)),
    )

    Mc_rot, Mc_trans = _resolve_cam_world_mats(
        state,
        cam_delta_to_world=cam_delta_to_world,
        cam_translation_to_world=cam_translation_to_world,
    )

    ref_b_depth = np.asarray(state.ref_basis, dtype=np.float64).reshape(3, 3)
    B_depth_stable = stabilize_palm_basis_continuity(B_depth, ref_b_depth)
    B = B_depth_stable
    ref_b_rot = ref_b_depth
    rot_source = "depth"
    rv_world_override = None
    if B_rot is not None and state.ref_basis_image is not None:
        B_img = np.asarray(B_rot, dtype=np.float64).reshape(3, 3)
        ref_img = np.asarray(state.ref_basis_image, dtype=np.float64).reshape(3, 3)
        if np.all(np.isfinite(B_img)) and np.all(np.isfinite(ref_img)):
            B_img = stabilize_palm_basis_continuity(B_img, ref_img)
            rv_depth_local = palm_local_rotvec_from_basis_delta(B_depth_stable, ref_b_depth)
            rv_img_local = palm_local_rotvec_from_basis_delta(B_img, ref_img)
            rv_hybrid_local = np.array(
                [float(rv_depth_local[0]), float(rv_depth_local[1]), float(rv_img_local[2])],
                dtype=np.float64,
            )
            rv_world_override = palm_world_rotvec_from_local_intrinsic(rv_hybrid_local)
            B = B_img
            ref_b_rot = ref_img
            rot_source = "hybrid"

    off_tgt, R_tgt, off_raw, rv_world = _rigid_target_from_hand(
        state,
        delta_cam_arm=delta_cam_arm,
        B=B,
        ref_b_rot=ref_b_rot,
        rv_world_override=rv_world_override,
        Mc_rot=Mc_rot,
        Mc_trans=Mc_trans,
        sign=sign,
        trans_scale=float(trans_scale),
        rot_scale=float(rot_scale),
        rot_gain=float(rot_gain),
        rot_world_z_scale=float(rot_world_z_scale),
        trans_deadzone_m=float(trans_deadzone_m),
        rot_deadzone_rad=float(rot_deadzone_rad),
    )
    rv_cam = palm_cam_rotvec_from_basis_delta(B, ref_b_rot)
    step_cap = float(max_rot_rad) if max_rot_rad > 0.0 else np.deg2rad(28.0)
    rv_apply = sanitize_palm_rotvec_apply(
        rv_world,
        rv_cam,
        prev_basis=state.prev_rot_basis
        if str(getattr(state, "prev_rot_source", "depth")) == rot_source
        else None,
        B_current=B,
        Mc_rot=Mc_rot,
        max_step_rad=step_cap,
    )
    if float(np.linalg.norm(rv_apply - rv_world)) > 1e-9:
        R_world = rotvec_to_R(rv_apply)
        zsc = float(rot_world_z_scale)
        if zsc != 1.0 and float(np.linalg.norm(rv_apply)) >= 1e-9:
            rv_z = rv_apply.copy()
            rv_z[2] *= zsc
            R_world = rotvec_to_R(rv_z)
        R_tgt = scale_rotation_matrix(R_world, scale=float(rot_scale), gain=float(rot_gain))
        rv_out = np.asarray(R_to_rotvec(R_tgt), dtype=np.float64).reshape(3)
        if float(np.linalg.norm(rv_out)) < float(rot_deadzone_rad):
            R_tgt = np.eye(3, dtype=np.float64)

    motion, w_rot, w_trans = axis_locked_trans_rot_blend_weights(
        off_tgt,
        rv_apply,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(max(rot_on_rad, rot_gate_rad)),
        rv_cam_rad=rv_cam,
        delta_cam_mm=delta_cam,
        delta_trans_mm=delta_cam_arm,
        secondary_frac=float(trans_rot_coupling),
    )
    rv_apply_norm = float(np.linalg.norm(rv_apply))
    off_tgt_norm = float(np.linalg.norm(np.asarray(off_tgt, dtype=np.float64).reshape(3)))
    if rv_apply_norm >= float(rot_gate_rad):
        # Core rigid-pose behavior: translation and rotation are not mutually exclusive.
        # The classifier only labels the dominant motion for debug; it must not erase a real palm rotation.
        w_rot = 1.0
        if off_tgt_norm >= float(trans_on_m):
            motion = "rigid"
    else:
        w_rot = 0.0
    if motion == "rotate" and float(rot_trans_tau_mm) > 0.0:
        if float(np.linalg.norm(delta_cam_arm)) < float(rot_trans_tau_mm):
            w_trans = 0.0
    if rejected and _reject_reason == "jump":
        motion = "hold_jump"
        w_trans = 0.0
        w_rot = 0.0
        off_tgt = np.asarray(state.ema_offset, dtype=np.float64).reshape(3).copy()
        R_tgt = rotvec_to_R(state.ema_rotvec)
    else:
        off_tgt = np.asarray(off_tgt, dtype=np.float64).reshape(3) * float(w_trans)
        R_tgt = scale_rotation_matrix(R_tgt, scale=float(w_rot), gain=1.0)
    state.last_axis_motion = str(motion)
    state.last_rot_source = str(rot_source)
    state.last_rot_blend_w = float(w_rot)
    state.last_trans_blend_w = float(w_trans)
    state.last_delta_h_raw_m = np.asarray(off_raw * w_trans, dtype=np.float64).reshape(3).copy()
    state.last_rv_cam_world = np.asarray(rv_cam, dtype=np.float64).reshape(3).copy()
    state.last_rv_pose_world = np.asarray(rv_world, dtype=np.float64).reshape(3).copy()
    state.last_delta_h_world = np.asarray(off_tgt, dtype=np.float64).reshape(3).copy()
    state.last_rv_cmd_world = np.asarray(R_to_rotvec(R_tgt), dtype=np.float64).reshape(3).copy()

    R_hold = rotvec_to_R(state.ema_rotvec)
    off_hold = np.asarray(state.ema_offset, dtype=np.float64).reshape(3)
    tb = 1.0 if float(trans_ema) <= 0.0 else float(np.clip(trans_ema, 0.15, 1.0))
    rb = 1.0 if float(rot_ema) <= 0.0 else float(np.clip(rot_ema, 0.15, 1.0))
    if rejected:
        state.last_pose_rejected = True
        state.last_reject_reason = str(_reject_reason)
        if _reject_reason in ("jump",):
            tb *= 0.55
            rb *= 0.50
        else:
            tb *= 0.72
            rb *= 0.65
    else:
        state.last_pose_rejected = False
        state.last_reject_reason = ""

    step_cap = float(max_rot_rad) if max_rot_rad > 0.0 else np.deg2rad(28.0)
    off_out, R_out = _smooth_rigid_pose(
        off_hold,
        R_hold,
        off_tgt,
        R_tgt,
        trans_blend=tb,
        rot_blend=rb,
        max_step_rad=step_cap,
        max_offset_m=float(max_offset_m),
        max_trans_step_m=float(max_trans_step_m),
    )
    sync_left_swarm_pose_output(state, off_out, R_out)

    state.basis_step_prev = np.asarray(B, dtype=np.float64).reshape(3, 3).copy()
    state.last_depth_outlier_prev = bool(getattr(state, "last_depth_outlier", False))
    if not rejected or _reject_reason not in ("jump",):
        state.prev_wrist = wrist_mm.copy()
        state.prev_palm_mm = np.asarray(palm_center, dtype=np.float64).reshape(3).copy()
        state.prev_basis = B.copy()
        state.prev_rot_basis = B.copy()
        state.prev_rot_source = str(rot_source)
        state.prev_hand_span_mm = float(hand_palm_span_mm(h, wrist_mm))

    return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)


def _vec3_s(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"({a[0]:+.1f},{a[1]:+.1f},{a[2]:+.1f})"


def print_left_swarm_pose_debug(
    state: LeftSwarmPoseState,
    *,
    frame_idx: int = 0,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
    trans_scale: float = 0.012,
) -> None:
    """Stdout debug: palm center (camera mm), palm/world deltas, pose R."""
    if not state.initialized:
        return
    ref_w = np.asarray(state.ref_wrist_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(ref_w)):
        ref_w = np.asarray(state.ref_wrist, dtype=np.float64).reshape(3)
    ref_pc = np.asarray(state.ref_palm_center, dtype=np.float64).reshape(3)
    pc = np.asarray(state.last_palm_center_mm, dtype=np.float64).reshape(3)
    w = np.asarray(state.last_wrist_mm, dtype=np.float64).reshape(3)
    mcp = np.asarray(state.last_mcp_center_mm, dtype=np.float64).reshape(3)
    mcp_n = int(state.last_mcp_valid_count)
    dc = np.asarray(state.last_delta_cam_mm, dtype=np.float64).reshape(3)
    dc_arm = np.asarray(state.last_delta_cam_arm_mm, dtype=np.float64).reshape(3)
    dp = np.asarray(state.last_delta_trans_palm_mm, dtype=np.float64).reshape(3)
    dh_raw = np.asarray(state.last_delta_h_raw_m, dtype=np.float64).reshape(3)
    dh_cmd = np.asarray(state.last_delta_h_world, dtype=np.float64).reshape(3)
    off = np.asarray(state.ema_offset, dtype=np.float64).reshape(3)
    rv_pose = np.asarray(state.last_rv_pose_world, dtype=np.float64).reshape(3)
    rv_cmd = np.asarray(state.last_rv_cmd_world, dtype=np.float64).reshape(3)
    R = rotvec_to_R(state.ema_rotvec)
    ang = float(np.degrees(np.linalg.norm(state.ema_rotvec)))
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)
    wx, wy, wz = float(dh_raw[0]), float(dh_raw[1]), float(dh_raw[2])
    trans_tag = "palm center cam→world"
    sep = "=" * 72
    print(sep, flush=True)
    print(
        f"[left-pose debug] frame={int(frame_idx)} rigid"
        f"{' hold' if bool(getattr(state, 'last_pose_rejected', False)) else ''}"
        f"{(' reason=' + str(getattr(state, 'last_reject_reason', ''))) if getattr(state, 'last_pose_rejected', False) else ''}"
        f"  axis_sign={tuple(float(x) for x in sign)}"
        f"  motion={getattr(state, 'last_axis_motion', 'none')}"
        f"  rot_src={getattr(state, 'last_rot_source', 'depth')}"
        f"  dual_src={getattr(state, 'last_dual_rot_source', 'depth')}"
        f" vis_min={getattr(state, 'last_dual_vis_min', 1.0):.2f}"
        f"/th={getattr(state, 'last_dual_vis_thresh', 0.0):.2f}"
        f"  blend(wT={getattr(state, 'last_trans_blend_w', 0.0):.2f},wR={getattr(state, 'last_rot_blend_w', 0.0):.2f})",
        flush=True,
    )
    print(
        f"  wrist (cam mm)  arm={_vec3_s(ref_w)}  now={_vec3_s(w)}  "
        f"Δframe={_vec3_s(w - ref_w if np.all(np.isfinite(w)) and np.all(np.isfinite(ref_w)) else dc)} "
        f"|Δ|={np.linalg.norm(w - ref_w if np.all(np.isfinite(w)) and np.all(np.isfinite(ref_w)) else dc):.1f}mm  "
        f"(debug only, not used for trans)",
        flush=True,
    )
    print(
        f"  palm center (cam mm)  arm={_vec3_s(ref_pc)}  now={_vec3_s(pc)}  "
        f"Δframe={_vec3_s(dc)} |Δ|={np.linalg.norm(dc):.1f}mm  "
        f"Δarm={_vec3_s(dc_arm)} |Δarm|={np.linalg.norm(dc_arm):.1f}mm  dz={dc[2]:+.1f}",
        flush=True,
    )
    print(
        f"  origin parts (cam mm) wrist={_vec3_s(w)}  roots_mean(n={mcp_n}/5)={_vec3_s(mcp)}  on_plane={_vec3_s(pc)}",
        flush=True,
    )
    px = getattr(state, "last_palm_center_color_px", None)
    if px is not None:
        print(f"  origin 2D centroid (color px) u,v=({int(px[0])},{int(px[1])})", flush=True)
    if bool(getattr(state, "last_depth_outlier", False)):
        print("  depth=hold (using previous Z)", flush=True)
    rv_c = np.asarray(state.last_rv_cam_world, dtype=np.float64).reshape(3)
    print(f"  rv_cam={_format_rotvec(rv_c)}", flush=True)
    print(
        f"  trans {trans_tag} arm_mm={_vec3_s(dc_arm)}  → world m (X,Y,Z)=({wx:+.4f},{wy:+.4f},{wz:+.4f})  "
        f"cmd={_vec3_s(dh_cmd)}",
        flush=True,
    )
    print(
        f"  world cmd Δ (snap×blend) = {_vec3_s(dh_cmd)}  ema_offset T={_vec3_s(off)}  |T|={np.linalg.norm(off):.3f}m",
        flush=True,
    )
    print(
        f"  rot  rv_pose(deg·ax)={_format_rotvec(rv_pose)}  rv_cmd={_format_rotvec(rv_cmd)}  "
        f"accum_angle≈{ang:.1f}°",
        flush=True,
    )
    print(
        f"  R_pose columns Xw={_vec3_s(R[:, 0])} Yw={_vec3_s(R[:, 1])} Zw={_vec3_s(R[:, 2])}",
        flush=True,
    )
    print(sep, flush=True)


def _format_rotvec(rv: np.ndarray) -> str:
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-7:
        return "0"
    ax = rv / ang
    return f"{np.degrees(ang):+.1f}° @({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"


def apply_rigid_to_targets(
    targets: np.ndarray,
    offset: np.ndarray,
    R: np.ndarray,
    *,
    ref_drone_xyz: np.ndarray | None = None,
    pivot: str = "per_drone",
) -> np.ndarray:
    """Apply world translation ``offset`` and rotation ``R``.

    ``pivot="centroid"`` (default): ``p' = R @ (p - c) + c + off`` with ``c = mean(p)`` — the
    formation spins rigidly about its current centroid (in-place rotation + world ``off``).

    ``pivot="per_drone"``: ``p' = ref_i + R @ (p - ref_i) + off`` — each drone pivots about its
    own arm-time slot instead of a shared center.
    """
    t = np.asarray(targets, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 3:
        return np.asarray(targets, dtype=np.float32)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    off = np.asarray(offset, dtype=np.float64).reshape(3)
    p = t[:, :3]
    key = str(pivot).strip().lower()
    if key == "per_drone" and ref_drone_xyz is not None:
        ref = np.asarray(ref_drone_xyz, dtype=np.float64)
        if ref.ndim == 2 and ref.shape[0] == p.shape[0] and ref.shape[1] >= 3:
            rel = p - ref[:, :3]
            out = ref[:, :3] + (R @ rel.T).T + off
        else:
            key = "centroid"
    if key != "per_drone":
        c = np.mean(p, axis=0)
        rel = p - c
        out = (R @ rel.T).T + c + off
    t2 = t.copy()
    t2[:, :3] = out
    return t2.astype(np.float32)
