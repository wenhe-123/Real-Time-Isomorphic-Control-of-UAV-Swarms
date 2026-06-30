"""Palm depth geometry, plane fit, and orthonormal basis from landmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from functions.mode_switch.hand_constants import (
    INDEX_MCP_ID,
    INDEX_TIP_ID,
    MCP_IDS,
    MIDDLE_DEPTH_CHAIN_IDS,
    MIDDLE_DIP_ID,
    MIDDLE_MCP_ID,
    MIDDLE_PIP_ID,
    MIDDLE_TIP_ID,
    PALM_CENTER_IDS,
    RING_MCP_ID,
    RING_TIP_ID,
    THUMB_DEPTH_CHAIN_IDS,
    THUMB_MCP_ID,
    THUMB_TIP_ID,
    WRIST_ID,
)
from functions.swarm_motion.left_pose_config import (
    DEFAULT_LEFT_PALM_BASIS,
    _PALM_PLANE_CORE_IDS,
    _PALM_PLANE_MAX_MEDIAN_RESID_MM,
    _PALM_PLANE_MIN_POINTS,
    _PALM_PLANE_OPTIONAL_TIP_IDS,
    _PALM_PLANE_OUTLIER_MM,
    _PALM_PLANE_RAY_Z0_MM,
    _PALM_PLANE_RAY_Z1_MM,
    _PALM_PLANE_Z_MAX_MM,
    _PALM_PLANE_Z_MIN_MM,
    _PALM_ROOT_IDS,
    palm_basis_pair_indices,
)

if TYPE_CHECKING:
    from functions.swarm_motion.left_swarm_pose_state import LeftSwarmPoseState

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


def _resolve_palm_origin_mm(
    h: np.ndarray,
    *,
    palm_center_override: np.ndarray | None = None,
) -> np.ndarray | None:
    """Palm-frame origin in camera mm (override → plane centroid → wrist)."""
    if palm_center_override is not None:
        pc_ov = np.asarray(palm_center_override, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(pc_ov)):
            return pc_ov
    pc = palm_frame_origin_mm(h)
    if pc is not None:
        return np.asarray(pc, dtype=np.float64).reshape(3)
    wrist = _landmark_mm_if_valid(h, WRIST_ID)
    if wrist is not None:
        return np.asarray(wrist, dtype=np.float64).reshape(3)
    return None


def _palm_basis_origin_mm(
    h: np.ndarray,
    *,
    palm_center_override: np.ndarray | None = None,
) -> np.ndarray | None:
    """Origin for palm **axes** — from landmark geometry in ``h``, not translation EMA."""
    pc = palm_frame_origin_mm(h)
    if pc is not None:
        return np.asarray(pc, dtype=np.float64).reshape(3)
    return _resolve_palm_origin_mm(h, palm_center_override=palm_center_override)


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
    z_outlier_mm: float,
    z_outlier_lateral_ratio: float,
    ema_alpha: float,
    color_px: tuple[int, int] | None = None,
    calibration=None,
    frame_h: int = 0,
    frame_w: int = 0,
    depth_aligned=None,
    depth_raw=None,
    depth_patch_r: int = 2,
    measured_depth_mm: float | None = None,
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


def _depth_chain_for_landmark(jidx: int) -> tuple[int, ...]:
    """Root→tip joint chain for thumb / middle; other landmarks use a single id."""
    j = int(jidx)
    if j in THUMB_DEPTH_CHAIN_IDS:
        return THUMB_DEPTH_CHAIN_IDS
    if j in MIDDLE_DEPTH_CHAIN_IDS:
        return MIDDLE_DEPTH_CHAIN_IDS
    return (j,)


def _first_joint_mm_in_chain(
    joint_mm,
    chain: tuple[int, ...],
) -> tuple[int | None, np.ndarray | None]:
    """Return the first landmark in ``chain`` (MCP first) with valid depth mm."""
    for jidx in chain:
        p = joint_mm(int(jidx))
        if p is not None:
            return int(jidx), np.asarray(p, dtype=np.float64).reshape(3)
    return None, None


def _fill_depth_chain(out: np.ndarray, joint_mm, chain: tuple[int, ...]) -> None:
    for jidx in chain:
        p = joint_mm(int(jidx))
        if p is not None:
            out[int(jidx)] = np.asarray(p, dtype=np.float64).reshape(3)


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
    """Wrist + palm landmarks in **depth-camera mm** (real translation in space).

    Thumb and middle finger use root→tip depth chains (MCP, then PIP/IP, …, tip): the
    pose gate passes when each chain has at least one good depth sample, not only MCPs.
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
    if wrist is None:
        return None
    chain_a = _depth_chain_for_landmark(ia)
    chain_b = _depth_chain_for_landmark(ib)
    _, p_ia = _first_joint_mm_in_chain(joint_mm, chain_a)
    _, p_ib = _first_joint_mm_in_chain(joint_mm, chain_b)
    if p_ia is None or p_ib is None:
        return None
    out = np.full((21, 3), np.nan, dtype=np.float64)
    out[WRIST_ID] = wrist
    _fill_depth_chain(out, joint_mm, chain_a)
    _fill_depth_chain(out, joint_mm, chain_b)
    for j in (*MCP_IDS, THUMB_MCP_ID):
        if j in chain_a or j in chain_b:
            continue
        pj = joint_mm(j)
        if pj is not None:
            out[j] = pj
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
    roots = [xy(int(j)) for j in _PALM_ROOT_IDS]
    pc2 = np.mean(np.stack(roots, axis=0), axis=0)
    v1 = xy(ia) - pc2
    v2 = xy(ib) - pc2
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
    return np.stack([e1, e2, e3], axis=1)


def orthonormal_basis_from_landmark_pair(
    h: np.ndarray,
    ia: int,
    ib: int,
    *,
    min_edge_mm: float = 1e-9,
    palm_center_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (palm_center, B) with axes from palm center toward two landmarks."""
    pc = _resolve_palm_origin_mm(h, palm_center_override=palm_center_override)
    if pc is None:
        return None
    v1 = np.asarray(h[int(ia), :3], dtype=np.float64).reshape(3) - pc
    v2 = np.asarray(h[int(ib), :3], dtype=np.float64).reshape(3) - pc
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
    return pc, B


def _origin_to_joint_vector_mm(
    h: np.ndarray,
    origin: np.ndarray,
    chain_ids: tuple[int, ...],
) -> np.ndarray | None:
    """Origin → first available joint in ``chain_ids`` (root before tip)."""
    for jidx in chain_ids:
        v = _landmark_mm_if_finite(h, int(jidx), origin)
        if v is not None and float(np.linalg.norm(v)) >= 1e-6:
            return v
    return None


# Finger-chain axial score on ±(wrist→MCP) must exceed this (mm) to flip sign.
_MIDDLE_Y_AUX_FLIP_MM = 10.0
# Below this |axial| (mm), keep ref / previous-frame +Y half-space (hysteresis).
_MIDDLE_Y_AUX_DEAD_MM = 5.0


def _middle_y_finger_axial_detail(
    h: np.ndarray, ey_u: np.ndarray
) -> tuple[float | None, dict[str, float]]:
    """MCP→tip chain axial projections onto ``ey_u`` (mm), per segment + total."""
    ey_u = np.asarray(ey_u, dtype=np.float64).reshape(3)
    segs: dict[str, float] = {}
    total = 0.0
    prev = _landmark_mm_if_valid(h, MIDDLE_MCP_ID)
    names = ("pip", "dip", "tip")
    ids = (MIDDLE_PIP_ID, MIDDLE_DIP_ID, MIDDLE_TIP_ID)
    for name, jidx in zip(names, ids):
        p = _landmark_mm_if_valid(h, int(jidx))
        if prev is None or p is None:
            if p is not None:
                prev = p
            continue
        seg = np.asarray(p, dtype=np.float64).reshape(3) - np.asarray(prev, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(seg)) >= 1e-6:
            ax = float(np.dot(seg, ey_u))
            segs[str(name)] = ax
            total += ax
        prev = p
    if not segs:
        return None, segs
    return float(total), segs


def _middle_y_finger_axial_mm(h: np.ndarray, ey_u: np.ndarray) -> float | None:
    total, _ = _middle_y_finger_axial_detail(h, ey_u)
    return total


def _fill_middle_y_depth_z(dbg, h: np.ndarray) -> None:
    from debug.middle_y_sign_debug import MiddleYSignDebug

    if not isinstance(dbg, MiddleYSignDebug):
        return
    for jidx, attr in (
        (WRIST_ID, "wrist_z_mm"),
        (MIDDLE_MCP_ID, "mcp_z_mm"),
        (MIDDLE_PIP_ID, "pip_z_mm"),
        (MIDDLE_TIP_ID, "tip_z_mm"),
    ):
        p = _landmark_mm_if_valid(h, int(jidx))
        if p is not None:
            setattr(dbg, attr, float(p[2]))


def _middle_y_unit_wrist_to_mcp(h: np.ndarray) -> np.ndarray | None:
    """Unit vector wrist → middle MCP (+Y direction line in depth-camera mm)."""
    wrist = _landmark_mm_if_valid(h, WRIST_ID)
    mcp = _landmark_mm_if_valid(h, MIDDLE_MCP_ID)
    if wrist is None or mcp is None:
        return None
    v = np.asarray(mcp, dtype=np.float64).reshape(3) - np.asarray(wrist, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return v / n


def _pick_y_halfspace(
    ey_raw: np.ndarray,
    *,
    axial_mm: float | None,
    prev_y: np.ndarray | None,
    prev_raw_y: np.ndarray | None = None,
) -> tuple[np.ndarray, str, str]:
    """Return (ey, reason, anchor_used). Sign: finger aux → prev frame → raw wrist→MCP."""
    ey_u = np.asarray(ey_raw, dtype=np.float64).reshape(3)
    ney = float(np.linalg.norm(ey_u))
    if ney < 1e-9:
        return ey_u, "degenerate", "none"
    ey_u = ey_u / ney
    plus = ey_u
    minus = -ey_u

    if axial_mm is not None:
        a = float(axial_mm)
        if a >= float(_MIDDLE_Y_AUX_FLIP_MM):
            return plus, f"aux>={_MIDDLE_Y_AUX_FLIP_MM:g}mm→+", "aux"
        if a <= -float(_MIDDLE_Y_AUX_FLIP_MM):
            return minus, f"aux<=-{_MIDDLE_Y_AUX_FLIP_MM:g}mm→−", "aux"
        if abs(a) >= float(_MIDDLE_Y_AUX_DEAD_MM):
            side = "+" if a > 0.0 else "−"
            cand = plus if a > 0.0 else minus
            if prev_y is not None:
                py = np.asarray(prev_y, dtype=np.float64).reshape(3)
                pn = float(np.linalg.norm(py))
                if pn >= 1e-9 and float(np.dot(cand, py / pn)) < 0.0:
                    pass  # weak aux vs prev — fall through to prev_hyst / geom
                else:
                    return cand, f"aux_mid({a:+.1f}mm)→{side}", "aux"
            else:
                return cand, f"aux_mid({a:+.1f}mm)→{side}", "aux"

    if prev_y is not None:
        py = np.asarray(prev_y, dtype=np.float64).reshape(3)
        pn = float(np.linalg.norm(py))
        if pn >= 1e-9:
            py = py / pn
            if float(np.dot(plus, py)) >= 0.0:
                return plus, "prev_hyst→+", "prev"
            if prev_raw_y is not None:
                pr = np.asarray(prev_raw_y, dtype=np.float64).reshape(3)
                prn = float(np.linalg.norm(pr))
                if prn >= 1e-9 and float(np.dot(ey_u, pr / prn)) > 0.995:
                    return plus, "geom_stable_raw+", "none"
            return minus, "prev_hyst→−", "prev"

    return plus, "geom_wrist→mcp+", "none"


def _middle_finger_axis(
    h: np.ndarray,
    *,
    ref_y: np.ndarray | None = None,
    prev_y: np.ndarray | None = None,
    prev_raw_y: np.ndarray | None = None,
    y_sign_debug: "MiddleYSignDebug | None" = None,
) -> np.ndarray | None:
    """+Y line: wrist → middle MCP; sign from finger aux, else prev, else raw geometry."""
    from debug.middle_y_sign_debug import MiddleYSignDebug

    debug_ref_y = ref_y
    ey_raw = _middle_y_unit_wrist_to_mcp(h)
    if ey_raw is None:
        return None
    axial, seg_ax = _middle_y_finger_axial_detail(h, ey_raw)
    ey_out, reason, anchor_used = _pick_y_halfspace(
        ey_raw, axial_mm=axial, prev_y=prev_y, prev_raw_y=prev_raw_y
    )
    if y_sign_debug is not None:
        y_sign_debug.reason = reason
        y_sign_debug.axial_mm = axial
        y_sign_debug.seg_axial_mm = dict(seg_ax)
        y_sign_debug.ey_raw = np.asarray(ey_raw, dtype=np.float64).reshape(3).copy()
        y_sign_debug.ey_out = np.asarray(ey_out, dtype=np.float64).reshape(3).copy()
        y_sign_debug.anchor_used = anchor_used
        _fill_middle_y_depth_z(y_sign_debug, h)
        rn = float(np.linalg.norm(ey_raw))
        if rn >= 1e-9:
            y_sign_debug.dot_out_raw = float(np.dot(ey_out, ey_raw / rn))
        if prev_y is not None:
            py = np.asarray(prev_y, dtype=np.float64).reshape(3)
            pn = float(np.linalg.norm(py))
            if pn >= 1e-9:
                y_sign_debug.dot_out_prev = float(np.dot(ey_out, py / pn))
        if prev_raw_y is not None:
            pr = np.asarray(prev_raw_y, dtype=np.float64).reshape(3)
            prn = float(np.linalg.norm(pr))
            if prn >= 1e-9:
                y_sign_debug.dot_raw_prev_raw = float(np.dot(ey_raw, pr / prn))
        if debug_ref_y is not None:
            ry = np.asarray(debug_ref_y, dtype=np.float64).reshape(3)
            rn = float(np.linalg.norm(ry))
            if rn >= 1e-9:
                y_sign_debug.dot_out_ref = float(np.dot(ey_out, ry / rn))
        if y_sign_debug.dot_out_raw is not None and float(y_sign_debug.dot_out_raw) < 0.0:
            y_sign_debug.flipped_vs_geom = True
        if y_sign_debug.dot_out_prev is not None:
            y_sign_debug.flipped = float(y_sign_debug.dot_out_prev) < 0.0
        elif prev_raw_y is not None and y_sign_debug.dot_raw_prev_raw is not None:
            y_sign_debug.flipped = float(y_sign_debug.dot_raw_prev_raw) < 0.0
    return ey_out


def _landmark_mm_if_finite(h: np.ndarray, jidx: int, origin: np.ndarray) -> np.ndarray | None:
    p = np.asarray(h[int(jidx), :3], dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)):
        return None
    return p - np.asarray(origin, dtype=np.float64).reshape(3)


def _thumb_vector_from_origin_mm(h: np.ndarray, origin: np.ndarray) -> np.ndarray | None:
    """Palm center → thumb for +Z (MCP → IP → tip chain)."""
    return _origin_to_joint_vector_mm(h, origin, THUMB_DEPTH_CHAIN_IDS)


def _thumb_tip_vector_from_origin_mm(h: np.ndarray, origin: np.ndarray) -> np.ndarray | None:
    """Palm center → thumb tip only (for +X)."""
    v = _landmark_mm_if_finite(h, THUMB_TIP_ID, origin)
    if v is None or float(np.linalg.norm(v)) < 1e-6:
        return None
    return v


def _palm_x_unit_perp_y(thumb_vec: np.ndarray, ey_u: np.ndarray, *, min_lat_mm: float = 1e-6) -> np.ndarray | None:
    """+X: unit vector of palm→thumb projected onto the plane perpendicular to ``ey_u`` (+Y)."""
    lat = _project_onto_plane(np.asarray(thumb_vec, dtype=np.float64).reshape(3), ey_u)
    nlat = float(np.linalg.norm(lat))
    if nlat < float(min_lat_mm):
        return None
    return lat / nlat


def _project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Remove component along ``plane_normal``."""
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return np.asarray(v, dtype=np.float64).reshape(3).copy()
    n = n / nn
    return np.asarray(v, dtype=np.float64).reshape(3) - float(np.dot(v, n)) * n


def _palm_z_unit_from_y_and_thumb(ey_u: np.ndarray, thumb_vec: np.ndarray, *, min_cross_mm: float = 1e-6) -> np.ndarray | None:
    """+Z: unit ``Y × (palm→thumb)`` — normal to the plane spanned by Y and thumb."""
    z = np.cross(np.asarray(ey_u, dtype=np.float64).reshape(3), np.asarray(thumb_vec, dtype=np.float64).reshape(3))
    nz = float(np.linalg.norm(z))
    if nz < float(min_cross_mm):
        return None
    return z / nz


def _build_palm_basis_middle_y_thumb_x(
    ey: np.ndarray,
    h: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray | None:
    """Orthonormal palm basis (camera mm); ``ey`` is fixed +Y (wrist → middle MCP).

    **+Z** = ``Y × (palm→thumb chain)``. **+X** = palm→thumb **tip** in plane ⊥Y.
    Right-handed: ``Z ≈ X × Y``.
    """
    ey_u = np.asarray(ey, dtype=np.float64).reshape(3)
    ney = float(np.linalg.norm(ey_u))
    if ney < 1e-9:
        return None
    ey_u = ey_u / ney
    thumb_vec = _thumb_vector_from_origin_mm(h, origin)
    thumb_tip_vec = _thumb_tip_vector_from_origin_mm(h, origin)
    if thumb_vec is None or thumb_tip_vec is None:
        return None
    ez = _palm_z_unit_from_y_and_thumb(ey_u, thumb_vec)
    if ez is None:
        return None
    ex = _palm_x_unit_perp_y(thumb_tip_vec, ey_u)
    if ex is None:
        return None
    if float(np.dot(np.cross(ex, ey_u), ez)) < 0.0:
        ex = -ex
    return np.stack([ex, ey_u, ez], axis=1)


def align_palm_basis_to_reference(
    B: np.ndarray,
    B_ref: np.ndarray,
    h: np.ndarray,
    origin: np.ndarray,
    *,
    prev_y: np.ndarray | None = None,
    prev_raw_y: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild +X/+Z from thumb; +Y from wrist→MCP (same sign rules as current frame)."""
    del B_ref
    ey = _middle_finger_axis(h, prev_y=prev_y, prev_raw_y=prev_raw_y)
    if ey is None:
        ey = np.asarray(B[:, 1], dtype=np.float64).reshape(3).copy()
    rebuilt = _build_palm_basis_middle_y_thumb_x(ey, h, origin)
    if rebuilt is not None:
        return rebuilt
    return np.asarray(B, dtype=np.float64).reshape(3, 3).copy()


def palm_orthonormal_basis_middle_y_thumb_x(
    h: np.ndarray,
    *,
    ref_basis: np.ndarray | None = None,
    palm_center_override: np.ndarray | None = None,
    prev_y: np.ndarray | None = None,
    prev_raw_y: np.ndarray | None = None,
    y_sign_debug: "MiddleYSignDebug | None" = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Palm frame: +Y wrist→middle MCP; sign from finger aux / prev / geometry; X/Z from origin."""
    pc_geom = _palm_basis_origin_mm(h, palm_center_override=palm_center_override)
    if pc_geom is None:
        return None
    pc_out = _resolve_palm_origin_mm(h, palm_center_override=palm_center_override)
    if pc_out is None:
        pc_out = pc_geom
    ref_y_dbg = (
        np.asarray(ref_basis[:, 1], dtype=np.float64).reshape(3)
        if ref_basis is not None
        else None
    )
    ey = _middle_finger_axis(
        h,
        ref_y=ref_y_dbg,
        prev_y=prev_y,
        prev_raw_y=prev_raw_y,
        y_sign_debug=y_sign_debug,
    )
    if ey is None:
        return None
    B = _build_palm_basis_middle_y_thumb_x(ey, h, pc_geom)
    if B is None:
        return None
    if ref_basis is not None:
        B = align_palm_basis_to_reference(
            B, ref_basis, h, pc_geom, prev_y=prev_y, prev_raw_y=prev_raw_y
        )
    return np.asarray(pc_out, dtype=np.float64).reshape(3), B


def palm_orthonormal_basis(
    h: np.ndarray,
    *,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
    ref_basis: np.ndarray | None = None,
    palm_center_override: np.ndarray | None = None,
    prev_y: np.ndarray | None = None,
    prev_raw_y: np.ndarray | None = None,
    y_sign_debug: "MiddleYSignDebug | None" = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (palm_center_mm, B) with B columns = palm X (thumb), Y (fingertip), Z."""
    key = str(palm_basis).strip().lower()
    if key in ("middle_thumb", "middle_y", "middle"):
        return palm_orthonormal_basis_middle_y_thumb_x(
            h,
            ref_basis=ref_basis,
            palm_center_override=palm_center_override,
            prev_y=prev_y,
            prev_raw_y=prev_raw_y,
            y_sign_debug=y_sign_debug,
        )
    ia, ib = palm_basis_pair_indices(palm_basis)
    out = orthonormal_basis_from_landmark_pair(
        h, ia, ib, palm_center_override=palm_center_override
    )
    if out is None:
        return None
    pc, B = out
    if ref_basis is not None:
        B = align_palm_basis_to_reference(
            B, ref_basis, h, pc, prev_y=prev_y, prev_raw_y=prev_raw_y
        )
    return np.asarray(pc, dtype=np.float64).reshape(3), B
