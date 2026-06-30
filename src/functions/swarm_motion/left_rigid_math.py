"""SO(3) helpers and rigid-target stepping for left-hand swarm pose."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from functions.swarm_motion.left_swarm_pose_state import LeftSwarmPoseState

def _apply_axis_sign_world_rotation(
    R_world: np.ndarray,
    axis_sign: np.ndarray | None,
) -> np.ndarray:
    """Match translation ``sign * (M @ v)``: conjugate world rotation by diag(axis_sign)."""
    if axis_sign is None:
        return np.asarray(R_world, dtype=np.float64).reshape(3, 3)
    s = np.asarray(axis_sign, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(s)) or float(np.max(np.abs(s))) < 1e-9:
        return np.asarray(R_world, dtype=np.float64).reshape(3, 3)
    S = np.diag(s)
    return S @ np.asarray(R_world, dtype=np.float64).reshape(3, 3) @ S


def palm_world_rotvec_from_basis_delta(
    Mc_rot: np.ndarray | None,
    B_current: np.ndarray,
    B_arm: np.ndarray,
    *,
    axis_sign: np.ndarray | None = None,
) -> np.ndarray:
    """Palm ΔR expressed in simulation/world coordinates."""
    B_cur = np.asarray(B_current, dtype=np.float64).reshape(3, 3)
    B0 = np.asarray(B_arm, dtype=np.float64).reshape(3, 3)
    R_delta_cam = B_cur @ B0.T
    if Mc_rot is not None:
        M = np.asarray(Mc_rot, dtype=np.float64).reshape(3, 3)
        R_world = M @ R_delta_cam @ M.T
    else:
        R_world = R_delta_cam
    R_world = _apply_axis_sign_world_rotation(R_world, axis_sign)
    rv = np.asarray(R_to_rotvec(R_world), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv)) > np.deg2rad(180.0):
        return np.zeros(3, dtype=np.float64)
    return rv


def palm_world_rotvec_from_local_delta(
    Mc_rot: np.ndarray | None,
    rv_local: np.ndarray,
    B_arm: np.ndarray,
    *,
    axis_sign: np.ndarray | None = None,
) -> np.ndarray:
    """Palm-local Δ rotvec expressed through camera→simulation rotation."""
    del B_arm
    rv = np.asarray(rv_local, dtype=np.float64).reshape(3).copy()
    R_local = rotvec_to_R(rv)
    if Mc_rot is not None:
        M = np.asarray(Mc_rot, dtype=np.float64).reshape(3, 3)
        R_world = M @ R_local @ M.T
    else:
        R_world = R_local
    R_world = _apply_axis_sign_world_rotation(R_world, axis_sign)
    return np.asarray(R_to_rotvec(R_world), dtype=np.float64).reshape(3)


def palm_cam_rotvec_from_basis_delta(B_current: np.ndarray, B_arm: np.ndarray) -> np.ndarray:
    """Intrinsic palm rotation in camera frame (for classify; less false rot on 3D translation)."""
    R = np.asarray(B_current, dtype=np.float64).reshape(3, 3) @ np.asarray(B_arm, dtype=np.float64).reshape(
        3, 3
    ).T
    rv = np.asarray(R_to_rotvec(R), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv)) > np.deg2rad(180.0):
        return np.zeros(3, dtype=np.float64)
    return rv


def palm_local_rotvec_from_basis_delta(B_current: np.ndarray, B_arm: np.ndarray) -> np.ndarray:
    """Palm-frame relative rotvec; local z is palm-normal twist."""
    R_local = np.asarray(B_arm, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        B_current, dtype=np.float64
    ).reshape(3, 3)
    rv = np.asarray(R_to_rotvec(R_local), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv)) > np.deg2rad(180.0):
        return np.zeros(3, dtype=np.float64)
    return rv


def sanitize_palm_rotvec_apply(
    rv_world: np.ndarray,
    *,
    prev_basis: np.ndarray | None,
    B_current: np.ndarray,
    Mc_rot: np.ndarray | None = None,
    delta_cam_mm: np.ndarray | None = None,
    max_step_rad: float | None = None,
) -> np.ndarray:
    """Drop rotation only when a large translation jump also causes a basis spike."""
    rv_w = np.asarray(rv_world, dtype=np.float64).reshape(3)
    step_cap = float(max_step_rad if max_step_rad is not None else np.deg2rad(32.0))
    pan_step_mm = 0.0
    if delta_cam_mm is not None:
        pan_step_mm = float(np.linalg.norm(np.asarray(delta_cam_mm, dtype=np.float64).reshape(3)))

    if prev_basis is not None:
        rv_step = palm_cam_rotvec_from_basis_delta(B_current, prev_basis)
        if Mc_rot is not None:
            rv_step = palm_world_rotvec_from_basis_delta(
                Mc_rot, B_current, prev_basis, axis_sign=None
            )
        if pan_step_mm > 95.0 and float(np.linalg.norm(rv_step)) > step_cap:
            return np.zeros(3, dtype=np.float64)

    return rv_w


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
    if s < 1e-10:
        return np.zeros(3, dtype=np.float64)
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
    """Scale a proper rotation about identity, allowing modest gain above 1."""
    s = float(np.clip(scale * max(0.0, gain), 0.0, 2.0))
    if s <= 1e-12:
        return np.eye(3, dtype=np.float64)
    if s > 1.0 + 1e-12:
        rv = np.asarray(R_to_rotvec(R), dtype=np.float64).reshape(3)
        return rotvec_to_R(rv * s)
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

    if rv_world_override is not None:
        rv_world = np.asarray(rv_world_override, dtype=np.float64).reshape(3).copy()
    else:
        rv_world = palm_world_rotvec_from_basis_delta(
            Mc_rot, B, ref_b_rot, axis_sign=sign
        )
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
    max_step_rad: float,
    max_offset_m: float,
    max_trans_step_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    off0 = np.asarray(off_hold, dtype=np.float64).reshape(3)
    off1 = np.asarray(off_tgt, dtype=np.float64).reshape(3)
    step = off1 - off0
    cap_m = float(max(max_trans_step_m, 0.0))
    sn = float(np.linalg.norm(step))
    if cap_m > 0.0 and sn > cap_m:
        off1 = off0 + step * (cap_m / sn)
    off = off1
    on = float(np.linalg.norm(off))
    if max_offset_m > 0.0 and on > max_offset_m:
        off *= max_offset_m / max(on, 1e-9)

    R_hold = np.asarray(R_hold, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R_tgt, dtype=np.float64).reshape(3, 3)
    cap = float(max(max_step_rad, 0.0))
    if cap > 0.0:
        rv_step = R_to_rotvec(R @ R_hold.T)
        sn = float(np.linalg.norm(rv_step))
        if sn > cap:
            R = rotvec_to_R(rv_step * (cap / sn)) @ R_hold
    return off, R


def _reject_noisy_pose_frame(
    *,
    delta_cam: np.ndarray,
    mcp_valid: int,
    depth_hold: bool = False,
    depth_outlier_prev: bool = False,
) -> tuple[bool, str]:
    """True → soft-reject (partial blend). Only obvious tracking loss (too few MCPs)."""
    if int(mcp_valid) < 3:
        return True, "mcp"
    dc = np.asarray(delta_cam, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(dc))
    dn_xy = float(np.hypot(float(dc[0]), float(dc[1])))
    depth_recover = bool(depth_hold or depth_outlier_prev)
    if depth_recover:
        return False, ""
    if dn_xy > 180.0 or dn > 320.0:
        return True, "jump"
    return False, ""
