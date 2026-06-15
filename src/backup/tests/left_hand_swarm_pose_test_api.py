"""Test-only helpers for left-hand swarm pose (not used by online_control)."""

from __future__ import annotations

import numpy as np

from backup.tests.left_hand_swarm_pose_legacy_gates import palm_translation_components_mm
from functions.swarm_motion.left_hand_swarm_pose import (
    PALM_AXIS_TO_WORLD_PERM,
    axis_locked_trans_metric_world_m,
    axis_locked_trans_rot_blend_weights,
    palm_world_rotvec_from_basis_delta,
)


def palm_components_in_camera_mm(delta_cam: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Project camera-mm wrist delta onto palm columns (x=thumb, y=fingertip, z=X×Y)."""
    Bc = np.asarray(B, dtype=np.float64).reshape(3, 3)
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3)
    return np.array([float(np.dot(d, Bc[:, i])) for i in range(3)], dtype=np.float64)


def palm_vector_palm_to_world(v: np.ndarray) -> np.ndarray:
    """Map palm-frame vector (x,y,z) to world/sim (X,Y,Z) via PALM_AXIS_TO_WORLD_PERM."""
    p = np.asarray(v, dtype=np.float64).reshape(3)
    wx, wy, wz = PALM_AXIS_TO_WORLD_PERM
    return np.array([float(p[wx]), float(p[wy]), float(p[wz])], dtype=np.float64)


def palm_pose_change_rad(
    B_current: np.ndarray,
    B_arm: np.ndarray,
    *,
    Mc_rot: np.ndarray | None = None,
) -> float:
    rv = palm_world_rotvec_from_basis_delta(Mc_rot, B_current, B_arm)
    return float(np.linalg.norm(rv))


def palm_center_translation_components(
    delta_cam: np.ndarray,
    B: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return palm_translation_components_mm(delta_cam, B, **kwargs)


def classify_pose_small_translate_vs_rotate(
    rv_pose_world_rad: np.ndarray,
    delta_world_m: np.ndarray,
    *,
    pose_rotate_rad: float,
    trans_on_m: float,
    rv_cam_rad: np.ndarray | None = None,
    rot_excl_ratio: float = 1.25,
    trans_excl_ratio: float = 1.15,
    secondary_frac: float = 0.30,
) -> str:
    del rot_excl_ratio, trans_excl_ratio
    mot, _, _ = axis_locked_trans_rot_blend_weights(
        delta_world_m,
        rv_pose_world_rad,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(pose_rotate_rad),
        rv_cam_rad=rv_cam_rad,
        secondary_frac=float(secondary_frac),
    )
    return mot


def classify_axis_locked_motion(
    trans_world_m: np.ndarray,
    rot_world_rad: np.ndarray,
    *,
    trans_on_m: float = 0.009,
    rot_on_rad: float = 0.011,
    rot_excl_ratio: float = 1.32,
    trans_excl_ratio: float = 1.85,
    rv_intrinsic_for_trans: np.ndarray | None = None,
    trans_wins_rot_below_rad: float = 0.95,
) -> str:
    rv_i = (
        np.asarray(rot_world_rad, dtype=np.float64).reshape(3)
        if rv_intrinsic_for_trans is None
        else np.asarray(rv_intrinsic_for_trans, dtype=np.float64).reshape(3)
    )
    tw = axis_locked_trans_metric_world_m(
        trans_world_m,
        rv_i,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(rot_on_rad),
    )
    rw = float(np.linalg.norm(np.asarray(rot_world_rad, dtype=np.float64).reshape(3)))
    t_on = float(max(trans_on_m, 1e-6))
    r_on = float(max(rot_on_rad, 1e-6))
    re = float(max(1.05, rot_excl_ratio))
    te = float(max(1.05, trans_excl_ratio))
    ts = tw / t_on
    rs = rw / r_on
    if rw < float(trans_wins_rot_below_rad) * r_on and ts >= 1.0:
        return "translate"
    if rs >= 1.0 and rs >= re * ts:
        return "rotate"
    if ts >= 1.0 and ts >= te * rs:
        return "translate"
    return "none"
