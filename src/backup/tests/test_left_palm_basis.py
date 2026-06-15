"""Palm orthonormal frame presets (left swarm rotation reference)."""

import numpy as np

from functions.mode_switch.hand_constants import (
    INDEX_MCP_ID,
    MIDDLE_MCP_ID,
    MIDDLE_TIP_ID,
    RING_MCP_ID,
    THUMB_MCP_ID,
    THUMB_TIP_ID,
    WRIST_ID,
)
from functions.swarm_motion.left_hand_swarm_pose import (
    DEFAULT_LEFT_PALM_BASIS,
    axis_locked_trans_rot_blend_weights,
    palm_cam_rotvec_from_basis_delta,
    palm_orthonormal_basis,
    stabilize_palm_basis_continuity,
)


def _synthetic_hand_mm() -> np.ndarray:
    """Open-ish left hand in camera mm: wrist origin, fingers spread in +X / ±Y."""
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = [0.0, 0.0, 0.0]
    h[INDEX_MCP_ID] = [55.0, -18.0, 8.0]
    h[MIDDLE_MCP_ID] = [58.0, 0.0, 10.0]
    h[RING_MCP_ID] = [52.0, 22.0, 7.0]
    return h


def test_default_basis_right_handed_orthonormal():
    h = _synthetic_hand_mm()
    out = palm_orthonormal_basis(h, palm_basis=DEFAULT_LEFT_PALM_BASIS)
    assert out is not None
    _, B = out
    assert np.allclose(B.T @ B, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(B)) > 0.0


def test_each_preset_returns_valid_frame():
    h = _synthetic_hand_mm()
    for name in ("index_middle", "index_ring", "middle_ring"):
        out = palm_orthonormal_basis(h, palm_basis=name)
        assert out is not None, name
        _, B = out
        assert abs(float(np.linalg.det(B)) - 1.0) < 1e-6, name


def test_stabilize_basis_removes_z_flip():
    B_ref = np.eye(3, dtype=np.float64)
    B_flip = B_ref.copy()
    B_flip[:, 0] *= -1.0
    B_flip[:, 2] *= -1.0
    B_fix = stabilize_palm_basis_continuity(B_flip, B_ref)
    assert float(np.dot(B_fix[:, 2], B_ref[:, 2])) > 0.99
    rv = palm_cam_rotvec_from_basis_delta(B_fix, B_ref)
    assert float(np.linalg.norm(rv)) < np.deg2rad(5.0)


def test_axis_locked_ignores_world_flip_when_rv_cam_zero():
    mot, w_rot, _ = axis_locked_trans_rot_blend_weights(
        np.array([0.05, 0.0, 0.0]),
        np.array([0.0, 0.0, 3.0]),
        trans_on_m=0.009,
        rot_on_rad=0.020,
        rv_cam_rad=np.zeros(3),
        delta_trans_mm=np.array([30.0, 0.0, 0.0]),
        secondary_frac=0.5,
    )
    assert mot == "translate"
    assert w_rot < 0.15
