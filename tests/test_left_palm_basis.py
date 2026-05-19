"""Palm orthonormal frame presets (left swarm rotation reference)."""

import numpy as np

from shared.left_hand_swarm_pose import (
    DEFAULT_LEFT_PALM_BASIS,
    IDX_INDEX_MCP,
    IDX_MIDDLE_MCP,
    IDX_RING_MCP,
    IDX_WRIST,
    palm_orthonormal_basis,
)


def _synthetic_hand_mm() -> np.ndarray:
    """Open-ish left hand in camera mm: wrist origin, fingers spread in +X / ±Y."""
    h = np.zeros((21, 3), dtype=np.float64)
    h[IDX_WRIST] = [0.0, 0.0, 0.0]
    h[IDX_INDEX_MCP] = [55.0, -18.0, 8.0]
    h[IDX_MIDDLE_MCP] = [58.0, 0.0, 10.0]
    h[IDX_RING_MCP] = [52.0, 22.0, 7.0]
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
