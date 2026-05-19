"""Palm basis: +Y middle up, +X thumb lateral, +Z = X×Y (right-hand rule)."""

import numpy as np

from shared.left_hand_swarm_pose import (
    CAM_OPTICAL_TOWARD_CAMERA,
    IDX_MIDDLE_MCP,
    IDX_MIDDLE_TIP,
    IDX_THUMB_MCP,
    IDX_THUMB_TIP,
    IDX_WRIST,
    align_palm_basis_to_reference,
    palm_orthonormal_basis_middle_y_thumb_x,
)


def test_middle_y_thumb_x_right_hand_basis():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[IDX_WRIST] = wrist
    h[IDX_MIDDLE_MCP] = wrist + np.array([5.0, -55.0, 12.0])
    h[IDX_MIDDLE_TIP] = wrist + np.array([6.0, -95.0, 18.0])
    h[IDX_THUMB_MCP] = wrist + np.array([-42.0, -38.0, 8.0])
    h[IDX_THUMB_TIP] = wrist + np.array([-58.0, -32.0, 10.0])
    out = palm_orthonormal_basis_middle_y_thumb_x(h)
    assert out is not None
    _, B = out
    mid = h[IDX_MIDDLE_TIP] - wrist
    mid = mid / max(float(np.linalg.norm(mid)), 1e-9)
    thumb = h[IDX_THUMB_TIP] - wrist
    assert abs(float(np.dot(B[:, 1], mid))) > 0.85
    assert float(np.dot(B[:, 0], thumb)) > 0.0
    assert float(np.dot(B[:, 2], np.cross(B[:, 0], B[:, 1]))) > 0.99
    assert abs(float(np.linalg.det(B)) - 1.0) < 0.02
    assert float(np.dot(B[:, 2], CAM_OPTICAL_TOWARD_CAMERA)) > 0.0


def test_align_keeps_thumb_on_positive_x_after_flip():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[IDX_WRIST] = wrist
    h[IDX_MIDDLE_MCP] = wrist + np.array([0.0, -60.0, 5.0])
    h[IDX_MIDDLE_TIP] = wrist + np.array([0.0, -100.0, 8.0])
    h[IDX_THUMB_MCP] = wrist + np.array([-45.0, -40.0, 3.0])
    h[IDX_THUMB_TIP] = wrist + np.array([-60.0, -35.0, 4.0])
    _, B0 = palm_orthonormal_basis_middle_y_thumb_x(h)
    B1 = B0.copy()
    B1[:, 0] *= -1.0
    B1[:, 2] *= -1.0
    B2 = align_palm_basis_to_reference(B1, B0, h, wrist)
    thumb = h[IDX_THUMB_MCP] - wrist
    assert float(np.dot(B2[:, 0], thumb)) > 0.0
