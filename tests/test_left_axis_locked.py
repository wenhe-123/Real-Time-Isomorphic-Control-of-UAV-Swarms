"""Principal-axis snap and palm-basis axis_locked motion."""

import numpy as np

from shared.left_hand_swarm_pose import (
    IDX_MIDDLE_MCP,
    IDX_THUMB_MCP,
    IDX_WRIST,
    axis_locked_gated_cam_delta_mm,
    axis_locked_gated_palm_components,
    classify_axis_locked_motion,
    classify_pose_small_translate_vs_rotate,
    palm_orthonormal_basis_middle_y_thumb_x,
)


def test_forward_uses_raw_delta_not_stripped():
    d = axis_locked_gated_cam_delta_mm(
        np.array([3.0, 2.0, 8.0]),
        forward_min_dz_mm=3.0,
    )
    assert np.allclose(d, [0.0, 0.0, 8.0])


def test_strong_depth_beats_lateral():
    d = axis_locked_gated_cam_delta_mm(
        np.array([11.0, 3.0, 14.0]),
        depth_strong_min_dz_mm=9.0,
        lateral_min_dx_mm=7.0,
    )
    assert np.allclose(d, [0.0, 0.0, 14.0])


def test_fallback_picks_largest_cam_axis():
    d = axis_locked_gated_cam_delta_mm(np.array([2.0, 9.0, 3.0]))
    assert np.allclose(d, [0.0, 9.0, 0.0])


def test_palm_gating_keeps_forward_when_dominant():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[IDX_WRIST] = wrist
    h[IDX_MIDDLE_MCP] = wrist + np.array([0.0, -60.0, 5.0])
    h[IDX_THUMB_MCP] = wrist + np.array([-45.0, -40.0, 3.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    d = axis_locked_gated_palm_components(np.array([2.0, 3.0, 25.0]), B)
    assert abs(float(d[2])) > 5.0
    assert abs(float(d[0])) < 1e-5 and abs(float(d[1])) < 1e-5


def test_pose_small_center_translates():
    motion = classify_pose_small_translate_vs_rotate(
        np.zeros(3),
        np.array([0.0, 0.022, 0.0]),
        pose_rotate_rad=0.008,
        trans_on_m=0.009,
    )
    assert motion == "translate"


def test_pose_large_rotates():
    motion = classify_pose_small_translate_vs_rotate(
        np.array([0.0, 0.15, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        pose_rotate_rad=0.008,
        trans_on_m=0.009,
    )
    assert motion == "rotate"


def test_cam_gating_vertical_before_shallow_depth():
    d = axis_locked_gated_cam_delta_mm(np.array([3.0, -12.0, 8.0]))
    assert np.allclose(d, [0.0, -12.0, 0.0])


def test_classify_axis_locked_prefers_translate_when_rot_small():
    motion = classify_axis_locked_motion(
        np.array([0.02, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.005]),
        trans_on_m=0.009,
        rot_on_rad=0.011,
    )
    assert motion == "translate"
