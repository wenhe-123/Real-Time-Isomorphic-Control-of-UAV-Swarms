"""Depth-dominant translation gating for left swarm (camera mm)."""

import numpy as np

from backup.tests.left_hand_swarm_pose_legacy_gates import effective_cam_delta_for_translation


def test_ratio_zero_passes_through():
    d = np.array([3.0, 4.0, 30.0])
    out = effective_cam_delta_for_translation(d, depth_dom_ratio=0.0, depth_dom_min_mm=0.0)
    np.testing.assert_allclose(out, d)


def test_below_min_mm_no_gate():
    d = np.array([50.0, 0.0, 2.0])
    out = effective_cam_delta_for_translation(d, depth_dom_ratio=2.0, depth_dom_min_mm=5.0)
    np.testing.assert_allclose(out, d)


def test_dominant_depth_zeros_xy():
    d = np.array([5.0, 0.0, 10.0])
    out = effective_cam_delta_for_translation(d, depth_dom_ratio=1.65, depth_dom_min_mm=3.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 10.0])


def test_non_dominant_depth_keeps_xy():
    d = np.array([10.0, 0.0, 10.0])
    out = effective_cam_delta_for_translation(d, depth_dom_ratio=1.65, depth_dom_min_mm=3.0)
    np.testing.assert_allclose(out, d)


def test_pure_depth_small_h():
    d = np.array([0.0, 0.0, 8.0])
    out = effective_cam_delta_for_translation(d, depth_dom_ratio=1.5, depth_dom_min_mm=0.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 8.0])
