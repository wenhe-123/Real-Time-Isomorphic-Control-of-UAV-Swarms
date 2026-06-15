"""Lateral-dominant translation gating (camera mm)."""

import numpy as np

from backup.tests.left_hand_swarm_pose_legacy_gates import (
    lateral_dom_keep_only_dx,
    strip_dz_when_abs_dx_dominates_dz,
    strip_dz_when_xy_motion_dominates,
)


def test_lateral_dom_off():
    d = np.array([3.0, 2.0, 4.0])
    out = lateral_dom_keep_only_dx(d, lateral_dom_ratio=0.0, lateral_dom_min_mm=0.0)
    np.testing.assert_allclose(out, d)


def test_lateral_dom_keeps_only_dx():
    d = np.array([12.0, 1.5, 2.0])
    out = lateral_dom_keep_only_dx(d, lateral_dom_ratio=2.0, lateral_dom_min_mm=3.0)
    np.testing.assert_allclose(out, [12.0, 0.0, 0.0])


def test_lateral_dom_small_dx_unchanged():
    d = np.array([1.0, 20.0, 5.0])
    out = lateral_dom_keep_only_dx(d, lateral_dom_ratio=2.0, lateral_dom_min_mm=5.0)
    np.testing.assert_allclose(out, d)


def test_strip_dz_off():
    d = np.array([10.0, 0.0, 8.0])
    out = strip_dz_when_xy_motion_dominates(d, ratio=0.0, min_hypot_xy_mm=0.0)
    np.testing.assert_allclose(out, d)


def test_strip_dz_when_lateral_pan():
    d = np.array([15.0, 0.0, 6.0])
    out = strip_dz_when_xy_motion_dominates(d, ratio=1.1, min_hypot_xy_mm=2.0)
    np.testing.assert_allclose(out, [15.0, 0.0, 0.0])


def test_strip_dz_keeps_intentional_in_out():
    d = np.array([2.0, 1.0, 25.0])
    out = strip_dz_when_xy_motion_dominates(d, ratio=1.1, min_hypot_xy_mm=2.0)
    np.testing.assert_allclose(out, d)


def test_dx_dom_strip_slow_lateral():
    d = np.array([0.8, 0.0, 0.6])
    out = strip_dz_when_abs_dx_dominates_dz(d, ratio=1.02, min_abs_dx_mm=0.28)
    np.testing.assert_allclose(out, [0.8, 0.0, 0.0])


def test_dx_dom_keeps_strong_in_out():
    d = np.array([1.0, 0.0, 25.0])
    out = strip_dz_when_abs_dx_dominates_dz(d, ratio=1.02, min_abs_dx_mm=0.28)
    np.testing.assert_allclose(out, d)
