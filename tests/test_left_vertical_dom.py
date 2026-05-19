"""Vertical-dominant translation gating (camera mm)."""

import numpy as np

from shared.left_hand_swarm_pose import vertical_dom_zero_optical_z


def test_vertical_dom_off():
    d = np.array([3.0, 20.0, 5.0])
    out = vertical_dom_zero_optical_z(d, vertical_dom_ratio=0.0, vertical_dom_min_mm=0.0)
    np.testing.assert_allclose(out, d)


def test_vertical_dom_keeps_only_dy():
    d = np.array([2.0, 15.0, 4.0])
    out = vertical_dom_zero_optical_z(d, vertical_dom_ratio=1.2, vertical_dom_min_mm=2.0)
    np.testing.assert_allclose(out, [0.0, 15.0, 0.0])


def test_vertical_dom_small_dy_unchanged():
    d = np.array([2.0, 1.0, 50.0])
    out = vertical_dom_zero_optical_z(d, vertical_dom_ratio=1.2, vertical_dom_min_mm=5.0)
    np.testing.assert_allclose(out, d)


def test_vertical_dom_optical_preserve_keeps_dz():
    # Without preserve, |dy| dominates hypot(dx,dz) and would zero dz. Preserve runs first when
    # |dz| is large vs hypot(dx,dy) (in-out with perspective dy).
    d = np.array([5.0, 40.0, 28.0])
    out = vertical_dom_zero_optical_z(
        d,
        vertical_dom_ratio=1.2,
        vertical_dom_min_mm=2.0,
        optical_preserve_ratio=0.65,
        optical_preserve_min_mm=1.0,
    )
    np.testing.assert_allclose(out, d)
