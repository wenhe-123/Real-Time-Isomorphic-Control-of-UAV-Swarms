"""translation_plane_dominates_depth gating for palm rotation suppression."""

import numpy as np
import pytest

from shared.left_hand_swarm_pose import translation_plane_dominates_depth


@pytest.mark.parametrize(
    ("delta", "ratio", "min_mm", "expected"),
    [
        (np.array([5.0, 0.0, 0.0]), 1.2, 2.0, True),
        (np.array([3.0, 4.0, 0.0]), 1.0, 2.0, True),
        (np.array([1.0, 1.0, 10.0]), 1.2, 0.5, False),
        (np.array([1.0, 0.0, 0.0]), 1.2, 2.0, False),
    ],
)
def test_planar_dom(
    delta: np.ndarray,
    ratio: float,
    min_mm: float,
    expected: bool,
) -> None:
    assert (
        translation_plane_dominates_depth(
            delta,
            planar_ratio=ratio,
            planar_min_mm=min_mm,
        )
        is expected
    )


def test_ratio_zero_disabled() -> None:
    assert not translation_plane_dominates_depth(
        np.array([10.0, 10.0, 0.0]),
        planar_ratio=0.0,
        planar_min_mm=0.0,
    )
