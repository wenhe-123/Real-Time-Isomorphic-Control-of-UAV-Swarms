"""Target spacing diagnostics."""

from __future__ import annotations

import numpy as np


def closest_pair(points: np.ndarray) -> tuple[float, int, int]:
    """Find the closest pair of drones in a target formation.

    Args:
        points: Drone positions, shape ``(n, 3+)`` in meters.

    Returns:
        ``(distance_m, index_i, index_j)`` for the minimum pairwise separation.
        Returns ``(inf, -1, -1)`` when fewer than two points are present.
    """
    p = np.asarray(points, dtype=np.float64)
    n = int(p.shape[0])
    if n < 2:
        return float("inf"), -1, -1
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        return float("inf"), -1, -1
    flat = d2[iu, ju]
    k = int(np.argmin(flat))
    return float(np.sqrt(float(flat[k]))), int(iu[k]), int(ju[k])
