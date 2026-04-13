"""Ray from origin vs axis-aligned box [-h, h]^3 — first exit (smallest positive t)."""

from __future__ import annotations

import numpy as np


def ray_box_exit_from_origin(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    half: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ray p(t) = t * d, t > 0, box [-half, half]^3. Return (px, py, pz, t) on first boundary hit.
    """
    h = float(half)
    eps = 1e-12
    tx = np.where(dx > eps, h / dx, np.where(dx < -eps, (-h) / dx, np.inf))
    ty = np.where(dy > eps, h / dy, np.where(dy < -eps, (-h) / dy, np.inf))
    tz = np.where(dz > eps, h / dz, np.where(dz < -eps, (-h) / dz, np.inf))
    t = np.minimum(np.minimum(tx, ty), tz)
    px = t * dx
    py = t * dy
    pz = t * dz
    return px, py, pz, t
