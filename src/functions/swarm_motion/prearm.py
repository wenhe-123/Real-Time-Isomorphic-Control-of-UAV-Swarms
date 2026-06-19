"""Pre-SPACE hover climb and ground spawn layouts (numpy only, no sim.step)."""

from __future__ import annotations

import numpy as np

from functions.swarm_motion.formation_spacing import lift_morph_to_hover_z
from functions.swarm_motion.spacing_guard import (
    closest_pair,
    enforce_min_separation,
    enforce_min_separation_xy,
)


def vertical_takeoff_layout(
    ground_layout: np.ndarray,
    *,
    takeoff_z: float,
    min_separation_m: float,
) -> np.ndarray:
    """Keep ground XY; lift every drone vertically to ``takeoff_z``."""
    layout = np.asarray(ground_layout, dtype=np.float32).copy()
    layout[:, 2] = float(takeoff_z)
    return enforce_min_separation(
        layout, float(min_separation_m), iters=10
    ).astype(np.float32)


def complete_prearm_takeoff(
    morph_layout: np.ndarray,
    *,
    hover_z: float,
    min_separation_m: float,
) -> np.ndarray:
    """Lift morph layout to hover_z with spacing guard."""
    layout = np.asarray(morph_layout, dtype=np.float32)
    return enforce_min_separation(
        lift_morph_to_hover_z(layout, float(hover_z)),
        float(min_separation_m),
        iters=12,
    ).astype(np.float32)


def sim_chessboard_ground_layout(
    n_drones: int,
    *,
    min_separation_m: float,
    z_ground: float,
    xy_half_extent_m: float,
) -> np.ndarray:
    """Checkerboard-staggered ground grid centered at origin, within ``xy_half_extent_m``."""
    n = int(n_drones)
    if n < 1:
        raise ValueError(f"n_drones must be >= 1, got {n}")
    sep = float(min_separation_m)
    pitch = max(sep * 1.10, 0.34)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    stagger = 0.5 * pitch
    width = (cols - 1) * pitch + (stagger if rows > 1 else 0.0)
    height = max(0.0, (rows - 1) * pitch)
    max_xy = max(0.25, float(xy_half_extent_m) - 0.12)
    span = max(width, height)
    if span > 2.0 * max_xy and span > 1e-6:
        scale = (2.0 * max_xy) / span
        pitch *= scale
        stagger *= scale
        width *= scale
        height *= scale

    z = float(z_ground)
    pts = np.zeros((n, 3), dtype=np.float32)
    x0 = -0.5 * width
    y0 = -0.5 * height
    for i in range(n):
        row = i // cols
        col = i % cols
        x = x0 + col * pitch + (stagger if (row % 2) == 1 else 0.0)
        y = y0 + row * pitch
        pts[i] = (x, y, z)
    out = enforce_min_separation_xy(pts, sep, z, iters=12)
    d, pi, pj = closest_pair(out)
    if d < sep * 0.98:
        raise ValueError(
            f"chessboard layout spacing {d:.3f}m < min_separation {sep:.3f}m "
            f"(pair {pi},{pj}); reduce n_drones or increase workspace"
        )
    return out
