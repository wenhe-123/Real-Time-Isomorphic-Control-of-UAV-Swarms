"""Pre-SPACE hover climb (numpy only, no sim.step)."""

from __future__ import annotations

import numpy as np

from functions.swarm_motion.formation_spacing import lift_morph_to_hover_z
from functions.swarm_motion.spacing_guard import enforce_min_separation


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
