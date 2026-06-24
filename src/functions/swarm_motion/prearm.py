"""Pre-SPACE hover climb and ground spawn layouts (numpy only, no sim.step)."""

from __future__ import annotations

import numpy as np

# Seconds to blend vertical-column XY → full hover morph when entering formation.
PREARM_FORMATION_RAMP_S = 3.0


def prearm_formation_setpoint(
    vertical_layout: np.ndarray,
    hover_layout: np.ndarray,
    *,
    elapsed_s: float,
    formation_start_s: float,
    ramp_s: float = PREARM_FORMATION_RAMP_S,
) -> np.ndarray:
    """Ramp setpoint from vertical column to hover morph (avoids instant 3 m MPC jump)."""
    v = np.asarray(vertical_layout, dtype=np.float32)
    h = np.asarray(hover_layout, dtype=np.float32)
    if formation_start_s < 0.0:
        alpha = 1.0
    else:
        alpha = min(
            1.0,
            max(0.0, (float(elapsed_s) - float(formation_start_s)) / max(float(ramp_s), 1e-6)),
        )
    return ((1.0 - alpha) * v + alpha * h).astype(np.float32)


def plane_ground_layout(
    plane_layout: np.ndarray,
    *,
    z_ground: float,
) -> np.ndarray:
    """Plane formation XY at ground altitude (sim home / prearm spawn)."""
    layout = np.asarray(plane_layout, dtype=np.float32).copy()
    layout[:, 2] = float(z_ground)
    return layout


def vertical_takeoff_layout(
    ground_layout: np.ndarray,
    *,
    takeoff_z: float,
    min_separation_m: float,
) -> np.ndarray:
    """Keep ground XY; lift every drone vertically to ``takeoff_z``."""
    del min_separation_m
    layout = np.asarray(ground_layout, dtype=np.float32).copy()
    layout[:, 2] = float(takeoff_z)
    return layout

