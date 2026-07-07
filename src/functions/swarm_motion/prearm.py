"""Pre-SPACE hover climb and ground spawn layouts (numpy only, no sim.step)."""

from __future__ import annotations

import numpy as np

# Seconds to blend vertical-column XY → full hover morph when entering formation.
PREARM_FORMATION_RAMP_S = 3.0
# Seconds to hold after HL in-place descend before HL land.
PREARM_PRE_LAND_HOVER_S = 3.0
# In-place HL descent distance after formation (m, relative −Z from current pose).
PREARM_HL_DESCEND_M = 0.50


def prearm_formation_setpoint(
    vertical_layout: np.ndarray,
    hover_layout: np.ndarray,
    *,
    elapsed_s: float,
    formation_start_s: float,
    ramp_s: float = PREARM_FORMATION_RAMP_S,
) -> np.ndarray:
    """Ramp setpoint from vertical column layout to hover morph formation.

    Args:
        vertical_layout: Vertical-column drone positions, shape ``(n, 3)``.
        hover_layout: Target hover morph positions, shape ``(n, 3)``.
        elapsed_s: Current elapsed time (s).
        formation_start_s: Time when the ramp begins (s); ``< 0`` skips ramping.
        ramp_s: Ramp duration (s).

    Returns:
        Blended setpoint array, shape ``(n, 3)``, float32.
    """
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
    """Place a plane formation at ground altitude for prearm spawn.

    Args:
        plane_layout: Plane formation XY layout, shape ``(n, 3+)``.
        z_ground: Ground altitude in simulation meters.

    Returns:
        Layout copy with all Z coordinates set to ``z_ground``, float32.
    """
    layout = np.asarray(plane_layout, dtype=np.float32).copy()
    layout[:, 2] = float(z_ground)
    return layout


def vertical_takeoff_layout(
    ground_layout: np.ndarray,
    *,
    takeoff_z: float,
    min_separation_m: float,
) -> np.ndarray:
    """Lift a ground layout vertically to takeoff altitude while preserving XY.

    Args:
        ground_layout: Ground spawn positions, shape ``(n, 3+)``.
        takeoff_z: Target takeoff altitude (m).
        min_separation_m: Reserved for spacing checks (currently unused).

    Returns:
        Layout copy with all Z coordinates set to ``takeoff_z``, float32.
    """
    del min_separation_m
    layout = np.asarray(ground_layout, dtype=np.float32).copy()
    layout[:, 2] = float(takeoff_z)
    return layout

