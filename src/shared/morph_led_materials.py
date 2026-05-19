"""Drone LED deck colors by morph mode (cf21B_500 ``led_top`` / ``led_bot`` materials).

- **M1**: per-drone color from depth (world *z*), continuous colormap (plasma).
- **M2–M5**: top + bottom caps share one color; side band shares another (by *z* quantiles).

Uses the same ``change_material`` API as ``led_deck.py``.
"""

from __future__ import annotations

import numpy as np

try:
    from crazyflow.sim.visualize import change_material as _change_material_led
except ImportError:  # pragma: no cover
    _change_material_led = None

try:
    import matplotlib as mpl

    if hasattr(mpl, "colormaps"):
        _PLASMA = mpl.colormaps["plasma"]
    else:
        from matplotlib import cm as _mpl_cm

        _PLASMA = _mpl_cm.get_cmap("plasma")
except Exception:  # pragma: no cover - optional matplotlib
    _PLASMA = None

_CAP_BAND = 0.18
_M1_EMISSION = 0.40
_FACE_EMISSION = 0.48


def _positions_world_xyz(sim) -> np.ndarray:
    return np.asarray(sim.data.states.pos[0], dtype=np.float64)


def led_rgba_for_morph_mode(morph_mode: int, pos_xyz: np.ndarray) -> np.ndarray:
    """Return per-drone RGBA (n, 4) from world positions (n, 3)."""
    pos_xyz = np.asarray(pos_xyz, dtype=np.float64).reshape(-1, 3)
    n = int(pos_xyz.shape[0])
    if n == 0:
        return np.zeros((0, 4), dtype=np.float64)

    z = pos_xyz[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    span = max(zmax - zmin, 1e-6)

    mm = int(morph_mode)
    if mm == 1:
        t = np.clip((z - zmin) / span, 0.0, 1.0)
        if _PLASMA is not None:
            rgba = np.asarray(_PLASMA(t), dtype=np.float64)
        else:
            rgba = np.zeros((n, 4), dtype=np.float64)
            rgba[:, 0] = t
            rgba[:, 1] = 0.35 * (1.0 - np.abs(t - 0.5) * 2.0)
            rgba[:, 2] = 1.0 - t
        rgba[:, 3] = 1.0
        return rgba

    # M2–M5: bottom + top caps (low / high z) same hue; middle "side" another.
    zn = (z - zmin) / span
    cap = (zn < _CAP_BAND) | (zn > 1.0 - _CAP_BAND)
    cap_rgba = np.array([1.0, 0.78, 0.12, 1.0], dtype=np.float64)
    side_rgba = np.array([0.12, 0.42, 0.95, 1.0], dtype=np.float64)
    rgba = np.where(cap[:, None], cap_rgba[None, :], side_rgba[None, :])
    return rgba.astype(np.float64)


def apply_morph_led_theme(sim, morph_mode: int) -> None:
    """Update ``led_top`` / ``led_bot`` for all drones; no-op if materials missing."""
    if _change_material_led is None:
        return
    pos = _positions_world_xyz(sim)
    if pos.size == 0:
        return
    n = int(pos.shape[0])
    ids = np.arange(n, dtype=int)
    try:
        rgba = led_rgba_for_morph_mode(morph_mode, pos)
        em = float(_M1_EMISSION if int(morph_mode) == 1 else _FACE_EMISSION)
        _change_material_led(sim, "led_top", ids, rgba=rgba, emission=em)
        _change_material_led(sim, "led_bot", ids, rgba=rgba, emission=em)
    except ValueError:
        pass
