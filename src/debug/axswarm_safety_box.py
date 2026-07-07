"""Draw axswarm position limits as a red wireframe box in the MuJoCo viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from crazyflow.sim.visualize import draw_line

if TYPE_CHECKING:
    from crazyflow.sim import Sim

_SAFETY_BOX_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _wireframe_loops(
    pos_min: np.ndarray | list[float],
    pos_max: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build bottom/top rectangle loops and vertical edges for a wireframe box.

    Args:
        pos_min: Lower corner ``(x, y, z)`` of the safety box in world meters.
        pos_max: Upper corner ``(x, y, z)`` of the safety box in world meters.

    Returns:
        ``(bottom, top, verticals)`` where ``bottom`` and ``top`` are closed rectangle
        polylines and ``verticals`` is a list of four corner edge segments.
    """
    lo = np.asarray(pos_min, dtype=np.float64)
    hi = np.asarray(pos_max, dtype=np.float64)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    bottom = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z0],
        ],
        dtype=np.float64,
    )
    top = np.array(
        [
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y0, z1],
        ],
        dtype=np.float64,
    )
    verticals = [
        np.array([[x0, y0, z0], [x0, y0, z1]], dtype=np.float64),
        np.array([[x1, y0, z0], [x1, y0, z1]], dtype=np.float64),
        np.array([[x1, y1, z0], [x1, y1, z1]], dtype=np.float64),
        np.array([[x0, y1, z0], [x0, y1, z1]], dtype=np.float64),
    ]
    return bottom, top, verticals


def draw_axswarm_safety_box(
    sim: Sim,
    pos_min: np.ndarray | list[float],
    pos_max: np.ndarray | list[float],
    *,
    rgba: np.ndarray | None = None,
    line_size: float = 0.12,
) -> None:
    """Draw a wireframe box from ``pos_min`` / ``pos_max`` in the MuJoCo viewer.

    Call each frame before :func:`render_sim`.

    Args:
        sim: Crazyflow simulation instance.
        pos_min: Lower corner ``(x, y, z)`` in world meters.
        pos_max: Upper corner ``(x, y, z)`` in world meters.
        rgba: Line color as ``(r, g, b, a)``; defaults to red when ``None``.
        line_size: MuJoCo line width for all box edges.

    Returns:
        None.
    """
    color = _SAFETY_BOX_RGBA if rgba is None else np.asarray(rgba, dtype=np.float64)
    bottom, top, verticals = _wireframe_loops(pos_min, pos_max)
    draw_line(sim, bottom, rgba=color, start_size=line_size, end_size=line_size)
    draw_line(sim, top, rgba=color, start_size=line_size, end_size=line_size)
    for edge in verticals:
        draw_line(sim, edge, rgba=color, start_size=line_size, end_size=line_size)


def draw_axswarm_safety_box_from_settings(
    sim: Sim,
    settings: Any,
    *,
    line_size: float = 0.12,
) -> None:
    """Draw the axswarm safety box using ``pos_min`` / ``pos_max`` from solver settings.

    Args:
        sim: Crazyflow simulation instance.
        settings: Axswarm ``SolverSettings`` (or compatible object) with ``pos_min`` and
            ``pos_max`` attributes.
        line_size: MuJoCo line width for all box edges.

    Returns:
        None.
    """
    draw_axswarm_safety_box(
        sim,
        np.asarray(settings.pos_min, dtype=np.float64),
        np.asarray(settings.pos_max, dtype=np.float64),
        line_size=line_size,
    )
