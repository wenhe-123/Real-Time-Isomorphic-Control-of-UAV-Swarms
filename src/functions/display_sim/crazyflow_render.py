"""Crazyflow state-control sim step + render (crazyflow 0.1.0 pattern)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from numpy.typing import NDArray


def _substeps(sim: Sim, *, outer_fps: int, max_substeps: int) -> int:
    """Compute physics substeps per outer (camera) frame.

    Args:
        sim: Crazyflow simulation instance.
        outer_fps: Outer loop rate in Hz (camera or control rate).
        max_substeps: Upper cap on substeps per outer frame.

    Returns:
        Number of ``sim.step`` calls to run, at least 1 and at most ``max_substeps``.
    """
    n = int(round(float(sim.freq) / max(float(outer_fps), 1.0)))
    return max(1, min(n, max(1, int(max_substeps))))


def step_sim_to_cmd(
    sim: Sim,
    targets: Any,
    *,
    outer_fps: int,
    max_substeps: int,
    velocities: Any | None = None,
    control_hz: float | None = None,
) -> int:
    """Send position/velocity setpoints via ``state_control``, then advance the simulation.

    Args:
        sim: Crazyflow simulation instance.
        targets: Drone position targets, shape ``(n_drones, 3)`` in world meters.
        outer_fps: Nominal outer-loop rate in Hz for substep calculation.
        max_substeps: Upper cap on physics substeps per call.
        velocities: Optional drone velocities, shape ``(n_drones, 3)``; written to control[3:6].
        control_hz: Override rate for substep calculation; defaults to ``outer_fps``.

    Returns:
        Number of physics substeps executed.

    Raises:
        ValueError: If ``targets`` or ``velocities`` shapes do not match ``sim.n_drones``.
    """
    pts = np.asarray(targets, dtype=np.float64)
    if pts.shape != (sim.n_drones, 3):
        raise ValueError(f"targets must be ({sim.n_drones}, 3), got {pts.shape}")
    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float64)
    control[..., :3] = pts
    if velocities is not None:
        vel = np.asarray(velocities, dtype=np.float64)
        if vel.shape != (sim.n_drones, 3):
            raise ValueError(f"velocities must be ({sim.n_drones}, 3), got {vel.shape}")
        control[..., 3:6] = vel
    sim.state_control(control)
    step_hz = float(control_hz) if control_hz is not None else float(outer_fps)
    n_steps = _substeps(sim, outer_fps=int(round(step_hz)), max_substeps=max_substeps)
    sim.step(n_steps)
    return n_steps


def render_sim(sim: Sim) -> NDArray | None:
    """Render the current simulation state to an image array.

    Args:
        sim: Crazyflow simulation instance.

    Returns:
        Rendered frame as a numpy array, or ``None`` when rendering is unavailable.
    """
    return sim.render()
