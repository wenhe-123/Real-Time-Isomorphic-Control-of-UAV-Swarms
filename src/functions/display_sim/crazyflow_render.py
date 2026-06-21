"""Crazyflow state-control sim step + render (crazyflow 0.1.0 pattern)."""

from __future__ import annotations
import time

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from numpy.typing import NDArray


def _substeps(sim: Sim, *, outer_fps: int, max_substeps: int) -> int:
    """Physics steps per outer (camera) frame: ≈ sim.freq / outer_fps."""
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
    """Send position/velocity setpoints via ``state_control``, then ``sim.step``."""
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
    """Render current state (call after :func:`step_sim_to_cmd`)."""
    return sim.render()


def advance_sim_frame(
    sim: Sim,
    targets: Any,
    *,
    outer_fps: int,
    max_substeps: int,
    render: bool = True,
) -> NDArray | None:
    """One outer frame: ``state_control`` + ``step``; optionally ``render``."""
    step_sim_to_cmd(sim, targets, outer_fps=outer_fps, max_substeps=max_substeps)
    return render_sim(sim) if render else None


render_targets = advance_sim_frame
