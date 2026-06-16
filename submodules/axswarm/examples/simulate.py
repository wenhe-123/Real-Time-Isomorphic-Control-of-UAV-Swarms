"""axswarm simulation of a four-drone position switch."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import yaml
from crazyflow import Sim
from crazyflow.utils import enable_cache
from numpy.typing import NDArray
from utils import draw_line, draw_points

from axswarm import SolverData, SolverSettings, solve

enable_cache()

logger = logging.getLogger(__name__)


np.random.seed(0)
rgbas = np.random.rand(200, 4)
rgbas[..., 3] = 1.0


def render_solutions(sim: Sim, trajectories: NDArray):
    for i, trajectory in enumerate(np.asarray(trajectories)):
        draw_points(sim, trajectory, rgba=rgbas[i], size=0.005)
        draw_line(sim, trajectory, rgba=rgbas[i])


def render_setpoints(sim: Sim, setpoints: dict[str, NDArray]):
    draw_points(sim, setpoints["pos"], rgba=np.array([0.0, 0.0, 0.0, 1.0]), size=0.025)


def generate_switch_problem(spacing: float = 1.5, height: float = 0.8):
    """Create initial states and fixed target setpoints for a four-drone swap."""
    half = spacing / 2
    initial_pos = np.array(
        [
            [-half, -half, height],
            [half, -half, height],
            [half, half, height],
            [-half, half, height],
        ],
        dtype=np.float32,
    )
    target_pos = initial_pos[[2, 3, 0, 1]]
    initial_states = np.zeros((4, 6), dtype=np.float32)
    initial_states[:, :3] = initial_pos
    setpoints = {
        "pos": target_pos,
        "vel": np.zeros_like(target_pos),
        "acc": np.zeros_like(target_pos),
    }
    return initial_states, setpoints


def load_settings() -> tuple[SolverSettings, dict[str, NDArray]]:
    with open(Path(__file__).resolve().parents[1] / "params/settings.yaml") as f:
        config = yaml.safe_load(f)

    settings = config["SolverSettings"]
    for k, v in settings.items():
        if isinstance(v, list):
            settings[k] = np.asarray(v)
    settings = SolverSettings(**settings)

    dynamics = config["Dynamics"]
    dynamics = {k: np.asarray(v) for k, v in dynamics.items()}
    return settings, dynamics


def simulate_axswarm(
    sim: Sim,
    initial_states: NDArray,
    setpoints: dict[str, NDArray],
    duration_sec: float = 6.0,
    render: bool = False,
) -> NDArray:
    """Run the axswarm simulation against fixed setpoints."""
    settings, dynamics = load_settings()
    n_steps = int(duration_sec * settings.freq)
    n_drones = initial_states.shape[0]

    solver_data = SolverData.init(
        setpoints=setpoints,
        initial_states=initial_states,
        K=settings.K,
        N=settings.N,
        A=dynamics["A"],
        B=dynamics["B"],
        A_prime=dynamics["A_prime"],
        B_prime=dynamics["B_prime"],
        freq=settings.freq,
        smoothness_weight=settings.smoothness_weight,
        input_smoothness_weight=settings.input_smoothness_weight,
        input_continuity_weight=settings.input_continuity_weight,
    )

    sim.reset()
    control = np.zeros((sim.n_worlds, n_drones, 13), dtype=np.float32)
    pos = sim.data.states.pos.at[0, ...].set(initial_states[:, :3])
    vel = sim.data.states.vel.at[0, ...].set(initial_states[:, 3:])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos, vel=vel))

    trajectories = np.zeros((n_steps, n_drones, 3), dtype=np.float32)
    solve_timings = []
    for step in range(n_steps):
        pos = np.asarray(sim.data.states.pos[0])
        vel = np.asarray(sim.data.states.vel[0])
        states = np.concatenate((pos, vel), axis=-1)
        tstart = time.perf_counter()
        success, _, solver_data = solve(states, solver_data, settings)
        tstop = time.perf_counter()
        solve_timings.append(tstop - tstart)
        if not all(success):
            logger.warning("Solve failed")

        solver_data = solver_data.step(solver_data)
        control[0, :, :3] = solver_data.u_pos[:, 0]
        control[0, :, 3:6] = solver_data.u_vel[:, 0]

        sim.state_control(control)
        sim.step(sim.freq // settings.freq)
        if render:
            render_solutions(sim, solver_data.pos)
            render_setpoints(sim, setpoints)
            sim.render()

        trajectories[step] = sim.data.states.pos[0]

    print(f"mean axswarm solve time: {np.mean(solve_timings) * 1000:.2f} ms")

    return trajectories


def plot_trajectories(initial_states: NDArray, setpoints: dict[str, NDArray], pos: NDArray):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(pos.shape[1]):
        p = pos[:, i, :]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], label=f"Drone {i}", color=rgbas[i])
        ax.scatter(*initial_states[i, :3], marker="o", color=rgbas[i])
        ax.scatter(*setpoints["pos"][i], marker="x", color=rgbas[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Four-Drone Position Switch")
    ax.legend()
    plt.show()


def main(render: bool = False, duration_sec: float = 60.0):
    initial_states, setpoints = generate_switch_problem()
    sim = Sim(
        n_drones=initial_states.shape[0],
        freq=400,
        state_freq=80,
        attitude_freq=400,
        control="state",
    )
    results = simulate_axswarm(
        sim, initial_states, setpoints, duration_sec=duration_sec, render=render
    )
    sim.close()

    plot_trajectories(initial_states, setpoints, results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)
    fire.Fire(main)
