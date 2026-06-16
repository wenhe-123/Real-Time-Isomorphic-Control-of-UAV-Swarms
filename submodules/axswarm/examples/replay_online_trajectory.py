"""Offline axswarm replay against a trajectory recorded from iso_swarm online_control.

Record (from iso_swarm/src)::

    pixi run online-dual -- --record-trajectory ../../data/online_run.npz

Replay (from this directory)::

    cd examples
    pixi run --manifest-path .. python replay_online_trajectory.py \\
        --trajectory ../../data/online_run.npz --plot
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import yaml

from axswarm import SolverData, SolverSettings, solve

logger = logging.getLogger(__name__)


def load_settings_yaml(settings_path: Path | None = None) -> tuple[SolverSettings, dict[str, np.ndarray]]:
    if settings_path is None:
        settings_path = Path(__file__).resolve().parents[1] / "params" / "settings.yaml"
    with open(settings_path) as f:
        config = yaml.safe_load(f)
    settings = config["SolverSettings"]
    for k, v in settings.items():
        if isinstance(v, list):
            settings[k] = np.asarray(v)
    dynamics = {k: np.asarray(v) for k, v in config["Dynamics"].items()}
    return SolverSettings(**settings), dynamics


def load_online_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        traj = {k: np.asarray(data[k]) for k in data.files if k != "meta_json"}
        traj["meta"] = json.loads(str(data["meta_json"]))
    return traj


def mpc_frame_indices(t: np.ndarray, mpc_hz: float) -> np.ndarray:
    """Indices into the recorded timeline at each MPC tick (8 Hz by default)."""
    if len(t) == 0:
        return np.array([], dtype=int)
    t0 = float(t[0])
    duration = float(t[-1] - t0)
    if duration <= 0.0:
        return np.array([0], dtype=int)
    n_mpc = max(1, int(np.floor(duration * mpc_hz)) + 1)
    mpc_times = t0 + np.arange(n_mpc, dtype=np.float64) / float(mpc_hz)
    return np.clip(np.searchsorted(t, mpc_times, side="left"), 0, len(t) - 1)


def _pairwise_min_dist(pos: np.ndarray) -> float:
    n = int(pos.shape[0])
    if n < 2:
        return float("inf")
    mins = []
    for i in range(n):
        for j in range(i + 1, n):
            mins.append(float(np.linalg.norm(pos[i] - pos[j])))
    return float(min(mins))


def diagnose_recording_vs_settings(traj: dict, settings: SolverSettings) -> list[str]:
    """Human-readable mismatches between online NPZ and default axswarm yaml."""
    sp = traj["setpoint"]
    pos = traj["sim_pos"]
    vel = traj["sim_vel"]
    lines: list[str] = []
    n_drones = int(sp.shape[1])
    pos_lo = np.minimum(sp.min(axis=(0, 1)), pos.min(axis=(0, 1)))
    pos_hi = np.maximum(sp.max(axis=(0, 1)), pos.max(axis=(0, 1)))
    if np.any(pos_lo < settings.pos_min) or np.any(pos_hi > settings.pos_max):
        lines.append(
            f"position outside yaml box: recording [{pos_lo}]..[{pos_hi}] vs "
            f"pos_min={np.asarray(settings.pos_min)} pos_max={np.asarray(settings.pos_max)}"
        )
    speed = np.linalg.norm(vel, axis=-1)
    if float(speed.max()) > float(settings.vel_max):
        lines.append(
            f"speed exceeds vel_max={settings.vel_max}: recorded max {float(speed.max()):.2f} m/s"
        )
    if n_drones > 4:
        lines.append(
            f"n_drones={n_drones} (yaml tuned for 4); max_collisions={settings.max_collisions} "
            "only avoids the closest few neighbours per drone"
        )
    min_sep = float(_pairwise_min_dist(sp[0]))
    env = float(np.min(settings.collision_envelope))
    if min_sep < 2.0 * env:
        lines.append(
            f"tight formation: min pairwise setpoint distance {min_sep:.2f} m vs "
            f"collision envelope ~{env:.2f} m per axis"
        )
    return lines


def adapt_settings_for_recording(
    settings: SolverSettings, traj: dict, meta: dict, *, pad_m: float = 0.35
) -> SolverSettings:
    """Widen limits so offline replay matches iso_swarm workspace (24 drones, ~±2.7 m, fast sim vel)."""
    sp = traj["setpoint"]
    pos = traj["sim_pos"]
    vel = traj["sim_vel"]
    n_drones = int(sp.shape[1])

    pos_lo = np.minimum(sp.min(axis=(0, 1)), pos.min(axis=(0, 1))) - pad_m
    pos_hi = np.maximum(sp.max(axis=(0, 1)), pos.max(axis=(0, 1))) + pad_m
    pos_lo[2] = max(0.05, float(pos_lo[2]))

    speed = np.linalg.norm(vel, axis=-1)
    vel_max = float(max(1.73, np.percentile(speed, 95) * 1.15))

    min_sep_m = float(meta.get("min_separation_m", 0.32))
    half_sep = max(0.08, 0.5 * min_sep_m)
    collision_envelope = np.array([half_sep, half_sep, half_sep], dtype=np.float64)

    max_collisions = int(min(max(4, n_drones // 3), n_drones - 1))

    return settings.replace(
        pos_min=np.asarray(pos_lo, dtype=np.float64),
        pos_max=np.asarray(pos_hi, dtype=np.float64),
        vel_max=vel_max,
        collision_envelope=collision_envelope,
        max_collisions=max_collisions,
    )


def states_for_solve(
    sim_pos: np.ndarray, sim_vel: np.ndarray, *, clip_velocity: bool, vel_max: float
) -> np.ndarray:
    states = np.concatenate([sim_pos, sim_vel], axis=-1).astype(np.float32)
    if not clip_velocity:
        return states
    speed = np.linalg.norm(states[:, 3:], axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(vel_max) / np.maximum(speed, 1e-6))
    states = states.copy()
    states[:, 3:] *= scale
    return states


def replay_axswarm_on_recording(
    trajectory_path: str | Path,
    *,
    settings_path: str | Path | None = None,
    plot: bool = False,
    animate: bool = False,
    save_plot: str | None = None,
    save_animation: str | None = None,
    show: bool = True,
    animate_fps: float = 12.0,
    trail_frames: int = 25,
    max_drones_plot: int = 8,
    use_camera_frames: bool = False,
    warmup: bool = True,
    adapt_settings: bool = True,
    clip_velocity: bool = True,
    warn_every_fail: bool = False,
) -> dict:
    traj = load_online_npz(Path(trajectory_path))
    meta = traj["meta"]
    settings, dynamics = load_settings_yaml(
        Path(settings_path) if settings_path is not None else None
    )
    base_settings = settings
    issues = diagnose_recording_vs_settings(traj, base_settings)
    if issues:
        print("Recording vs default axswarm settings:")
        for line in issues:
            print(f"  - {line}")
    if adapt_settings:
        settings = adapt_settings_for_recording(settings, traj, meta)
        print(
            "Adapted settings for replay: "
            f"pos_min={np.asarray(settings.pos_min).round(2).tolist()} "
            f"pos_max={np.asarray(settings.pos_max).round(2).tolist()} "
            f"vel_max={settings.vel_max:.2f} "
            f"max_collisions={settings.max_collisions} "
            f"collision_envelope={np.asarray(settings.collision_envelope).round(3).tolist()}"
        )

    setpoint = traj["setpoint"]
    sim_pos = traj["sim_pos"]
    sim_vel = traj["sim_vel"]
    t = traj["t"]
    n_drones = int(setpoint.shape[1])

    if use_camera_frames:
        idx = np.arange(len(t), dtype=int)
        print(f"MPC replay: one solve per camera frame ({len(idx)} steps).")
    else:
        idx = mpc_frame_indices(t, float(settings.freq))
        print(f"MPC replay: {len(idx)} steps at {settings.freq} Hz (use --use-camera-frames for {len(t)} steps).")
    if len(idx) == 0:
        raise RuntimeError("Recording has no frames")

    initial_states = states_for_solve(
        sim_pos[0], sim_vel[0], clip_velocity=clip_velocity, vel_max=float(settings.vel_max)
    )
    solver_data = SolverData.init(
        setpoints={
            "pos": setpoint[idx[0]],
            "vel": np.zeros((n_drones, 3), dtype=np.float32),
            "acc": np.zeros((n_drones, 3), dtype=np.float32),
        },
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

    if warmup:
        solve(initial_states, solver_data, settings)

    planned_pos = np.zeros((len(idx), n_drones, 3), dtype=np.float32)
    planned_u_pos = np.zeros((len(idx), n_drones, 3), dtype=np.float32)
    success_flags = []
    solve_ms = []
    n_fail = 0

    for k, fi in enumerate(idx):
        states = states_for_solve(
            sim_pos[fi],
            sim_vel[fi],
            clip_velocity=clip_velocity,
            vel_max=float(settings.vel_max),
        )
        sp = {
            "pos": np.asarray(setpoint[fi], dtype=np.float32),
            "vel": np.zeros((n_drones, 3), dtype=np.float32),
            "acc": np.zeros((n_drones, 3), dtype=np.float32),
        }
        solver_data = solver_data.replace(
            setpoints={
                "pos": sp["pos"],
                "vel": sp["vel"],
                "acc": sp["acc"],
            }
        )
        t0 = time.perf_counter()
        success, _, solver_data = solve(states, solver_data, settings)
        solve_ms.append((time.perf_counter() - t0) * 1000.0)
        ok = bool(np.all(success))
        success_flags.append(ok)
        if not ok:
            n_fail += 1
            if warn_every_fail:
                logger.warning("solve failed at mpc step %d (frame %d)", k, int(fi))

        planned_u_pos[k] = np.asarray(solver_data.u_pos[:, 0])
        planned_pos[k] = np.asarray(solver_data.pos[:, 1])
        solver_data = solver_data.step(solver_data)

    # Compare axswarm first planned position to online actual on next MPC frame
    errs = []
    for k in range(len(idx) - 1):
        fi_next = int(idx[k + 1])
        err = np.linalg.norm(planned_pos[k] - sim_pos[fi_next], axis=-1)
        errs.append(err)
    errs = np.stack(errs, axis=0) if errs else np.zeros((0, n_drones))

    if n_fail:
        print(
            f"[INFO] {n_fail}/{len(success_flags)} MPC steps did not fully satisfy constraints "
            f"(common with 24-drone recordings + default yaml; use adapted settings or see --no-adapt-settings)."
        )

    summary = {
        "trajectory": str(Path(trajectory_path).resolve()),
        "n_frames": int(len(t)),
        "n_mpc_steps": int(len(idx)),
        "n_drones": n_drones,
        "mpc_hz": float(settings.freq),
        "adapted_settings": bool(adapt_settings),
        "clip_velocity": bool(clip_velocity),
        "solve_ok_frac": float(np.mean(success_flags)) if success_flags else 0.0,
        "solve_fail_count": int(n_fail),
        "solve_ms_mean": float(np.mean(solve_ms)) if solve_ms else 0.0,
        "plan_vs_sim_pos_mean_m": float(np.mean(errs)) if errs.size else float("nan"),
        "plan_vs_sim_pos_max_m": float(np.max(errs)) if errs.size else float("nan"),
        "meta": meta,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))

    traj_path = Path(trajectory_path)
    default_png = traj_path.with_name(traj_path.stem + "_replay_static.png")
    default_gif = traj_path.with_name(traj_path.stem + "_flight.gif")
    png_out = Path(save_plot) if save_plot else (default_png if plot else None)
    gif_out = Path(save_animation) if save_animation else (default_gif if animate else None)

    if plot or png_out is not None:
        print("[plot] Static 3D: online sim path vs setpoint vs axswarm plan (not MuJoCo viewer).")
        _plot_comparison(
            traj,
            idx,
            planned_pos,
            planned_u_pos,
            max_drones=max_drones_plot,
            save_path=png_out,
            show=show and plot,
        )
        if png_out is not None:
            print(f"[plot] Saved static figure: {png_out.resolve()}")

    if animate or gif_out is not None:
        print("[animate] Playing recorded online_control flight (sim_pos from NPZ).")
        _animate_recorded_flight(
            traj,
            save_path=gif_out,
            show=show and animate,
            fps=float(animate_fps),
            trail_frames=int(trail_frames),
            max_drones=max_drones_plot,
        )
        if gif_out is not None:
            print(f"[animate] Saved animation: {gif_out.resolve()}")

    if (plot or animate) and show:
        print("[plot] Close all matplotlib windows to exit the program.")

    return summary


def _plot_comparison(
    traj: dict,
    idx: np.ndarray,
    planned_pos: np.ndarray,
    planned_u_pos: np.ndarray,
    *,
    max_drones: int = 8,
    save_path: Path | None = None,
    show: bool = True,
):
    sim_pos = traj["sim_pos"]
    setpoint = traj["setpoint"]
    n_drones = min(int(sim_pos.shape[1]), max(1, int(max_drones)))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_drones, 2)))
    for d in range(n_drones):
        ax.plot(
            sim_pos[:, d, 0],
            sim_pos[:, d, 1],
            sim_pos[:, d, 2],
            color=colors[d],
            alpha=0.35,
            label=f"sim {d}" if d < 4 else None,
        )
        ax.plot(
            setpoint[:, d, 0],
            setpoint[:, d, 1],
            setpoint[:, d, 2],
            color=colors[d],
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            planned_pos[:, d, 0],
            planned_pos[:, d, 1],
            planned_pos[:, d, 2],
            color=colors[d],
            linewidth=2.0,
        )
    ax.set_title(
        f"Recorded Crazyflow flight (faint, {n_drones} drones), "
        "setpoint (dashed), axswarm MPC plan (bold)"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if n_drones <= 4:
        ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _animate_recorded_flight(
    traj: dict,
    *,
    save_path: Path | None = None,
    show: bool = True,
    fps: float = 12.0,
    trail_frames: int = 25,
    max_drones: int = 8,
) -> None:
    """Animate sim_pos from online_control — this is the actual recorded swarm flight."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    sim_pos = np.asarray(traj["sim_pos"])
    t = np.asarray(traj["t"], dtype=np.float64)
    n_frames, n_all = sim_pos.shape[0], sim_pos.shape[1]
    n_drones = min(n_all, max(1, int(max_drones)))
    sim_pos = sim_pos[:, :n_drones, :]
    trail_frames = max(1, int(trail_frames))

    lo = sim_pos.reshape(-1, 3).min(axis=0)
    hi = sim_pos.reshape(-1, 3).max(axis=0)
    pad = 0.15 * np.maximum(hi - lo, 0.5)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_drones, 2)))
    trails = [
        ax.plot([], [], [], color=colors[d], alpha=0.85, linewidth=1.5)[0] for d in range(n_drones)
    ]
    heads = [
        ax.scatter([], [], [], s=40, color=colors[d], depthshade=True) for d in range(n_drones)
    ]
    title = ax.set_title("")

    def _update(fi: int):
        i0 = max(0, int(fi) - trail_frames + 1)
        for d in range(n_drones):
            seg = sim_pos[i0 : int(fi) + 1, d]
            trails[d].set_data(seg[:, 0], seg[:, 1])
            trails[d].set_3d_properties(seg[:, 2])
            p = sim_pos[int(fi), d]
            heads[d]._offsets3d = ([p[0]], [p[1]], [p[2]])
        title.set_text(f"Recorded online flight  t={t[int(fi)]:.2f}s  frame {fi}/{n_frames - 1}")
        return trails + heads + [title]

    interval_ms = int(1000.0 / max(float(fps), 1.0))
    ani = FuncAnimation(fig, _update, frames=n_frames, interval=interval_ms, blit=False)

    if save_path is not None:
        ani.save(str(save_path), writer=PillowWriter(fps=max(float(fps), 1.0)))
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(
    trajectory: str,
    settings: str | None = None,
    plot: bool = False,
    animate: bool = False,
    save_plot: str | None = None,
    save_animation: str | None = None,
    no_show: bool = False,
    animate_fps: float = 12.0,
    trail_frames: int = 25,
    max_drones_plot: int = 8,
    use_camera_frames: bool = False,
    no_warmup: bool = False,
    no_adapt_settings: bool = False,
    no_clip_velocity: bool = False,
    warn_every_fail: bool = False,
):
    replay_axswarm_on_recording(
        trajectory,
        settings_path=settings,
        plot=plot,
        animate=animate,
        save_plot=save_plot,
        save_animation=save_animation,
        show=not no_show,
        animate_fps=animate_fps,
        trail_frames=trail_frames,
        max_drones_plot=max_drones_plot,
        use_camera_frames=use_camera_frames,
        warmup=not no_warmup,
        adapt_settings=not no_adapt_settings,
        clip_velocity=not no_clip_velocity,
        warn_every_fail=warn_every_fail,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    fire.Fire(main)
