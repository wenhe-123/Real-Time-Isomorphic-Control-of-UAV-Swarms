"""Replay logged morph points as offline Crazyflow drone targets."""

import argparse
import re
from collections import deque
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line


DEFAULT_LOG_PATH = Path("/home/mia/.cursor/projects/home-mia-semester-thesis/terminals/1.txt")
N_RE = re.compile(r"\bn=(?P<n>\d+)\b")
POINT_RE = re.compile(
    r"(?P<idx>\d+):\("
    r"(?P<x>[-+]?\d+(?:\.\d+)?),"
    r"(?P<y>[-+]?\d+(?:\.\d+)?),"
    r"(?P<z>[-+]?\d+(?:\.\d+)?)\)"
)


def load_points(log_path: Path, point_count: int) -> np.ndarray:
    """Parse terminal output and return frames x point_count x xyz."""
    frames: list[np.ndarray] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "points=" not in line:
            continue
        n_match = N_RE.search(line)
        if n_match is not None and int(n_match.group("n")) != point_count:
            continue

        pts = np.full((point_count, 3), np.nan, dtype=float)
        for m in POINT_RE.finditer(line):
            idx = int(m.group("idx"))
            if idx >= point_count:
                continue
            pts[idx] = [float(m.group("x")), float(m.group("y")), float(m.group("z"))]
        if np.isfinite(pts).all():
            frames.append(pts)
    if not frames:
        raise ValueError(f"No complete n={point_count} point frames found in {log_path}")
    return np.stack(frames, axis=0)


def normalize_for_crazyflow(
    points_mm: np.ndarray,
    xy_radius: float,
    z_center: float,
    z_amplitude: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Map terminal millimeter-like coordinates into a safe Crazyflow box."""
    pts = np.asarray(points_mm, dtype=float).copy()
    xy_scale = max(float(np.max(np.abs(pts[..., :2]))), 1.0)
    z_scale = max(float(np.max(np.abs(pts[..., 2]))), 1.0)

    out = np.zeros_like(pts)
    out[..., 0] = pts[..., 0] / xy_scale * xy_radius
    out[..., 1] = pts[..., 1] / xy_scale * xy_radius
    out[..., 2] = z_center + pts[..., 2] / z_scale * z_amplitude
    out[..., 2] = np.clip(out[..., 2], z_min, z_max)
    return out.astype(np.float32)


def closest_pair(points: np.ndarray) -> tuple[float, int, int]:
    """Return distance and indices of the closest pair in one target frame."""
    best_dist = float("inf")
    best_i = -1
    best_j = -1
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            dist = float(np.linalg.norm(points[i] - points[j]))
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_j = j
    return best_dist, best_i, best_j


def print_spacing_report(traj: np.ndarray) -> None:
    """Print the closest target spacing across all loaded frames."""
    min_dist = float("inf")
    min_frame = -1
    min_i = -1
    min_j = -1
    for frame_idx, frame in enumerate(traj):
        dist, i, j = closest_pair(frame)
        if dist < min_dist:
            min_dist = dist
            min_frame = frame_idx
            min_i = i
            min_j = j
    print(
        f"Closest target spacing: frame={min_frame}, pair=({min_i},{min_j}), "
        f"dist={min_dist:.2f}m"
    )


def plot_trajectories(traj: np.ndarray, out_path: Path) -> None:
    """Save a 3D plot of all loaded target trajectories."""
    n_drones = traj.shape[1]
    labels = [str(i) for i in range(n_drones)]
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, (label, color) in enumerate(zip(labels, colors)):
        xyz = traj[:, i, :]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, label=f"{i} {label}", linewidth=1.8)
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, marker="o", s=22)
        ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker="x", s=30)

    ax.set_title(f"{n_drones}-point trajectories from terminal log")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    ax.set_box_aspect((1, 1, 0.7))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def stepped_target(t: float, points: np.ndarray, waypoint_period: float) -> np.ndarray:
    """Hold one target frame for each waypoint period."""
    idx = int(np.floor(t / max(waypoint_period, 1e-6)))
    idx = max(0, min(idx, points.shape[0] - 1))
    return points[idx]


def run_crazyflow(traj: np.ndarray, duration: float, fps: int, waypoint_period: float) -> None:
    """Run the Crazyflow visualization and state controller."""
    n_worlds = 1
    n_drones = traj.shape[1]
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.state)
    sim.reset()

    if duration <= 0.0:
        duration = max((traj.shape[0] - 1) * waypoint_period, waypoint_period)
    takeoff_duration = 2.0
    transition_duration = 2.0
    # Keep line history short: many drones x long line history can exceed MuJoCo maxgeom.
    trail_history = int(max(2, min(8, 96 // max(n_drones, 1))))
    pos_buffer = deque(maxlen=trail_history)
    formation_radius = max(0.90, 0.35 * n_drones / np.pi)
    formation_angles = np.linspace(0.0, 2.0 * np.pi, n_drones, endpoint=False)
    takeoff_start = np.stack(
        [
            formation_radius * np.cos(formation_angles),
            formation_radius * np.sin(formation_angles),
            np.full((n_drones,), 0.60, dtype=float),
        ],
        axis=1,
    ).astype(np.float32)
    takeoff_hover = np.stack(
        [
            formation_radius * np.cos(formation_angles),
            formation_radius * np.sin(formation_angles),
            np.full((n_drones,), 1.20, dtype=float),
        ],
        axis=1,
    ).astype(np.float32)
    first_target = traj[0].copy()
    first_target[:, 2] = np.maximum(first_target[:, 2], 1.10)
    smooth_target = takeoff_start.copy()
    alpha_smooth = 0.06
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))

    # Start the simulated bodies at the takeoff formation. Otherwise reset() starts
    # all drones on the default grid while the controller immediately pulls them
    # toward the custom formation, which can cause collision/tilt at t=0.
    zeros = jnp.zeros_like(sim.data.states.pos)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=jnp.asarray(takeoff_start[None, :, :], device=sim.device),
            vel=zeros,
            ang_vel=zeros,
        )
    )
    min_takeoff_dist = float(
        np.min(
            [
                np.linalg.norm(takeoff_start[i, :2] - takeoff_start[j, :2])
                for i in range(n_drones)
                for j in range(i + 1, n_drones)
            ]
        )
    )
    print(
        f"Takeoff formation radius={formation_radius:.2f}m, "
        f"min_xy_dist={min_takeoff_dist:.2f}m, z={takeoff_start[0, 2]:.2f}->{takeoff_hover[0, 2]:.2f}m, "
        f"trail_history={trail_history}"
    )

    try:
        total_duration = float(duration) + float(takeoff_duration) + float(transition_duration)
        for i in range(int(total_duration * sim.control_freq)):
            t = i / sim.control_freq
            if t < takeoff_duration:
                alpha = t / max(takeoff_duration, 1e-6)
                alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0))
                raw_target = (1.0 - alpha) * takeoff_start + alpha * takeoff_hover
            elif t < takeoff_duration + transition_duration:
                alpha = (t - takeoff_duration) / max(transition_duration, 1e-6)
                alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0))
                raw_target = (1.0 - alpha) * takeoff_hover + alpha * first_target
            else:
                raw_target = stepped_target(
                    t - takeoff_duration - transition_duration,
                    traj,
                    waypoint_period,
                )
            smooth_target = alpha_smooth * raw_target + (1.0 - alpha_smooth) * smooth_target

            cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
            cmd[..., :3] = smooth_target
            # State command layout is [pos, vel, acc, yaw, roll_rate, pitch_rate, yaw_rate].
            # Keep yaw/rates at zero; do not write quaternion values here.
            cmd[..., 9] = 0.0

            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)

            if i % 8 == 0:
                pos_buffer.append(sim.data.states.pos[0].copy())

            if ((i * fps) % sim.control_freq) < fps:
                if len(pos_buffer) > 1:
                    lines = np.asarray(pos_buffer)
                    for d in range(n_drones):
                        draw_line(
                            sim,
                            lines[:, d, :],
                            rgba=colors[d].tolist(),
                            start_size=0.5,
                            end_size=2.0,
                        )
                sim.render()
    finally:
        sim.close()


def main() -> None:
    """CLI entry point for offline Crazyflow control."""
    parser = argparse.ArgumentParser(description="Run terminal point trajectories in Crazyflow.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--point-count", type=int, default=14)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--waypoint-period", type=float, default=1.0)
    parser.add_argument("--xy-radius", type=float, default=1.20)
    parser.add_argument("--z-center", type=float, default=1.40)
    parser.add_argument("--z-amplitude", type=float, default=0.35)
    parser.add_argument("--z-min", type=float, default=1.10)
    parser.add_argument("--z-max", type=float, default=2.10)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--plot", type=Path, default=Path("offline_control_trajectory.png"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    points_mm = load_points(args.log, point_count=int(args.point_count))
    traj = normalize_for_crazyflow(
        points_mm,
        xy_radius=float(args.xy_radius),
        z_center=float(args.z_center),
        z_amplitude=float(args.z_amplitude),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
    )
    print_spacing_report(traj)
    plot_trajectories(traj, args.plot)
    print(
        f"Loaded {traj.shape[0]} frames, {traj.shape[1]} point trajectories, "
        f"waypoint_period={args.waypoint_period:.2f}s, "
        f"xy_radius={args.xy_radius:.2f}m."
    )
    print(f"Saved trajectory plot to {args.plot}")

    if not args.plot_only:
        run_crazyflow(
            traj,
            duration=float(args.duration),
            fps=int(args.fps),
            waypoint_period=float(args.waypoint_period),
        )


if __name__ == "__main__":
    main()
