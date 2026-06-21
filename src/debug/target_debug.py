"""Debug print helpers for drone target arrays."""

from __future__ import annotations

import numpy as np

from functions.swarm_motion.spacing_guard import closest_pair


def debug_print_drone_targets(
    pts_m: np.ndarray,
    *,
    frame_idx: int,
    morph_mode: int,
    open_v: float | None,
    label: str,
    compare_to: np.ndarray | None = None,
    min_separation_m: float | None = None,
    snap_state: str | None = None,
) -> tuple[float, int, int]:
    """Print per-drone world xyz; return (min_pair_dist_m, i, j) for HUD."""
    p = np.asarray(pts_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 3:
        print(f"[debug {label}] f={frame_idx} invalid shape {getattr(p, 'shape', None)}")
        return float("inf"), -1, -1
    op = f"{float(open_v):.3f}" if open_v is not None else "-"
    z0, z1 = float(np.min(p[:, 2])), float(np.max(p[:, 2]))
    xy_r = np.linalg.norm(p[:, :2], axis=1)
    d_min, pi, pj = closest_pair(p)
    snap_s = f" snap={snap_state}" if snap_state else ""
    line = (
        f"[debug {label}] frame={frame_idx} M={int(morph_mode)} open={op}{snap_s} n={p.shape[0]} "
        f"z_span={z1 - z0:.3f}m z[{z0:.3f},{z1:.3f}] xy_r_max={float(np.max(xy_r)):.3f}m "
        f"min_pair=({pi},{pj}) d={d_min:.4f}m"
    )
    if min_separation_m is not None and d_min < float(min_separation_m) - 1e-5:
        line += f" *** closer than min_sep={float(min_separation_m):.3f}m ***"
    if compare_to is not None:
        c = np.asarray(compare_to, dtype=np.float64)
        if c.shape == p.shape:
            err = np.linalg.norm(p - c, axis=1)
            line += f" | vs_ref mean={float(np.mean(err)):.3f}m max={float(np.max(err)):.3f}m"
    print(line, flush=True)
    for i in range(p.shape[0]):
        print(
            f"  drone {i:2d}: x={p[i, 0]:8.4f} y={p[i, 1]:8.4f} z={p[i, 2]:8.4f}",
            flush=True,
        )
    return d_min, pi, pj


def debug_print_drone_positions(
    pts_m: np.ndarray,
    *,
    frame_idx: int,
    label: str,
    compare_to: np.ndarray | None = None,
) -> None:
    """Print each drone xyz (compact; for --debug-drone-pos-every)."""
    p = np.asarray(pts_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 3:
        print(f"[pos {label}] f={frame_idx} invalid shape {getattr(p, 'shape', None)}", flush=True)
        return
    d_min, pi, pj = closest_pair(p)
    line = (
        f"[pos {label}] frame={frame_idx} n={p.shape[0]} "
        f"min_pair=({pi},{pj}) d={d_min:.4f}m"
    )
    if compare_to is not None:
        c = np.asarray(compare_to, dtype=np.float64)
        if c.shape == p.shape:
            err = np.linalg.norm(p - c, axis=1)
            line += f" | track_err mean={float(np.mean(err)):.4f}m max={float(np.max(err)):.4f}m"
    print(line, flush=True)
    for i in range(p.shape[0]):
        print(
            f"  drone {i:2d}: x={p[i, 0]:8.4f} y={p[i, 1]:8.4f} z={p[i, 2]:8.4f}",
            flush=True,
        )
