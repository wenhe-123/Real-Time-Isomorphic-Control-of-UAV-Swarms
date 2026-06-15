"""Target spacing and per-step motion caps (shared by online_control and axswarm)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def closest_pair(points: np.ndarray) -> tuple[float, int, int]:
    """Smallest pairwise distance (m) and drone indices (i, j)."""
    p = np.asarray(points, dtype=np.float64)
    n = int(p.shape[0])
    if n < 2:
        return float("inf"), -1, -1
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        return float("inf"), -1, -1
    flat = d2[iu, ju]
    k = int(np.argmin(flat))
    return float(np.sqrt(float(flat[k]))), int(iu[k]), int(ju[k])


@dataclass(frozen=True)
class FormationSpacingReport:
    """Spacing audit before/after guards (compare to axswarm ``collision_envelope``)."""

    label: str
    n_drones: int
    open_alpha: float
    min_mm: float
    min_raw_m: float
    min_safe_m: float
    pair_raw: tuple[int, int]
    pair_safe: tuple[int, int]
    min_separation_m: float
    collision_envelope_m: float

    def print_lines(self) -> None:
        gap_raw = self.min_separation_m - self.min_raw_m
        gap_safe = self.min_separation_m - self.min_safe_m
        env = self.collision_envelope_m
        print(
            f"[spacing {self.label}] n={self.n_drones} open={self.open_alpha:.2f} "
            f"mm_min={self.min_mm:.1f}mm | "
            f"pre_filter raw={self.min_raw_m:.3f}m pair={self.pair_raw} "
            f"(Δvs min_sep={gap_raw:+.3f}m, vs axswarm_env={self.min_raw_m - env:+.3f}m)"
        )
        if abs(self.min_safe_m - self.min_raw_m) > 1e-4:
            print(
                f"  after enforce_min_separation: {self.min_safe_m:.3f}m pair={self.pair_safe} "
                f"(Δvs min_sep={gap_safe:+.3f}m)"
            )
        if self.min_raw_m < env - 0.02:
            print(
                f"  WARN: morph→world spacing tighter than axswarm collision_envelope ({env:.2f}m); "
                f"raise --morph-world-scale or --min-separation-m."
            )


def clamp_targets_step(prev: np.ndarray, nxt: np.ndarray, max_step_m: float) -> np.ndarray:
    """Per-drone clamp: move from prev toward nxt by at most max_step_m (L2 per row)."""
    if max_step_m <= 0:
        return np.asarray(nxt, dtype=np.float32)
    p = np.asarray(prev, dtype=np.float64)
    x = np.asarray(nxt, dtype=np.float64)
    d = x - p
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_step_m) / np.maximum(dist, 1e-9))
    out = p + d * s
    return out.astype(np.float32)


def enforce_min_separation(points: np.ndarray, min_sep: float, *, iters: int = 6) -> np.ndarray:
    """Lightweight collision guard: iteratively repel pairs closer than ``min_sep``."""
    pts = np.asarray(points, dtype=np.float64).copy()
    n = int(pts.shape[0])
    if n < 2 or min_sep <= 0:
        return pts.astype(np.float32)
    ms = float(min_sep)
    for _ in range(max(1, int(iters))):
        disp = np.zeros_like(pts)
        moved = False
        for i in range(n - 1):
            d = pts[i + 1 :] - pts[i]
            dist = np.linalg.norm(d, axis=1)
            mask = dist < ms
            if not np.any(mask):
                continue
            moved = True
            idx = np.where(mask)[0]
            dist_m = np.maximum(dist[idx], 1e-6)
            dir_m = d[idx] / dist_m[:, None]
            push = 0.5 * (ms - dist_m)[:, None] * dir_m
            disp[i] -= np.sum(push, axis=0)
            disp[i + 1 + idx] += push
        pts += disp
        if not moved:
            break
    return pts.astype(np.float32)


def conservative_degraded_target(
    gesture: np.ndarray,
    anchor: np.ndarray,
    *,
    min_separation_m: float,
    sep_mult: float = 1.3,
    gesture_blend: float = 0.2,
    max_step_m: float | None = None,
) -> np.ndarray:
    """On MPC failure: move slowly from anchor toward gesture with extra spacing."""
    g = np.asarray(gesture, dtype=np.float32)
    a = np.asarray(anchor, dtype=np.float32)
    b = float(np.clip(gesture_blend, 0.0, 1.0))
    tgt = ((1.0 - b) * a + b * g).astype(np.float32)
    if max_step_m is not None and max_step_m > 0:
        tgt = clamp_targets_step(a, tgt, max_step_m)
    return enforce_min_separation(
        tgt, float(min_separation_m) * float(sep_mult), iters=12
    )
