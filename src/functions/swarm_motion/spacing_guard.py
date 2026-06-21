"""Target spacing diagnostics."""

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
        env = self.collision_envelope_m
        print(
            f"[spacing {self.label}] n={self.n_drones} open={self.open_alpha:.2f} "
            f"mm_min={self.min_mm:.1f}mm | "
            f"pre_filter raw={self.min_raw_m:.3f}m pair={self.pair_raw} "
            f"(Δvs min_sep={gap_raw:+.3f}m, vs axswarm_env={self.min_raw_m - env:+.3f}m)"
        )
        if self.min_raw_m < env - 0.02:
            print(
                f"  WARN: morph→world spacing tighter than axswarm collision_envelope ({env:.2f}m); "
                f"raise --morph-world-scale or --min-separation-m."
            )
