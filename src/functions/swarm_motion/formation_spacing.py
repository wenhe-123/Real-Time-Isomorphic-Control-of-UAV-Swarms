"""Audit morph→world spacing vs ``enforce_min_separation`` / axswarm collision envelope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions.swarm_motion.spacing_guard import FormationSpacingReport, closest_pair, enforce_min_separation


def closed_shell_scale_boost(
    *,
    n_drones: int,
    open_alpha: float,
    min_separation_m: float,
    user_scale: float = 1.0,
) -> float:
    """Extra mm→world scale when a closed shell packs surface samples (open≈0, n large).

    With default ``reference_xy_extent_mm=100`` and ``xy_radius=3``, n=24 open=0 often lands
    near ~0.20 m raw spacing while ``min_separation_m`` / axswarm envelope target ~0.30 m.
    """
    oa = float(np.clip(open_alpha, 0.0, 1.0))
    if oa >= 0.32 or n_drones < 12:
        return float(max(1.0, user_scale))
    est_raw_min_m = max(0.11, 0.0088 * float(n_drones))
    auto = float(min_separation_m) / est_raw_min_m * 1.06
    auto = float(np.clip(auto, 1.0, 1.85))
    return float(max(user_scale, auto))


def audit_formation_spacing(
    points_mm: np.ndarray,
    points_m_raw: np.ndarray,
    *,
    label: str,
    n_drones: int,
    open_alpha: float,
    min_separation_m: float,
    collision_envelope_m: float = 0.30,
) -> FormationSpacingReport:
    """Compare topo/mm targets, pre-filter world targets, and after ``enforce_min_separation``."""
    mm = np.asarray(points_mm, dtype=np.float64)
    raw = np.asarray(points_m_raw, dtype=np.float64)
    safe = enforce_min_separation(raw, float(min_separation_m), iters=10)
    d_mm, _, _ = closest_pair(mm)
    d_raw, pi, pj = closest_pair(raw)
    d_safe, si, sj = closest_pair(safe)
    rep = FormationSpacingReport(
        label=label,
        n_drones=int(n_drones),
        open_alpha=float(open_alpha),
        min_mm=float(d_mm),
        min_raw_m=float(d_raw),
        min_safe_m=float(d_safe),
        pair_raw=(pi, pj),
        pair_safe=(si, sj),
        min_separation_m=float(min_separation_m),
        collision_envelope_m=float(collision_envelope_m),
    )
    rep.print_lines()
    return rep


def uniform_scale_formation_to_max_extent(
    points: np.ndarray,
    max_extent_m: float,
    *,
    z_center: float | None = None,
) -> np.ndarray:
    """Uniformly shrink a formation so max(X/Y/Z span) <= ``max_extent_m``.

    Scales about the XY origin and ``z_center`` (default: mean Z) so hover/morph Z
    semantics stay aligned with :func:`lift_morph_to_hover_z`.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return pts
    limit = float(max_extent_m)
    if not np.isfinite(limit) or limit <= 0.0:
        return pts
    ext = pts.max(axis=0) - pts.min(axis=0)
    span = float(np.max(ext[:3]))
    if span <= limit + 1e-9:
        return pts
    zc = float(np.mean(pts[:, 2]) if z_center is None else z_center)
    s = limit / max(span, 1e-9)
    pts[:, 0] *= s
    pts[:, 1] *= s
    pts[:, 2] = zc + (pts[:, 2] - zc) * s
    return pts.astype(np.float32, copy=False)


def lift_morph_to_hover_z(points: np.ndarray, hover_z: float) -> np.ndarray:
    """Keep morph XY distribution; shift formation so mean Z equals hover_z."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 2] += float(hover_z) - float(np.mean(pts[:, 2]))
    return pts


def formation_z_over_xy(points: np.ndarray) -> float:
    """Extent ratio z / max(x,y); ~1 for sphere, ~0 for flat plane in XY."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape[0] < 2:
        return 1.0
    ext = p.max(0) - p.min(0)
    return float(ext[2] / max(ext[0], ext[1], 1e-9))


def is_flat_formation(points: np.ndarray, *, z_xy_thresh: float = 0.55) -> bool:
    return formation_z_over_xy(points) < float(z_xy_thresh)


def is_sphere_like_formation(points: np.ndarray, *, z_xy_thresh: float = 0.82) -> bool:
    return formation_z_over_xy(points) > float(z_xy_thresh)


def formation_aspect_ratio(points: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (z_extent/xy_extent, z_clipped_fraction, extents_xyz)."""
    p = np.asarray(points, dtype=np.float64)
    ext = p.max(0) - p.min(0)
    xy = float(max(ext[0], ext[1], 1e-9))
    z_over_xy = float(ext[2] / xy)
    return z_over_xy, 0.0, ext


def debug_morph_mapping(
    points_mm: np.ndarray,
    points_m: np.ndarray,
    *,
    scale: object,
    world_scale: float,
    s_fit: float,
    z_clip_count: int,
) -> None:
    """Explain flattening: mm sphere vs world extents and z clip."""
    mm = np.asarray(points_mm, dtype=np.float64)
    w = np.asarray(points_m, dtype=np.float64)
    mm_ext = mm.max(0) - mm.min(0)
    w_ext = w.max(0) - w.min(0)
    mm_ratio = float(mm_ext[2] / max(mm_ext[0], mm_ext[1], 1e-9))
    w_ratio = float(w_ext[2] / max(w_ext[0], w_ext[1], 1e-9))
    print(
        f"[morph shape] mm extent={mm_ext.round(1)} z/xy={mm_ratio:.3f} (1.0≈sphere in topo)"
    )
    print(
        f"[morph shape] world extent={w_ext.round(3)} z/xy={w_ratio:.3f} "
        f"world_scale={world_scale:.3f} s_fit={s_fit:.3f} z_clipped={z_clip_count}/{w.shape[0]}"
    )
    if z_clip_count > 0 and w_ratio < 0.95:
        print(
            "  WARN: Z clamped to [z_min,z_max] after scaling — formation looks flattened. "
            "Fixed in normalize_morph_points by fitting s_fit with world_scale first."
        )
    elif abs(w_ratio - mm_ratio) > 0.08:
        print("  WARN: world z/xy differs from mm — check z_mm_scale or non-uniform scaling.")


@dataclass(frozen=True)
class MorphMappingProbe:
    """One (mode, open) probe for the spacing check script."""

    morph_mode: int
    open_alpha: float
    min_raw_m: float
    min_safe_m: float
    scale_applied: float
