"""Morph sample mm coordinates → Crazyflow world meters (workspace scale config)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions.open_close.morph_lp_plot import MORPH_PLANE_RADIUS_A, MORPH_PLANE_RADIUS_B, mode_epsilon_pair
from functions.open_close.morph_renderers import init_fixed_surface_points, mapped_fixed_surface_points


@dataclass
class ScaleConfig:
    xy_radius: float
    hover_z: float
    z_amplitude: float
    reference_xy_extent_mm: float
    reference_z_extent_mm: float
    z_mm_scale: float = 1.0
    morph_world_scale: float = 0.55
    # Legacy gesture band (normalize_morph_points only); defaults track hover_z.
    z_center: float | None = None
    z_min: float | None = None
    z_max: float | None = None

    def __post_init__(self) -> None:
        hz = float(self.hover_z)
        if self.z_center is None:
            object.__setattr__(self, "z_center", hz)
        if self.z_min is None:
            object.__setattr__(self, "z_min", hz - float(self.z_amplitude))
        if self.z_max is None:
            object.__setattr__(self, "z_max", hz + float(self.z_amplitude))


def summarize_target_workspace(points_m: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (max_xy_radius_m, xyz_min, xyz_max) for a target set in meters."""
    pts = np.asarray(points_m, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        z = np.zeros((3,), dtype=np.float32)
        return 0.0, z, z
    xy_r = np.linalg.norm(pts[:, :2], axis=1)
    return float(np.max(xy_r)), np.min(pts[:, :3], axis=0), np.max(pts[:, :3], axis=0)


def fixed_morph_points(
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
) -> np.ndarray:
    """Generate fixed indexed morph points (same as webcam/Orbbec demos)."""
    init_fixed_surface_points(point_count)
    epsilon1, epsilon2 = mode_epsilon_pair(int(morph_mode), shape_t)
    return mapped_fixed_surface_points(
        radius=float(radius_mm),
        open_alpha=float(open_alpha),
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        plane_radius_a=MORPH_PLANE_RADIUS_A,
        plane_radius_b=MORPH_PLANE_RADIUS_B,
        morph_mode=int(morph_mode),
    )


def _morph_xy_scale(points_mm: np.ndarray, scale: ScaleConfig) -> float:
    """XY meters-per-mm without coupling to the legacy gesture Z band."""
    pts = np.asarray(points_mm, dtype=np.float32)
    xy_den = max(float(scale.reference_xy_extent_mm), 1.0)
    xy_s0 = float(scale.xy_radius) / xy_den
    world_scale = float(max(0.25, scale.morph_world_scale))
    return xy_s0 * world_scale


def _morph_z_rel_scale(points_mm: np.ndarray, scale: ScaleConfig) -> float:
    """Z topology meters-per-mm (relative offsets before anchoring at ``hover_z``)."""
    pts = np.asarray(points_mm, dtype=np.float32)
    z_den = max(float(scale.reference_z_extent_mm), 1.0)
    z_s0 = (float(scale.xy_radius) / z_den) * float(scale.z_mm_scale)
    world_scale = float(max(0.25, scale.morph_world_scale))
    return z_s0 * world_scale


def _morph_world_scales(
    points_mm: np.ndarray,
    scale: ScaleConfig,
) -> tuple[float, float]:
    """Return ``(xy_s, z_s)`` meters-per-mm scale factors for morph mapping."""
    pts = np.asarray(points_mm, dtype=np.float32)
    xy_den = max(float(scale.reference_xy_extent_mm), 1.0)
    z_den = max(float(scale.reference_z_extent_mm), 1.0)
    xy_s0 = float(scale.xy_radius) / xy_den
    z_s0 = (float(scale.xy_radius) / z_den) * float(scale.z_mm_scale)
    world_scale = float(max(0.25, scale.morph_world_scale))
    z_off = pts[:, 2] * z_s0 * world_scale
    max_d = float(np.max(z_off)) if z_off.size else 0.0
    min_d = float(np.min(z_off)) if z_off.size else 0.0
    z_top = float(scale.z_max) - float(scale.z_center)
    z_bot = float(scale.z_center) - float(scale.z_min)
    margin = 5e-3
    s_up = (z_top - margin) / max(max_d, 1e-9) if max_d > 1e-9 else 1e9
    s_dn = (z_bot - margin) / max(-min_d, 1e-9) if min_d < -1e-9 else 1e9
    s_fit = float(min(1.0, s_up, s_dn))
    xy_s = xy_s0 * s_fit * world_scale
    z_s = z_s0 * s_fit * world_scale
    return xy_s, z_s


def normalize_morph_points(
    points_mm: np.ndarray,
    scale: ScaleConfig,
) -> np.ndarray:
    """Map morph-renderer millimeter targets into the Crazyflow workspace."""
    pts = np.asarray(points_mm, dtype=np.float32)
    xy_s, z_s = _morph_world_scales(pts, scale)
    zc = float(scale.z_center)
    zmin = float(scale.z_min)
    zmax = float(scale.z_max)
    z_raw = zc + pts[:, 2] * z_s
    z_clip = np.clip(z_raw, zmin, zmax)
    n_clip = int(np.sum(np.abs(z_clip - z_raw) > 1e-6))
    if n_clip > 0:
        margin_z = 5e-3
        z_hi_need = float(np.max(z_raw)) - zc
        z_lo_need = zc - float(np.min(z_raw))
        s_z_hi = (zmax - margin_z - zc) / max(z_hi_need, 1e-9)
        s_z_lo = (zc - margin_z - zmin) / max(z_lo_need, 1e-9)
        shrink = float(min(1.0, s_z_hi, s_z_lo))
        if shrink < 0.999:
            xy_s *= shrink
            z_s *= shrink
            z_raw = zc + pts[:, 2] * z_s
            z_clip = np.clip(z_raw, zmin, zmax)
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = pts[:, 0] * xy_s
    out[:, 1] = pts[:, 1] * xy_s
    out[:, 2] = z_clip.astype(np.float32)
    return out


def normalize_morph_points_at_hover(
    points_mm: np.ndarray,
    scale: ScaleConfig,
) -> np.ndarray:
    """Map morph mm → world m; superellipsoid center at (0, 0, hover_z) in sim."""
    pts = np.asarray(points_mm, dtype=np.float32)
    # Fixed surface samples are not symmetric about the parametric origin (plane
    # blend / mode layout); recenter in mm so sim centroid lands on hover_z.
    cent = pts.mean(axis=0)
    q = pts - cent
    xy_s = _morph_xy_scale(pts, scale)
    z_s = _morph_z_rel_scale(pts, scale)
    hover_z = float(scale.hover_z)
    z_rel = q[:, 2] * z_s
    if z_rel.size:
        amp = float(scale.z_amplitude)
        if amp > 0.0:
            z_need = max(float(np.max(z_rel)), float(-np.min(z_rel)))
            if z_need > amp + 1e-9:
                z_rel *= amp / z_need
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = q[:, 0] * xy_s
    out[:, 1] = q[:, 1] * xy_s
    out[:, 2] = (hover_z + z_rel).astype(np.float32)
    return out
