"""Morph sample mm coordinates → Crazyflow world meters (workspace scale config)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions.open_close.morph_lp_plot import MORPH_PLANE_RADIUS_A, MORPH_PLANE_RADIUS_B, mode_epsilon_pair
from functions.open_close.morph_renderers import init_fixed_surface_points, mapped_fixed_surface_points
from functions.swarm_motion.formation_spacing import (
    closed_shell_scale_boost,
    uniform_scale_formation_to_max_extent,
)


@dataclass
class ScaleConfig:
    xy_radius: float
    z_center: float
    z_amplitude: float
    z_min: float
    z_max: float
    reference_xy_extent_mm: float
    reference_z_extent_mm: float
    z_mm_scale: float = 1.0
    morph_world_scale: float = 0.55
    formation_max_extent_m: float | None = None


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


def normalize_morph_points(
    points_mm: np.ndarray,
    scale: ScaleConfig,
    *,
    n_drones: int | None = None,
    open_alpha: float | None = None,
    min_separation_m: float | None = None,
) -> np.ndarray:
    """Map morph-renderer millimeter targets into the Crazyflow workspace."""
    pts = np.asarray(points_mm, dtype=np.float32)
    xy_den = max(float(scale.reference_xy_extent_mm), 1.0)
    z_den = max(float(scale.reference_z_extent_mm), 1.0)
    xy_s0 = float(scale.xy_radius) / xy_den
    z_s0 = (float(scale.xy_radius) / z_den) * float(scale.z_mm_scale)
    world_scale = float(max(0.25, scale.morph_world_scale))
    spacing_extra = 1.0
    if (
        n_drones is not None
        and open_alpha is not None
        and min_separation_m is not None
        and float(min_separation_m) > 0.0
    ):
        shell_boost = closed_shell_scale_boost(
            n_drones=int(n_drones),
            open_alpha=float(open_alpha),
            min_separation_m=float(min_separation_m),
            user_scale=1.0,
        )
        spacing_extra = float(max(1.0, shell_boost / max(world_scale, 1e-6)))
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
    if spacing_extra > 1.0 + 1e-6:
        zc = float(scale.z_center)
        zmin = float(scale.z_min)
        zmax = float(scale.z_max)
        margin_z = 5e-3
        z_test = zc + pts[:, 2] * z_s * spacing_extra
        z_hi_need = float(np.max(z_test)) - zc
        z_lo_need = zc - float(np.min(z_test))
        s_extra = min(
            spacing_extra,
            (zmax - margin_z - zc) / max(z_hi_need, 1e-9),
            (zc - margin_z - zmin) / max(z_lo_need, 1e-9),
        )
        s_extra = float(max(1.0, s_extra))
        xy_s *= s_extra
        z_s *= s_extra
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
    cap_m = scale.formation_max_extent_m
    if cap_m is not None and float(cap_m) > 0.0:
        out = uniform_scale_formation_to_max_extent(
            out,
            float(cap_m),
            z_center=float(scale.z_center),
        )
    return out
