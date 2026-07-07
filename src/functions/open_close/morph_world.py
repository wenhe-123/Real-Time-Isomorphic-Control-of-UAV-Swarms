"""Morph sample mm coordinates → Crazyflow world meters (workspace scale config)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions.open_close.morph_lp_plot import MORPH_PLANE_RADIUS_A, MORPH_PLANE_RADIUS_B, mode_epsilon_pair
from functions.open_close.morph_renderers import init_fixed_surface_points, mapped_fixed_surface_points


@dataclass
class ScaleConfig:
    """Morph mm→sim-m workspace scaling parameters."""

    xy_radius: float
    hover_z: float
    z_amplitude: float
    reference_xy_extent_mm: float
    reference_z_extent_mm: float
    z_mm_scale: float = 1.0
    morph_world_scale: float = 0.55


def summarize_target_workspace(points_m: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Summarize XY extent and axis-aligned bounds of a target set.

    Args:
        points_m: Target positions in sim meters, shape ``(N, 3)``.

    Returns:
        Tuple ``(max_xy_radius_m, xyz_min, xyz_max)``. Returns zeros when input
        is empty or malformed.
    """
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
    """Generate fixed-index morph surface samples in millimeters.

    Args:
        point_count: Number of surface sample indices to initialize.
        radius_mm: Superellipsoid radius in millimeters.
        morph_mode: Morph mode index (1–5).
        open_alpha: Openness in ``[0, 1]``.
        shape_t: Optional left-hand shape blend for ε pair selection.

    Returns:
        Mapped surface points in mm, shape ``(point_count, 3)``.
    """
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
    """Compute XY meters-per-mm without coupling to the legacy gesture Z band.

    Args:
        points_mm: Morph samples in mm (values unused; kept for API symmetry).
        scale: Workspace scaling configuration.

    Returns:
        XY scale factor in meters per millimeter.
    """
    pts = np.asarray(points_mm, dtype=np.float32)
    xy_den = max(float(scale.reference_xy_extent_mm), 1.0)
    xy_s0 = float(scale.xy_radius) / xy_den
    world_scale = float(max(0.25, scale.morph_world_scale))
    return xy_s0 * world_scale


def _morph_z_rel_scale(points_mm: np.ndarray, scale: ScaleConfig) -> float:
    """Compute Z topology meters-per-mm before anchoring at ``hover_z``.

    Args:
        points_mm: Morph samples in mm (values unused; kept for API symmetry).
        scale: Workspace scaling configuration.

    Returns:
        Relative Z scale factor in meters per millimeter.
    """
    pts = np.asarray(points_mm, dtype=np.float32)
    z_den = max(float(scale.reference_z_extent_mm), 1.0)
    z_s0 = (float(scale.xy_radius) / z_den) * float(scale.z_mm_scale)
    world_scale = float(max(0.25, scale.morph_world_scale))
    return z_s0 * world_scale


def normalize_morph_points_at_hover(
    points_mm: np.ndarray,
    scale: ScaleConfig,
) -> np.ndarray:
    """Map morph mm targets to sim m with superellipsoid center at hover height.

    Recenters samples in mm so the formation centroid lands on ``(0, 0, hover_z)``.

    Args:
        points_mm: Morph samples in millimeters, shape ``(N, 3)``.
        scale: Workspace scaling configuration.

    Returns:
        Normalized target in sim meters, shape ``(N, 3)``.
    """
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
