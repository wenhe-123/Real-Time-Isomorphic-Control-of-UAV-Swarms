"""Bootstrap and per-frame morph target glue."""

from __future__ import annotations

import numpy as np

from functions.mode_switch.modes_runtime import ModeState, RightHandState
from functions.mode_switch.morph_shape_control import LpShapePipelineState
from functions.open_close.morph_lp_plot import MORPH_PLANE_RADIUS_A, MORPH_PLANE_RADIUS_B, mode_epsilon_pair
from functions.open_close.morph_renderers import mapped_fixed_surface_points
from functions.open_close.morph_world import (
    ScaleConfig,
    fixed_morph_points,
    normalize_morph_points_at_hover,
    summarize_target_workspace,
)
from functions.runtime.live_target import LiveTargetState
from functions.swarm_motion.spacing_guard import closest_pair


def _bootstrap_initial_target(
    *,
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
    scale: ScaleConfig,
) -> np.ndarray:
    """Generate and normalize the initial morph target; log spacing diagnostics.

    Args:
        point_count: Number of fixed surface sample indices.
        radius_mm: Superellipsoid radius in millimeters.
        morph_mode: Morph mode index (1–5).
        open_alpha: Initial openness in ``[0, 1]``.
        shape_t: Optional left-hand shape blend parameter for ε tuning.
        scale: Workspace mm→m scaling configuration.

    Returns:
        Normalized Crazyflow target in sim meters, shape ``(point_count, 3)``.
    """
    points_mm = fixed_morph_points(point_count, radius_mm, morph_mode, open_alpha, shape_t)
    target = normalize_morph_points_at_hover(points_mm, scale)
    dist, i, j = closest_pair(target)
    print(
        f"Initial target: mode={morph_mode}, open={open_alpha:.2f}, n={point_count}, "
        f"radius_mm={radius_mm:.1f}"
    )
    xyz_min = np.min(points_mm, axis=0)
    xyz_max = np.max(points_mm, axis=0)
    print(
        "raw_mm_range="
        f"x[{xyz_min[0]:.1f},{xyz_max[0]:.1f}] "
        f"y[{xyz_min[1]:.1f},{xyz_max[1]:.1f}] "
        f"z[{xyz_min[2]:.1f},{xyz_max[2]:.1f}]"
    )
    print(f"Closest initial target spacing: pair=({i},{j}), dist={dist:.2f}m")
    return target


def make_initial_live_target(
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
    scale: ScaleConfig,
) -> LiveTargetState:
    """Bootstrap morph targets and wrap them in thread-safe live state.

    Args:
        point_count: Number of fixed surface sample indices.
        radius_mm: Superellipsoid radius in millimeters.
        morph_mode: Initial morph mode index (1–5).
        open_alpha: Initial openness in ``[0, 1]``.
        shape_t: Optional left-hand shape blend parameter for ε tuning.
        scale: Workspace mm→m scaling configuration.

    Returns:
        ``LiveTargetState`` seeded with the normalized initial target and mode/open.
    """
    target = _bootstrap_initial_target(
        point_count=point_count,
        radius_mm=radius_mm,
        morph_mode=morph_mode,
        open_alpha=open_alpha,
        shape_t=shape_t,
        scale=scale,
    )
    state = LiveTargetState(target)
    state.mode = int(morph_mode)
    state.open_alpha = float(open_alpha)
    r_xy, xyz_min_m, xyz_max_m = summarize_target_workspace(target)
    print(
        "target_range_m="
        f"x[{xyz_min_m[0]:.2f},{xyz_max_m[0]:.2f}] "
        f"y[{xyz_min_m[1]:.2f},{xyz_max_m[1]:.2f}] "
        f"z[{xyz_min_m[2]:.2f},{xyz_max_m[2]:.2f}] "
        f"xy_radius_max={r_xy:.2f}"
    )
    return state


def update_live_target_from_state(
    *,
    live_target: LiveTargetState,
    mode_state: ModeState,
    right_state: RightHandState,
    lp_shape: LpShapePipelineState,
    scale: ScaleConfig,
    radius_mm: float,
    open_out: float | None,
) -> None:
    """Refresh the live morph target from debounced mode/open hand state.

    Skips the update when the normalized XY extent is too small (degenerate
    or unstable morph layout).

    Args:
        live_target: Shared target state updated in place when layout is valid.
        mode_state: Debounced morph mode from the gesture pipeline.
        right_state: Right-hand openness pipeline state.
        lp_shape: Left-hand shape-t EMA for ε pair selection.
        scale: Workspace mm→m scaling configuration.
        radius_mm: Superellipsoid radius in millimeters.
        open_out: Override openness; falls back to right-hand or previous target.
    """
    open_v = float(
        open_out
        if open_out is not None
        else (
            float(right_state.last_open_out)
            if right_state.last_open_out is not None
            else float(live_target.open_alpha)
        )
    )
    epsilon1, epsilon2 = mode_epsilon_pair(int(mode_state.morph_mode), lp_shape.left_shape_t_ema)
    points_mm = mapped_fixed_surface_points(
        radius=float(radius_mm),
        open_alpha=open_v,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        plane_radius_a=MORPH_PLANE_RADIUS_A,
        plane_radius_b=MORPH_PLANE_RADIUS_B,
        morph_mode=int(mode_state.morph_mode),
    )
    target = normalize_morph_points_at_hover(points_mm, scale)
    xy_r_max, _xyz_min, _xyz_max = summarize_target_workspace(target)
    if not np.isfinite(xy_r_max) or xy_r_max < 0.35:
        return
    live_target.set(target, mode=int(mode_state.morph_mode), open_alpha=open_v)
