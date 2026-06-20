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
from functions.runtime.online_defaults import _DEFAULT_MIN_SEPARATION_M
from functions.swarm_motion.spacing_guard import closest_pair


def _bootstrap_initial_target(
    *,
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
    scale: ScaleConfig,
    min_separation_m: float,
) -> np.ndarray:
    del min_separation_m
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
    target = _bootstrap_initial_target(
        point_count=point_count,
        radius_mm=radius_mm,
        morph_mode=morph_mode,
        open_alpha=open_alpha,
        shape_t=shape_t,
        scale=scale,
        min_separation_m=_DEFAULT_MIN_SEPARATION_M,
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
    min_separation_m: float = _DEFAULT_MIN_SEPARATION_M,
) -> None:
    del min_separation_m
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
