"""Superellipsoid morph 3D plot for webcam / Orbbec / dual: maps mode→(ε₁,ε₂), draws surface, wires ``update_3d_plot_modes``.

Pipelines pass only topology analysis, coordinate frame, and axis limits; ε-ranges and mesh params live here.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from shared.hand_constants import HAND_CONNECTIONS
from shared.morph_renderers import draw_superellipsoid_morph_canonical
from shared.modes_runtime import update_3d_plot_modes
from shared.topology_utils import clamp01

# Mode → base (ε₁, ε₂). Barr form: latitude ε₁, xy longitude ε₂.
# Tuned for readable silhouettes: sphere=unit; cylinder=round XY + flat meridian; cube=low both;
# square column=round profile + square XY; mode 5 asymmetric high ε₁ with ε₂=1.
MODE_EPSILON_BASE = {
    1: (1.0, 1.0),  # 标准球
    2: (0.10, 1.0),  # 圆柱：水平截面圆，子午向压扁
    3: (0.10, 0.10),  # 方体（超二次 “cube”）
    4: (1.0, 0.10),  # 方柱：侧视圆、俯视方
    5: (2.0, 1.0),  # 非对称超高次（ε₁>1 与 ε₂=1 组合）
}

_EPS_LO = 0.06
_EPS_HI = 2.5


def _clamp_eps_pair(e1: float, e2: float) -> Tuple[float, float]:
    return (
        float(max(_EPS_LO, min(_EPS_HI, float(e1)))),
        float(max(_EPS_LO, min(_EPS_HI, float(e2)))),
    )


def mode_epsilon_pair(morph_mode: int, shape_t: Optional[float]) -> Tuple[float, float]:
    mid = MODE_EPSILON_BASE.get(int(morph_mode), MODE_EPSILON_BASE[1])
    t = 0.5 if shape_t is None else float(clamp01(float(shape_t)))
    lo1, hi1 = mid[0] * 0.92, mid[0] * 1.08
    lo2, hi2 = mid[1] * 0.92, mid[1] * 1.08
    e1 = lo1 + t * (hi1 - lo1)
    e2 = lo2 + t * (hi2 - lo2)
    return _clamp_eps_pair(e1, e2)


MORPH_LP_MESH_ETA = 40
MORPH_LP_MESH_OMEGA = 52

MORPH_PLANE_RADIUS_A = 0.90
MORPH_PLANE_RADIUS_B = 0.95
MORPH_PLANE_GRID_N = 5
MORPH_PLANE_GRID_SPHERE_ALPHA = 0.22

MORPH_SAMPLE_SCATTER_S = 22
MORPH_SAMPLE_ALPHA = 0.65


def update_3d_plot_lp(
    ax_hand,
    ax_topo,
    hands_3d,
    *,
    morph_mode: int,
    morph_alpha_smoothed=None,
    control_label: str = "",
    analyze_hand_topology_fn: Callable[..., Any],
    clamp01_fn: Callable[[float], float],
    shape_normalized: bool = False,
    hand_frame: str,
    hand_3d_source: str,
    hand_frame_palm_plane: str,
    norm_axis_halflim: float,
    morph_axis_lim_mm: float,
    hand_connections: Sequence[Sequence[int]] = HAND_CONNECTIONS,
    mode_shape_t: Optional[float] = None,
    epsilon_pair_display: Optional[Tuple[float, float]] = None,
    lp_show_refs: bool = True,
    mesh_n_eta: int = MORPH_LP_MESH_ETA,
    mesh_n_omega: int = MORPH_LP_MESH_OMEGA,
) -> List[Any]:
    t_shape = 0.5 if mode_shape_t is None else float(clamp01_fn(float(mode_shape_t)))
    e1, e2 = (
        mode_epsilon_pair(morph_mode, t_shape)
        if epsilon_pair_display is None
        else _clamp_eps_pair(epsilon_pair_display[0], epsilon_pair_display[1])
    )

    def _draw_se(ax, radius, open_alpha, show_refs=True):
        return draw_superellipsoid_morph_canonical(
            ax,
            radius,
            open_alpha,
            epsilon1=e1,
            epsilon2=e2,
            plane_radius_a=MORPH_PLANE_RADIUS_A,
            plane_radius_b=MORPH_PLANE_RADIUS_B,
            plane_grid_n=MORPH_PLANE_GRID_N,
            sample_scatter_s=MORPH_SAMPLE_SCATTER_S,
            sample_alpha=MORPH_SAMPLE_ALPHA,
            show_refs=show_refs and lp_show_refs,
            mesh_n_eta=mesh_n_eta,
            mesh_n_omega=mesh_n_omega,
            morph_mode=int(morph_mode),
        )

    return update_3d_plot_modes(
        ax_hand,
        ax_topo,
        hands_3d,
        morph_mode=morph_mode,
        morph_alpha_smoothed=morph_alpha_smoothed,
        control_label=f"{control_label} | ε1={e1:.2f} ε2={e2:.2f}",
        shape_normalized=shape_normalized,
        hand_frame=hand_frame,
        hand_3d_source=hand_3d_source,
        hand_frame_palm_plane=hand_frame_palm_plane,
        hand_connections=hand_connections,
        norm_axis_halflim=norm_axis_halflim,
        morph_axis_lim_mm=morph_axis_lim_mm,
        analyze_hand_topology_fn=analyze_hand_topology_fn,
        draw_mode1_fn=_draw_se,
        draw_mode2_fn=_draw_se,
        draw_mode3_fn=_draw_se,
        clamp01_fn=clamp01_fn,
    )
