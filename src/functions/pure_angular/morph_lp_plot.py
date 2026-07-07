"""Superellipsoid morph 3D plot for webcam / Orbbec / dual: maps mode→(ε₁,ε₂), draws surface, wires ``update_3d_plot_modes``.

Pipelines pass only topology analysis, coordinate frame, and axis limits; ε-ranges and mesh params live here.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from functions.mode_switch.hand_constants import HAND_CONNECTIONS
from functions.open_close.morph_renderers import draw_superellipsoid_morph_canonical
from functions.mode_switch.modes_runtime import update_3d_plot_modes
from functions.mode_switch.topology_utils import clamp01

# Mode → base (ε₁, ε₂). Barr form: latitude ε₁, xy longitude ε₂.
# Tuned for readable silhouettes: sphere=unit; cylinder=round XY + flat meridian; cube=low both;
# square column=round profile + square XY; mode 5 asymmetric high ε₁ with ε₂=1.
MODE_EPSILON_BASE = {
    1: (1.0, 1.0),
    2: (0.10, 1.0),
    3: (0.10, 0.10),
    4: (1.0, 0.10),
    5: (2.0, 1.0),
}

_EPS_LO = 0.06
_EPS_HI = 2.5


def _clamp_eps_pair(e1: float, e2: float) -> Tuple[float, float]:
    return (
        float(max(_EPS_LO, min(_EPS_HI, float(e1)))),
        float(max(_EPS_LO, min(_EPS_HI, float(e2)))),
    )


def mode_epsilon_pair(morph_mode: int, shape_t: Optional[float]) -> Tuple[float, float]:
    """Map morph mode and shape blend to clamped superellipsoid ε pair.

    Args:
        morph_mode: Morph mode index (1–5).
        shape_t: Optional shape blend in ``[0, 1]``; defaults to ``0.5`` when
            ``None``.

    Returns:
        Tuple ``(epsilon1, epsilon2)`` clamped to ``[_EPS_LO, _EPS_HI]``.
    """
    mid = MODE_EPSILON_BASE.get(int(morph_mode), MODE_EPSILON_BASE[1])
    t = 0.5 if shape_t is None else float(clamp01(float(shape_t)))
    lo1, hi1 = mid[0] * 0.92, mid[0] * 1.08
    lo2, hi2 = mid[1] * 0.92, mid[1] * 1.08
    e1 = lo1 + t * (hi1 - lo1)
    e2 = lo2 + t * (hi2 - lo2)
    return _clamp_eps_pair(e1, e2)


MORPH_LP_MESH_ETA = 18
MORPH_LP_MESH_OMEGA = 24

MORPH_PLANE_RADIUS_A = 1.05  #scale of closed shape
MORPH_PLANE_RADIUS_B = 0.60  #scale of open shape
MORPH_PLANE_GRID_N = 5

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
    show_sample_ids: bool = False,
    mesh_n_eta: int = MORPH_LP_MESH_ETA,
    mesh_n_omega: int = MORPH_LP_MESH_OMEGA,
    topo_radius_override_mm: Optional[float] = None,
) -> List[Any]:
    """Update dual 3D Matplotlib panels for superellipsoid morph visualization.

    Resolves ε from mode/shape, draws the canonical superellipsoid surface, and
    delegates hand topology overlay to :func:`update_3d_plot_modes`.

    Args:
        ax_hand: Matplotlib 3D axis for hand landmarks.
        ax_topo: Matplotlib 3D axis for morph topology samples.
        hands_3d: Hand landmark arrays for overlay drawing.
        morph_mode: Active morph mode index (1–5).
        morph_alpha_smoothed: Smoothed morph alpha for topology coloring.
        control_label: Extra label text appended to the mode control string.
        analyze_hand_topology_fn: Callable that analyzes hand topology metrics.
        clamp01_fn: Callable clamping floats to ``[0, 1]``.
        shape_normalized: Whether hand shape coordinates are normalized.
        hand_frame: Hand coordinate frame label for plotting.
        hand_3d_source: Source tag for 3D hand data (depth/world/etc.).
        hand_frame_palm_plane: Palm-plane frame label for topology overlay.
        norm_axis_halflim: Half-limit for normalized hand axis display.
        morph_axis_lim_mm: Half-limit for morph axis display in mm.
        hand_connections: MediaPipe hand edge index pairs.
        mode_shape_t: Optional shape blend for ε resolution.
        epsilon_pair_display: Optional override ``(ε1, ε2)`` for display only.
        lp_show_refs: Draw reference geometry on the morph axis.
        show_sample_ids: Annotate fixed sample indices on scatter points.
        mesh_n_eta: Superellipsoid mesh latitude divisions.
        mesh_n_omega: Superellipsoid mesh longitude divisions.
        topo_radius_override_mm: Optional radius override for topology axis.

    Returns:
        List of Matplotlib artists returned by :func:`update_3d_plot_modes`.
    """
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
            show_sample_ids=show_sample_ids,
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
        topo_radius_override_mm=topo_radius_override_mm,
    )
