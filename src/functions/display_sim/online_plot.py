"""Matplotlib 3D plot helpers for online control."""

from __future__ import annotations

import matplotlib.pyplot as plt

from functions.mode_switch.morph_shape_control import LpShapePipelineState
from functions.display_sim.orbbec_hand import (
    HAND_3D_SOURCE_MP,
    HAND_FRAME_SCALED,
    update_3d_plot,
)

def init_3d_plot(plot_every_n: int, title: str):
    """Create the morph 3D matplotlib figure and axes when plotting is enabled.

    Args:
        plot_every_n: Plot update interval in frames; ``<= 0`` disables plotting.
        title: Window / figure title string.

    Returns:
        ``(plot_enabled, fig, ax_hand, ax_topo)`` where ``ax_hand`` is ``None`` and
        ``ax_topo`` holds the single 3D subplot when enabled.
    """
    plot_enabled = int(plot_every_n) > 0
    if not plot_enabled:
        return False, None, None, None
    plt.ion()
    fig = plt.figure(title)
    ax_hand = None
    ax_topo = fig.add_subplot(111, projection="3d")
    try:
        fig.tight_layout()
    except Exception:
        pass
    plt.show(block=False)
    return True, fig, ax_hand, ax_topo


def close_3d_plot(fig) -> None:
    """Close the matplotlib 3D figure and disable interactive mode.

    Args:
        fig: Matplotlib figure handle, or ``None``.

    Returns:
        None.
    """
    if fig is None:
        return
    plt.ioff()
    plt.close(fig)


def refresh_3d_plot_nonblocking(fig) -> None:
    """Refresh the matplotlib canvas without blocking in ``plt.pause()``.

    Args:
        fig: Matplotlib figure handle, or ``None``.

    Returns:
        None.
    """
    if fig is None:
        return
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        pass


def update_online_3d_plot(
    *,
    ax_hand,
    ax_topo,
    hands_3d,
    morph_mode: int,
    open_out: float | None,
    lp_shape: "LpShapePipelineState",
    topo_radius_override_mm: float | None = None,
    debug_report_viz: bool = False,
    show_morph_refs: bool | None = None,
):
    """Update the shared online 3D plot with Orbbec-aligned configuration.

    Args:
        ax_hand: Matplotlib 3D axis for the hand skeleton, or ``None``.
        ax_topo: Matplotlib 3D axis for morph topology.
        hands_3d: List of per-hand 21×3 mm keypoint arrays.
        morph_mode: Active morph mode index (1–5).
        open_out: Smoothed open-hand scalar for morph visualization.
        lp_shape: Left-hand shape pipeline state (shape_t EMA, epsilon pair).
        topo_radius_override_mm: Optional fixed topology axis radius in mm.
        debug_report_viz: Legacy flag; enables morph refs when ``show_morph_refs`` is ``None``.
        show_morph_refs: When ``True``, draw LP reference geometry on the topology axis.

    Returns:
        Return value from :func:`update_3d_plot` (typically ``None``).
    """
    refs = bool(show_morph_refs) if show_morph_refs is not None else bool(debug_report_viz)
    return update_3d_plot(
        ax_hand,
        ax_topo,
        hands_3d,
        morph_alpha_smoothed=open_out,
        morph_mode=morph_mode,
        mode_shape_t=lp_shape.left_shape_t_ema,
        epsilon_pair_display=lp_shape.epsilon_pair_display,
        shape_normalized=True,
        hand_frame=HAND_FRAME_SCALED,
        hand_3d_source=HAND_3D_SOURCE_MP,
        topo_radius_override_mm=topo_radius_override_mm,
        control_label="online open+p",
        lp_show_refs=refs,
    )
