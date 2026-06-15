"""Matplotlib 3D plot helpers for online control."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from functions.swarm_motion.left_hand_swarm_pose import R_to_rotvec
from functions.mode_switch.morph_shape_control import LpShapePipelineState
from functions.display_sim.orbbec_hand import (
    HAND_3D_SOURCE_MP,
    HAND_FRAME_SCALED,
    update_3d_plot,
)

def init_3d_plot(plot_every_n: int, title: str):
    """Create morph 3D figure/axes only when enabled."""
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
    if fig is None:
        return
    plt.ioff()
    plt.close(fig)


def refresh_3d_plot_nonblocking(fig) -> None:
    """Refresh matplotlib canvas without blocking in plt.pause()."""
    if fig is None:
        return
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        pass

def _clear_formation_rigid_debug(ax_topo) -> None:
    for art in getattr(ax_topo, "_iso_swarm_debug_artists", None) or []:
        try:
            art.remove()
        except Exception:
            pass
    ax_topo._iso_swarm_debug_artists = []
    tx = getattr(ax_topo, "_iso_swarm_debug_text", None)
    if tx is not None:
        try:
            tx.remove()
        except Exception:
            pass
    ax_topo._iso_swarm_debug_text = None


def draw_formation_rigid_debug_on_topo(
    ax_topo,
    p_before_m: np.ndarray,
    p_after_m: np.ndarray,
    *,
    off_m: np.ndarray | None = None,
    R_pose: np.ndarray | None = None,
) -> None:
    """Overlay morph-only vs L-move targets on the topo axis (scaled to ~mm span for visibility).

    Blue dots: morph-only targets before left-hand rigid; magenta: after ``apply_rigid_to_targets``.
    Used only when ``--formation-rigid-3d-debug`` is passed (default: overlay off).
    """
    _clear_formation_rigid_debug(ax_topo)
    pb = np.asarray(p_before_m, dtype=np.float64)
    pa = np.asarray(p_after_m, dtype=np.float64)
    if pb.ndim != 2 or pa.ndim != 2 or pb.shape != pa.shape or pb.shape[1] < 3:
        return
    c = np.mean(pb[:, :3], axis=0)
    d0 = pb[:, :3] - c
    d1 = pa[:, :3] - c
    r = max(
        float(np.max(np.linalg.norm(d0, axis=1))),
        float(np.max(np.linalg.norm(d1, axis=1))),
        0.12,
    )
    s = 85.0 / r
    q0 = d0 * s
    q1 = d1 * s
    art0 = ax_topo.scatter(
        q0[:, 0],
        q0[:, 1],
        q0[:, 2],
        c="tab:blue",
        s=22,
        alpha=0.88,
        depthshade=False,
        label="morph cmd",
    )
    art1 = ax_topo.scatter(
        q1[:, 0],
        q1[:, 1],
        q1[:, 2],
        c="tab:magenta",
        s=40,
        alpha=0.92,
        marker="^",
        depthshade=False,
        edgecolors="k",
        linewidths=0.45,
        label="+ L-move",
    )
    ax_topo._iso_swarm_debug_artists = [art0, art1]
    if off_m is not None and R_pose is not None:
        offv = np.asarray(off_m, dtype=np.float64).ravel()
        R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
        rv = R_to_rotvec(R)
        ang = float(np.linalg.norm(rv))
        if ang > 1e-8:
            ax = rv / ang
            rot_s = f"rot≈{float(np.degrees(ang)):+.1f}°  n̂≈({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"
        else:
            rot_s = "rot≈0°"
        t = (
            f"L-swarm: |Δ|={float(np.linalg.norm(offv)):.3f}m  "
            f"Δ=({offv[0]:+.2f},{offv[1]:+.2f},{offv[2]:+.2f})  {rot_s}"
        )
        try:
            tx = ax_topo.text2D(
                0.02,
                0.98,
                t,
                transform=ax_topo.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="tab:purple",
            )
            ax_topo._iso_swarm_debug_text = tx
        except Exception:
            ax_topo._iso_swarm_debug_text = None


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
    """Shared 3D update config aligned with Orbbec runtime."""
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
