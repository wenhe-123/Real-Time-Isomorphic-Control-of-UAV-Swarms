"""Matplotlib 3D plot update for one online control frame."""

from __future__ import annotations

import numpy as np

from functions.display_sim.gesture_report_debug import (
    ReportDebugPanels,
    update_report_debug_figures,
)
from functions.display_sim.online_plot import (
    _clear_formation_rigid_debug,
    draw_formation_rigid_debug_on_topo,
    refresh_3d_plot_nonblocking,
    update_online_3d_plot,
)
from functions.mode_switch.online_frame_gesture import GestureFrameResult
from functions.mode_switch.webcam_mode_defaults import analyze_hand_topology
from functions.runtime.online_boot import OnlineBoot
from functions.runtime.online_runtime_config import OnlineRuntimeConfig


def _pick_hand_for_plot(
    hands_3d: list | None,
    hands_3d_all: list,
    idx_l: int | None,
) -> list | None:
    if hands_3d:
        return hands_3d
    if idx_l is not None and 0 <= idx_l < len(hands_3d_all):
        return [hands_3d_all[idx_l]]
    if hands_3d_all:
        return [hands_3d_all[0]]
    return None


def update_online_plot_frame(
    *,
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    gest: GestureFrameResult,
    raw_target: np.ndarray,
    morph_targets_before_left_m: np.ndarray,
    left_swarm_R: np.ndarray | None,
    left_swarm_off: np.ndarray | None,
) -> bool:
    """Update 3D plot if due; return updated plot_enabled (False if disabled after error)."""
    plot_enabled = boot.plot_enabled
    plot_every_n = cfg.plot_every_n
    frame_idx = boot.frame_idx
    fig = boot.fig
    ax_hand = boot.ax_hand
    ax_topo = boot.ax_topo
    hands_3d = gest.hands_3d
    hands_3d_all = gest.hands_3d_all
    idx_l = gest.idx_l
    mode_state = boot.mode_state
    open_out = gest.open_out
    lp_shape = boot.lp_shape
    formation_rigid_3d_debug = cfg.formation_rigid_3d_debug
    left_pose_state = boot.left_pose_state
    left_pose_runtime_armed = boot.left_pose_runtime_armed
    report_panels = boot.extras.get("report_panels")
    report_debug_figs = boot.extras.get("report_debug_figs")
    pts_l_pose_mm = gest.pts_l_pose_mm

    if not plot_enabled or plot_every_n <= 0:
        return plot_enabled
    if (frame_idx % plot_every_n) != 0:
        return plot_enabled

    panels = report_panels or ReportDebugPanels()
    hands_3d_plot = _pick_hand_for_plot(hands_3d, hands_3d_all, idx_l)
    hand_points = hands_3d_plot[0] if hands_3d_plot else None
    morph_enabled = ax_topo is not None and (panels.morph or (fig is not None and not panels.any_enabled()))

    if morph_enabled:
        try:
            update_online_3d_plot(
                ax_hand=ax_hand,
                ax_topo=ax_topo,
                hands_3d=hands_3d_plot or [],
                morph_mode=mode_state.morph_mode,
                open_out=open_out,
                lp_shape=lp_shape,
                show_morph_refs=bool(panels.morph),
            )
            if (
                formation_rigid_3d_debug
                and left_pose_state.enabled
                and (left_pose_runtime_armed or left_pose_state.is_unwinding())
                and left_swarm_R is not None
            ):
                draw_formation_rigid_debug_on_topo(
                    ax_topo,
                    morph_targets_before_left_m,
                    raw_target,
                    off_m=left_swarm_off,
                    R_pose=left_swarm_R,
                )
            else:
                _clear_formation_rigid_debug(ax_topo)
            if fig is not None:
                refresh_3d_plot_nonblocking(fig)
        except Exception as exc:
            print(f"[WARN] Disabled Matplotlib 3D updates after plotting error: {exc}")
            return False

    if report_debug_figs is not None and panels.extra_enabled():
        try:
            analysis = analyze_hand_topology(hand_points) if hand_points is not None else None
            update_report_debug_figures(
                report_debug_figs,
                hand_points=hand_points,
                analysis=analysis,
                open_out=open_out,
                pts_l_pose_mm=pts_l_pose_mm,
                left_pose_state=left_pose_state,
            )
        except Exception as exc:
            print(f"[WARN] Report debug figure update failed (main plot still active): {exc}")

    return plot_enabled
