"""Matplotlib 3D plot update for one online control frame."""

from __future__ import annotations

import numpy as np

from debug.gesture_report_debug import (
    ReportDebugPanels,
    update_report_debug_figures,
)
from debug.formation_rigid_plot_debug import (
    _clear_formation_rigid_debug,
    draw_formation_rigid_debug_on_topo,
)
from functions.display_sim.online_plot import (
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
    """Select the hand list to plot, preferring filtered left-hand data.

    Args:
        hands_3d: Pre-filtered hand list for plotting, or ``None``.
        hands_3d_all: All detected hands as 21×3 mm arrays.
        idx_l: Index of the left hand in ``hands_3d_all``, or ``None``.

    Returns:
        Single-element hand list for plotting, or ``None`` when no hands are available.
    """
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
    """Update the 3D matplotlib plot when due for the current online control frame.

    Args:
        boot: Online runtime boot state (axes, figure, frame index, pose state).
        cfg: Online runtime configuration (plot interval, debug flags).
        gest: Gesture frame result with hand 3D points and left-hand index.
        raw_target: Morph targets after left-hand rigid transform, shape ``(N, 3)``.
        morph_targets_before_left_m: Morph-only targets before L-move, shape ``(N, 3)``.
        left_swarm_R: Left-hand swarm rotation matrix, shape ``(3, 3)``, or ``None``.
        left_swarm_off: Left-hand swarm translation in meters, shape ``(3,)``, or ``None``.

    Returns:
        Updated ``plot_enabled`` flag; ``False`` when plotting is disabled after an error.
    """
    if not boot.plot_enabled or cfg.plot_every_n <= 0:
        return boot.plot_enabled
    if (boot.frame_idx % cfg.plot_every_n) != 0:
        return boot.plot_enabled

    report_panels = boot.extras.get("report_panels")
    report_debug_figs = boot.extras.get("report_debug_figs")
    panels = report_panels or ReportDebugPanels()
    hands_3d_plot = _pick_hand_for_plot(gest.hands_3d, gest.hands_3d_all, gest.idx_l)
    hand_points = hands_3d_plot[0] if hands_3d_plot else None
    morph_enabled = boot.ax_topo is not None and (
        panels.morph or (boot.fig is not None and not panels.any_enabled())
    )

    if morph_enabled:
        try:
            update_online_3d_plot(
                ax_hand=boot.ax_hand,
                ax_topo=boot.ax_topo,
                hands_3d=hands_3d_plot or [],
                morph_mode=boot.mode_state.morph_mode,
                open_out=gest.open_out,
                lp_shape=boot.lp_shape,
                show_morph_refs=bool(panels.morph),
            )
            if (
                cfg.formation_rigid_3d_debug
                and boot.left_pose_state.enabled
                and (boot.left_pose_runtime_armed or boot.left_pose_state.is_unwinding())
                and left_swarm_R is not None
            ):
                draw_formation_rigid_debug_on_topo(
                    boot.ax_topo,
                    morph_targets_before_left_m,
                    raw_target,
                    off_m=left_swarm_off,
                    R_pose=left_swarm_R,
                )
            else:
                _clear_formation_rigid_debug(boot.ax_topo)
            if boot.fig is not None:
                refresh_3d_plot_nonblocking(boot.fig)
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
                open_out=gest.open_out,
                pts_l_pose_mm=gest.pts_l_pose_mm,
                left_pose_state=boot.left_pose_state,
            )
        except Exception as exc:
            print(f"[WARN] Report debug figure update failed (main plot still active): {exc}")

    return boot.plot_enabled
