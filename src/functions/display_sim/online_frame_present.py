"""LED, trail, sim render, HUD overlay for one online control frame."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from debug.online_control_debug import draw_drone_target_debug_hud, print_center_trace
from functions.display_sim.crazyflow_render import render_targets
from functions.display_sim.morph_led_materials import apply_morph_led_theme
from functions.swarm_motion.swarm_workspace_box import SwarmWorkspaceBox, draw_swarm_workspace_box_in_sim


def present_online_frame(
    *,
    frame: np.ndarray,
    sim: Sim,
    cmd_target: np.ndarray,
    safe_target: np.ndarray,
    raw_target: np.ndarray,
    filter_src: np.ndarray,
    mode_state: Any,
    right_state: Any,
    left_pose_state: Any,
    left_pose_dbg: str,
    left_pose_runtime_armed: bool,
    axswarm_rt: Any,
    swarm_workspace: SwarmWorkspaceBox,
    gesture_control_enabled: bool,
    gesture_control_enabled_box: list,
    mode_raw: Any,
    open_out: float | None,
    tier_count: int,
    frame_idx: int,
    elapsed: float,
    center_trace: bool,
    center_trace_every: int,
    center_trace_prev: dict[str, float | None],
    debug_drone_targets_every: int,
    min_separation_m: float,
    led_every_n: int,
    trail_every_n: int,
    sim_render_every: int,
    n_drones: int,
    pos_buffer: deque,
    trail_rgba: list,
    render_enabled: bool,
    section: Callable[[str], None] | None = None,
) -> bool:
    """Draw debug/LED/trail/sim/HUD. Returns updated render_enabled."""

    def _sec(name: str) -> None:
        if section is not None:
            section(name)

    if debug_drone_targets_every > 0 and (frame_idx % debug_drone_targets_every) == 0:
        draw_drone_target_debug_hud(
            frame,
            frame_idx=frame_idx,
            mode_state=mode_state,
            right_state=right_state,
            open_out=open_out,
            min_separation_m=float(min_separation_m),
            raw_target=raw_target,
            filter_src=filter_src,
            cmd_target=cmd_target,
            sim=sim,
        )
    if frame_idx % led_every_n == 0:
        apply_morph_led_theme(sim, int(mode_state.morph_mode))
    _sec("led")
    if trail_every_n > 0:
        pos_buffer.append(np.asarray(cmd_target, dtype=np.float64))
    if (
        render_enabled
        and trail_every_n > 0
        and len(pos_buffer) > 1
        and (frame_idx % trail_every_n) == 0
    ):
        lines = np.asarray(pos_buffer)
        for d in range(n_drones):
            try:
                draw_line(
                    sim,
                    lines[:, d, :],
                    rgba=trail_rgba[d],
                    start_size=0.5,
                    end_size=2.0,
                )
            except Exception as exc:
                render_enabled = False
                print(f"[WARN] Disabled Crazyflow trail drawing after render error: {exc}")
                break
    _sec("trail")
    if render_enabled and sim_render_every > 0 and (frame_idx % sim_render_every) == 0:
        try:
            if swarm_workspace.enabled and swarm_workspace.armed:
                draw_swarm_workspace_box_in_sim(sim, swarm_workspace)
            render_targets(sim, np.asarray(cmd_target, dtype=np.float64))
        except Exception as exc:
            render_enabled = False
            print(f"[WARN] Disabled Crazyflow rendering after render error: {exc}")
    _sec("sim_render")
    if bool(center_trace):
        print_center_trace(
            elapsed=float(elapsed),
            frame_idx=frame_idx,
            center_trace_every=center_trace_every,
            center_trace_prev=center_trace_prev,
            raw_target=raw_target,
            cmd_target=cmd_target,
            safe_target=safe_target,
            left_pose_state=left_pose_state,
            swarm_workspace=swarm_workspace,
        )
    if left_pose_state.enabled:
        if left_pose_state.is_unwinding():
            pose_hint = f" | L-move:restore{left_pose_dbg}"
        elif left_pose_runtime_armed:
            pose_hint = f" | L-move:ON{left_pose_dbg}"
        else:
            pose_hint = " | L-move:[0]"
    else:
        pose_hint = ""
    ax_hint = f" | {axswarm_rt.status_line()}" if axswarm_rt is not None else ""
    cv2.putText(
        frame,
        f"ONLINE {'ARMED' if gesture_control_enabled_box[0] else 'HOLD DEFAULT - press SPACE'} "
        f"M{mode_state.morph_mode} raw:{mode_raw} open:{open_out if open_out is not None else '-'} "
        f"tier:{tier_count if tier_count >= 0 else '-'}{pose_hint}{ax_hint}",
        (16, frame.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    _sec("overlay_hud")
    return render_enabled
