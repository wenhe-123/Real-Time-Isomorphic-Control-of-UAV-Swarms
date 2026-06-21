"""LED, trail, sim render, HUD overlay for one online control frame."""

from __future__ import annotations

import cv2
import numpy as np

from crazyflow.sim.visualize import draw_line
from debug.online_control_debug import (
    draw_drone_target_debug_hud,
    print_center_trace,
    print_drone_position_debug,
)
from functions.display_sim.crazyflow_render import render_sim, step_sim_to_cmd
from functions.display_sim.morph_led_materials import apply_morph_led_theme
from functions.display_sim.online_present_input import PresentFrameInput


def _draw_online_hud_overlay(inp: PresentFrameInput, *, label: str, color: tuple[int, int, int]) -> None:
    boot = inp.boot
    gest = inp.gest
    left_pose_state = boot.left_pose_state
    if left_pose_state.enabled:
        if left_pose_state.is_unwinding():
            pose_hint = f" | L-move:restore{inp.left_pose_dbg}"
        elif boot.left_pose_runtime_armed:
            pose_hint = f" | L-move:ON{inp.left_pose_dbg}"
        else:
            pose_hint = " | L-move:[0]"
    else:
        pose_hint = ""
    ax_hint = f" | {boot.axswarm_rt.status_line()}"
    if boot.gesture_control_enabled_box[0]:
        phase = "ARMED (Space disarm)"
    elif boot.prearm_phase_box[0] == "formation":
        phase = "HOVER FORM - press 1 shrink vertical"
    elif boot.prearm_phase_box[0] == "vertical":
        if boot.prearm_vertical_leg_box[0] == "descend":
            phase = "VERT DESC - press 1 ground"
        else:
            phase = "VERT CLIMB - press 1 formation"
    else:
        phase = "GROUND - press 1 takeoff"
    cv2.putText(
        inp.frame,
        f"{label} {phase} "
        f"M{boot.mode_state.morph_mode} raw:{gest.mode_raw} "
        f"open:{gest.open_out if gest.open_out is not None else '-'} "
        f"tier:{gest.tier_count if gest.tier_count >= 0 else '-'}{pose_hint}{ax_hint}",
        (16, inp.frame.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
        cv2.LINE_AA,
    )


def _maybe_print_center_trace(inp: PresentFrameInput) -> None:
    if not bool(inp.cfg.center_trace):
        return
    print_center_trace(
        elapsed=float(inp.elapsed),
        frame_idx=inp.boot.frame_idx,
        center_trace_every=inp.cfg.center_trace_every,
        center_trace_prev=inp.boot.center_trace_prev,
        raw_target=inp.raw_target,
        cmd_target=inp.filt.cmd_target,
        left_pose_state=inp.boot.left_pose_state,
    )


def present_online_frame(inp: PresentFrameInput) -> bool:
    """Draw debug/LED/trail/sim/HUD. Returns updated render_enabled."""
    boot = inp.boot
    cfg = inp.cfg
    gest = inp.gest
    filt = inp.filt
    frame_idx = boot.frame_idx
    render_enabled = boot.render_enabled

    def _sec(name: str) -> None:
        if inp.section is not None:
            inp.section(name)

    if cfg.debug_drone_targets_every > 0 and (frame_idx % cfg.debug_drone_targets_every) == 0:
        draw_drone_target_debug_hud(
            inp.frame,
            frame_idx=frame_idx,
            mode_state=boot.mode_state,
            right_state=boot.right_state,
            open_out=gest.open_out,
            min_separation_m=float(cfg.min_separation_m),
            raw_target=inp.raw_target,
            axswarm_input=filt.axswarm_input,
            cmd_target=filt.cmd_target,
            sim=boot.sim,
        )
    if frame_idx % cfg.led_every_n == 0 and boot.sim is not None:
        apply_morph_led_theme(boot.sim, int(boot.mode_state.morph_mode))
    _sec("led")
    if cfg.trail_every_n > 0:
        boot.pos_buffer.append(np.asarray(filt.cmd_target, dtype=np.float64))
    if (
        render_enabled
        and cfg.trail_every_n > 0
        and len(boot.pos_buffer) > 1
        and (frame_idx % cfg.trail_every_n) == 0
        and boot.sim is not None
    ):
        lines = np.asarray(boot.pos_buffer)
        for d in range(boot.n_drones):
            try:
                draw_line(
                    boot.sim,
                    lines[:, d, :],
                    rgba=boot.trail_rgba[d],
                    start_size=0.5,
                    end_size=2.0,
                )
            except Exception as exc:
                render_enabled = False
                print(f"[WARN] Disabled Crazyflow trail drawing after render error: {exc}")
                break
    _sec("trail")
    if boot.sim is not None:
        try:
            step_sim_to_cmd(
                boot.sim,
                np.asarray(filt.cmd_target, dtype=np.float64),
                outer_fps=int(cfg.fps),
                max_substeps=int(cfg.max_sim_substeps),
                velocities=np.asarray(filt.cmd_velocity, dtype=np.float64),
            )
        except Exception as exc:
            render_enabled = False
            print(f"[WARN] Disabled Crazyflow sim step after error: {exc}")
    _sec("sim_step")
    if cfg.debug_drone_pos_every > 0 and (frame_idx % cfg.debug_drone_pos_every) == 0:
        _hold_z: float | None = None
        if not boot.gesture_control_enabled:
            if boot.prearm_phase == "ground":
                _hold_z = float(boot.ground_z)
            elif boot.prearm_phase == "vertical" and boot.prearm_vertical_leg == "climb":
                _hold_z = float(boot.prearm_takeoff_z)
        print_drone_position_debug(
            frame_idx=frame_idx,
            cmd_target=filt.cmd_target,
            pre_axswarm=filt.axswarm_input,
            sim=boot.sim,
            raw_target=inp.raw_target,
            hold_z=_hold_z,
            axswarm_status=boot.axswarm_rt.status_line(),
        )
    if (
        render_enabled
        and cfg.sim_render_every > 0
        and (frame_idx % cfg.sim_render_every) == 0
        and boot.sim is not None
    ):
        try:
            render_sim(boot.sim)
        except Exception as exc:
            render_enabled = False
            print(f"[WARN] Disabled Crazyflow rendering after render error: {exc}")
    _sec("sim_render")
    _maybe_print_center_trace(inp)
    _draw_online_hud_overlay(inp, label="ONLINE", color=(0, 255, 0))
    _sec("overlay_hud")
    return render_enabled
