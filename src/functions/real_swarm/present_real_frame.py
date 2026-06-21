"""Orbbec HUD + real Crazyflie setpoints (no MuJoCo)."""

from __future__ import annotations

from debug.online_control_debug import draw_drone_target_debug_hud, print_drone_position_debug
from functions.display_sim.online_frame_present import _draw_online_hud_overlay, _maybe_print_center_trace
from functions.display_sim.online_present_input import PresentFrameInput


def present_real_online_frame(inp: PresentFrameInput) -> None:
    """Draw HUD and stream ``cmd_target`` to physical drones."""
    boot = inp.boot
    cfg = inp.cfg
    gest = inp.gest
    filt = inp.filt
    frame_idx = boot.frame_idx

    def _sec(name: str) -> None:
        if inp.section is not None:
            inp.section(name)

    if cfg.debug_drone_targets_every > 0 and (frame_idx % cfg.debug_drone_targets_every) == 0:
        real_pos = boot.real_executor.get_positions_for_debug() if boot.real_executor else None
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
            sim=None,
        )
        if real_pos is not None:
            print_drone_position_debug(
                frame_idx=frame_idx,
                cmd_target=filt.cmd_target,
                real_pos=real_pos,
            )
    _sec("debug_hud")

    if boot.real_executor is not None:
        boot.real_executor.track_frame(
            filt.cmd_target,
            gesture_enabled=bool(boot.gesture_control_enabled),
            just_armed=bool(inp.just_gesture_armed),
            morph_mode=int(boot.mode_state.morph_mode),
            led_every_n=int(cfg.led_every_n),
            frame_idx=int(frame_idx),
            prearm_phase=str(boot.prearm_phase),
            prearm_vertical_leg=str(boot.prearm_vertical_leg),
            just_prearm_phase=bool(inp.just_prearm_phase),
            prearm_vertical_layout=boot.prearm_vertical_layout,
        )
    _sec("real_cmd")

    if cfg.debug_drone_pos_every > 0 and (frame_idx % cfg.debug_drone_pos_every) == 0:
        real_pos = boot.real_executor.get_positions_for_debug() if boot.real_executor else None
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
            real_pos=real_pos,
            raw_target=inp.raw_target,
            hold_z=_hold_z,
            axswarm_status=boot.axswarm_rt.status_line(),
        )

    _maybe_print_center_trace(inp)
    _draw_online_hud_overlay(inp, label="REAL", color=(0, 200, 255))
    _sec("overlay_hud")
