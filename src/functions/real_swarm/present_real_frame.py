"""Orbbec HUD + real Crazyflie setpoints (no MuJoCo)."""

from __future__ import annotations

import numpy as np

from debug.online_control_debug import draw_drone_target_debug_hud
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
            filter_src=filt.filter_src,
            cmd_target=filt.cmd_target,
            sim=None,
        )
        if real_pos is not None:
            print(
                f"[debug real_pos] frame={frame_idx} "
                f"centroid_z={float(np.mean(real_pos[:, 2])):.3f}m",
                flush=True,
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
            prearm_climb_enabled=bool(boot.prearm_climb_enabled),
        )
    _sec("real_cmd")

    _maybe_print_center_trace(inp)
    _draw_online_hud_overlay(inp, label="REAL", color=(0, 200, 255))
    _sec("overlay_hud")
