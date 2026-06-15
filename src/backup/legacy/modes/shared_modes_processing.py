"""Compatibility shim to `shared.mode_switch.modes_runtime`."""

from functions.mode_switch.modes_runtime import (
    draw_bottom_status,
    process_left_mode,
    process_right_open,
    update_hud_cache,
    update_snap_visual_state_for_modes,
)

# Backward-compatible name used by older callers.
update_snap_visual_state = update_snap_visual_state_for_modes

