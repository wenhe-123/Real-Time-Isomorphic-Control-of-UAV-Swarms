"""Compatibility shim to `shared.modes_runtime`."""

from shared.modes_runtime import (
    draw_bottom_status,
    process_left_mode,
    process_right_open,
    update_hud_cache,
    update_snap_visual_state_for_modes,
)

# Backward-compatible name used by older callers.
update_snap_visual_state = update_snap_visual_state_for_modes

