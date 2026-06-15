"""Re-exports for backup Orbbec/webcam demos (shared webcam + orbbec constants)."""

from functions.display_sim.orbbec_hand import (
    DEPTH_FUSION_WEIGHT,
    HAND_3D_SOURCE_FUSED,
    HAND_3D_SOURCE_MP,
    HAND_FRAME_METRIC_MM,
    HAND_FRAME_PALM_PLANE,
    HAND_FRAME_SCALED,
    OPEN_REMAP_HI,
    OPEN_REMAP_LO,
)
from functions.mode_switch.webcam_mode_defaults import (
    ENABLE_3D_PLOT,
    HUD_METRIC_STEP,
    HUD_OPEN_STEP,
    HUD_UPDATE_EVERY_N_FRAMES,
    MODE_DEBOUNCE_FRAMES,
    MORPH_AXIS_LIM_MM,
    NORM_AXIS_HALFLIM,
    PLOT_EVERY_N_FRAMES,
    PLANE_SNAP_OFF,
    PLANE_SNAP_ON,
    SNAP_HOLD_AFTER_RELEASE_FRAMES,
    SNAP_SHOW_AFTER_FRAMES,
    SPHERE_SNAP_OFF,
    SPHERE_SNAP_ON,
    classify_mode_from_fingers,
)
