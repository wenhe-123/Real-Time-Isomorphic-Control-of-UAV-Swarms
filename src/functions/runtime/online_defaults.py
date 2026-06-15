"""Production defaults for online Orbbec control."""

from __future__ import annotations

# Matplotlib 3D: production off (shared); debug cadence in debug.pipeline_tuning.
_DEFAULT_LEFT_POSE_FRAME_VIZ_EVERY = 10
_DEFAULT_WEBCAM_ROT_STRIDE = 6
_TRAIL_DRAW_EVERY_FRAMES = 0
_LED_APPLY_EVERY_FRAMES = 3
_DEFAULT_SIM_RENDER_EVERY = 2
_ONLINE_ORBBEC_FPS = "30"
# draw_hand: MP world only by default; enable depth fusion with --debug-webcam-pipeline.
_ONLINE_MP_INPUT_SCALE = 0.5
# Run HandLandmarker every N camera frames; reuse last result on skip (depth pose still updates).
_ONLINE_MP_DETECT_EVERY = 2
_DEFAULT_ONLINE_IMSHOW_EVERY = 4
_DEFAULT_TARGET_ALPHA = 0.44
_DEFAULT_DRONE_MODEL = "cf21B_500"
_DEFAULT_MIN_SEPARATION_M = 0.32
# Lower than 1.2 so open≈1 plane shell fits the default 3.5 m workspace (see formation_max_extent_m).
_DEFAULT_MORPH_WORLD_SCALE = 0.55
# Morph mode debounce: see online_pipeline_defaults (production: 1 frame).
_ONLINE_MODE_ROT_FREEZE_LATCH = 18
# Left-hand MP visibility min before morph mode (M1..M5) may change; blocks occlusion false counts.
_DEFAULT_MODE_VIS_MIN = 0.55
_DEFAULT_OPEN_VIS_MIN = 0.55
# History length for MuJoCo trail polylines (smaller = shorter tail on screen).
_TRAIL_BUFFER_MAXLEN = 6
# Legacy cap for max physics substeps (axswarm yaml cap sync only; no sim.step).
_ONLINE_MAX_SIM_SUBSTEPS_PER_FRAME = 160
# EMA on morph targets before separation (0 = off). Keep online control direct by default.
_DEFAULT_RAW_TARGET_EMA = 0.0
# Max |Δposition| per drone per physics substep toward blended target (m); 0 = no cap.
_DEFAULT_MAX_TARGET_STEP_M = 0.038
# When |Δopen| >= this after a frame's gesture update, restart post-SPACE slow blend (0=off).
_DEFAULT_OPEN_JUMP_RESET = 0.34
# Before SPACE: morph layout hover (fits default 3.5 m workspace: floor 0.05 + half 1.75).
_DEFAULT_PREARM_HOVER_Z = 1.80
_DEFAULT_PREARM_TAKEOFF_Z = 0.92
# Left hand: rigid motion of whole swarm (translation + rotation about formation centroid)
# Pose uses MediaPipe world landmarks in mm (wrist is not fixed at origin unlike HAND_FRAME_SCALED).
# fwd_y: dz→sim Y (in/out). Flip Y so hand-toward-camera matches expected swarm direction.
# middle_thumb palm embed: +palm Z (optical forward when palm faces cam) → world +Y; use +1 on Y.
# palm X→world X (lateral), palm Z→world Y (fwd/back), palm Y→world Z (up). Default Z=+1: fingertip → altitude.
_DEFAULT_LEFT_AXIS_SIGN: tuple[float, float, float] = (1.0, 1.0, 1.0)
# Palm/camera mm → world meters (× on palm components before axis_sign). ~15 mm ≈ 0.11 m.
_DEFAULT_LEFT_TRANS_SCALE_MM = 0.0075
_DEFAULT_LEFT_ROT_SCALE = 1.0
_DEFAULT_LEFT_TRANS_EMA = 1.0
_DEFAULT_LEFT_ROT_EMA = 1.0
_DEFAULT_LEFT_MAX_OFFSET_M = 1.35
# While armed: hold last rigid pose when the hand is briefly lost (1.0 = no decay).
_DEFAULT_LEFT_LOST_DECAY = 1.0

_DEFAULT_LEFT_MAX_ROT_RAD = 3.14
_DEFAULT_LEFT_UNWIND_S = 2.6
# Cube edge (m) centered on swarm centroid at press-0; 0 = disabled.
_DEFAULT_SWARM_WORKSPACE_BOX_M = 3.5
_DEFAULT_SWARM_WORKSPACE_WALL_MARGIN_M = 0.03
_DEFAULT_SWARM_WORKSPACE_CLEAR_MARGIN_M = 0.015
_DEFAULT_SWARM_WORKSPACE_MODE = "clip"
_DEFAULT_LEFT_PALM_DEPTH_OUTLIER_Z_MM = 105.0
_DEFAULT_LEFT_PALM_DEPTH_OUTLIER_LAT_RATIO = 2.6
_DEFAULT_LEFT_PALM_CENTER_DEPTH_EMA = 0.42
# Ignore |rotation vector| below this (rad axis-angle) after rot_scale*rot_gain — higher = calmer tilt.
_DEFAULT_LEFT_ROT_GATE_RAD = 0.014
_DEFAULT_LEFT_YAW_MIN_HORIZ = 0.17
_DEFAULT_LEFT_ROT_GAIN = 1.00
# While wrist moves: rotation cmd *= exp(-Δmm / tau). Larger tau = less suppression during pans.
_DEFAULT_LEFT_ROT_TRANS_TAU_MM = 0.0
# Damp world-up component of axis–angle (formation spin about global Z).
# 1.0: do not damp ω_world Z (0.12 made palm twist ≈ dead with ``camera`` / fwd_y).
_DEFAULT_LEFT_ROT_WORLD_Z_SCALE = 1.0
# Image dy → sim altitude (fwd_y third row). 1.0 = full −cam Y → sim Z.
_DEFAULT_LEFT_CAM_Y_TO_WORLD_Z = 1.0
# Z-up sim embedding: X=lateral, Y=in-out, Z=altitude. ``camera`` (identity) maps dy→sim Y (wrong).
_DEFAULT_LEFT_CAM_PRESET = "fwd_y"
# ``camera_at_arm``: press 0 locks depth-camera → sim map; baseline wrist + palm = origin.
_DEFAULT_LEFT_WORLD_FRAME = "camera_at_arm"
_DEFAULT_AXIS_TRANS_DEADZONE_M = 0.004
_DEFAULT_AXIS_ROT_DEADZONE_RAD = 0.014
_DEFAULT_AXIS_TRANS_ON_M = 0.004
_DEFAULT_AXIS_ROT_ON_RAD = 0.020
_DEFAULT_LEFT_PALM_BASIS = "middle_thumb"
_DEFAULT_LEFT_PLANE_ROT_SCALE_MUL = 1.0
# centroid: whole formation rotates about its centroid. per_drone: each slot pivots separately.
_DEFAULT_LEFT_ROT_PIVOT = "centroid"
# When Orbbec MP visibility min drops below threshold, estimate palm rotation from USB webcam 2D.
_DEFAULT_LEFT_DUAL_WEBCAM_ROT = True
_DEFAULT_LEFT_ROT_WEBCAM_VIS_THRESH = 0.42
_DEFAULT_LEFT_ROT_WEBCAM_INDEX = -1
_WCAM_PREVIEW_WINDOW = "Online Control Webcam (dual rotation)"
# Orbbec: optional horizontal BGR flip before MediaPipe (default off = same as pre-mirror pipeline).
_DEFAULT_ORBBEC_FLIP_HORIZONTAL = False
# Orbbec: depth→color transformed_depth (K4A API). Off by default — Femto Bolt / pyk4a wrapper can
# abort (descriptor mismatch) when enabled; use raw depth + map_color_pixel instead.
_DEFAULT_ORBBEC_USE_TRANSFORMED_DEPTH = False
_DEFAULT_ORBBEC_HAND_SWAP = "auto"
