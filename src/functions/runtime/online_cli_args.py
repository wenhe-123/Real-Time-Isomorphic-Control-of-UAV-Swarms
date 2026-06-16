"""Argparse definitions for online Crazyflow control."""

from __future__ import annotations

import argparse

import numpy as np

from functions.runtime.online_defaults import (
    _DEFAULT_AXIS_ROT_DEADZONE_RAD,
    _DEFAULT_AXIS_ROT_ON_RAD,
    _DEFAULT_AXIS_TRANS_DEADZONE_M,
    _DEFAULT_AXIS_TRANS_ON_M,
    _DEFAULT_DRONE_MODEL,
    _DEFAULT_LEFT_CAM_PRESET,
    _DEFAULT_LEFT_CAM_Y_TO_WORLD_Z,
    _DEFAULT_LEFT_MAX_OFFSET_M,
    _DEFAULT_LEFT_MAX_ROT_RAD,
    _DEFAULT_LEFT_PALM_BASIS,
    _DEFAULT_LEFT_PALM_CENTER_DEPTH_EMA,
    _DEFAULT_LEFT_PALM_DEPTH_OUTLIER_LAT_RATIO,
    _DEFAULT_LEFT_PALM_DEPTH_OUTLIER_Z_MM,
    _DEFAULT_LEFT_PLANE_ROT_SCALE_MUL,
    _DEFAULT_LEFT_POSE_FRAME_VIZ_EVERY,
    _DEFAULT_LEFT_ROT_EMA,
    _DEFAULT_LEFT_ROT_GAIN,
    _DEFAULT_LEFT_ROT_GATE_RAD,
    _DEFAULT_LEFT_ROT_PIVOT,
    _DEFAULT_LEFT_ROT_SCALE,
    _DEFAULT_LEFT_ROT_TRANS_TAU_MM,
    _DEFAULT_LEFT_ROT_WEBCAM_INDEX,
    _DEFAULT_LEFT_ROT_WEBCAM_VIS_THRESH,
    _DEFAULT_LEFT_ROT_WORLD_Z_SCALE,
    _DEFAULT_LEFT_TRANS_EMA,
    _DEFAULT_LEFT_TRANS_SCALE_MM,
    _DEFAULT_LEFT_UNWIND_S,
    _DEFAULT_LEFT_WORLD_FRAME,
    _DEFAULT_LEFT_YAW_MIN_HORIZ,
    _DEFAULT_MIN_SEPARATION_M,
    _DEFAULT_MODE_VIS_MIN,
    _DEFAULT_MORPH_WORLD_SCALE,
    _DEFAULT_ONLINE_IMSHOW_EVERY,
    _DEFAULT_OPEN_JUMP_RESET,
    _DEFAULT_OPEN_VIS_MIN,
    _DEFAULT_ORBBEC_FLIP_HORIZONTAL,
    _DEFAULT_ORBBEC_HAND_SWAP,
    _DEFAULT_ORBBEC_USE_TRANSFORMED_DEPTH,
    _DEFAULT_PREARM_HOVER_Z,
    _DEFAULT_PREARM_TAKEOFF_Z,
    _DEFAULT_RAW_TARGET_EMA,
    _DEFAULT_SIM_RENDER_EVERY,
    _DEFAULT_SWARM_WORKSPACE_BOX_M,
    _DEFAULT_SWARM_WORKSPACE_CLEAR_MARGIN_M,
    _DEFAULT_SWARM_WORKSPACE_MODE,
    _DEFAULT_SWARM_WORKSPACE_WALL_MARGIN_M,
    _DEFAULT_WEBCAM_ROT_STRIDE,
    _LED_APPLY_EVERY_FRAMES,
    _ONLINE_MAX_SIM_SUBSTEPS_PER_FRAME,
    _ONLINE_MP_DETECT_EVERY,
    _TRAIL_DRAW_EVERY_FRAMES,
)
from functions.runtime.pipeline_tuning import ONLINE_PLOT_EVERY_N
from functions.swarm_motion.left_hand_swarm_pose import LEFT_PALM_BASIS_PRESETS


def build_online_control_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online Crazyflow control from Orbbec/Webcam morph targets."
    )
    parser.add_argument(
        "--point-count",
        type=int,
        default=24,
        help="Number of surface samples (>=8). Interactive TTY still prompts; stdin non-TTY uses this value with no prompt.",
    )
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--open", type=float, default=1.0, dest="open_alpha")
    parser.add_argument("--shape-t", type=float, default=None)
    parser.add_argument("--radius-mm", type=float, default=50.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run time in seconds; <=0 means run until q/Enter.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Nominal outer-loop rate for timestamps and sim substeps (≈ sim.freq/fps steps per iteration). "
        "Use 30 with Orbbec to match hand_tracking_orbbec.py (FPS.FPS_30).",
    )
    parser.add_argument(
        "--min-separation-m",
        type=float,
        default=_DEFAULT_MIN_SEPARATION_M,
        help="Minimum inter-drone spacing guard in meters.",
    )
    parser.add_argument("--xy-radius", type=float, default=3.00)
    parser.add_argument("--z-center", type=float, default=1.40)
    parser.add_argument(
        "--z-amplitude",
        type=float,
        default=0.35,
        help="Unused for mapping (kept for CLI compatibility); vertical extent uses --z-mm-scale.",
    )
    parser.add_argument("--z-min", type=float, default=1.05)
    parser.add_argument("--z-max", type=float, default=2.25)
    parser.add_argument("--reference-xy-extent-mm", type=float, default=100.0)
    parser.add_argument(
        "--reference-z-extent-mm",
        type=float,
        default=100.0,
        help="Mm span used for Z scaling (match --reference-xy-extent-mm for same aspect as the 3D topo plot).",
    )
    parser.add_argument(
        "--z-mm-scale",
        type=float,
        default=1.0,
        help="Extra factor on Z mm→m only (1.0 = isotropic with XY when references match).",
    )
    parser.add_argument(
        "--debug-drone-targets-every",
        type=int,
        default=0,
        metavar="N",
        help="Print each drone target xyz every N frames (0=off); also prints sim positions as sim_pos.",
    )
    parser.add_argument(
        "--formation-rigid-3d-debug",
        action="store_true",
        help="On the Matplotlib topo axis, draw blue (morph-only) vs magenta (after L-hand rigid) debug points.",
    )
    parser.add_argument(
        "--left-pose-frame-viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw Orbbec overlay: hand palm RGB axes (hX/hY/hZ) + arm ref (0*) + swarm centroid inset.",
    )
    parser.add_argument(
        "--left-pose-frame-viz-every",
        type=int,
        default=_DEFAULT_LEFT_POSE_FRAME_VIZ_EVERY,
        metavar="N",
        help="Draw pose axis overlay every N frames (default 3).",
    )
    parser.add_argument(
        "--webcam-rot-stride",
        type=int,
        default=_DEFAULT_WEBCAM_ROT_STRIDE,
        metavar="N",
        help="When using dual webcam rotation (low visibility), read webcam at most every N frames (default 3).",
    )
    parser.add_argument(
        "--orbbec-flip-horizontal",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_ORBBEC_FLIP_HORIZONTAL,
        help="Orbbec: flip BGR (and depth when same resolution) before MediaPipe. Default off (legacy); "
        "use --orbbec-flip-horizontal if the preview looks mirrored vs the room.",
    )
    parser.add_argument(
        "--orbbec-hand-swap",
        type=str,
        choices=("auto", "on", "off"),
        default=_DEFAULT_ORBBEC_HAND_SWAP,
        help="Orbbec: swap MediaPipe left/right for mode vs open. auto=swap only when --orbbec-flip-horizontal is on; "
        "on/off=force always/never.",
    )
    parser.add_argument(
        "--orbbec-use-transformed-depth",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_ORBBEC_USE_TRANSFORMED_DEPTH,
        help="Orbbec: use SDK color-aligned depth (transformed_depth). Default off: Femto Bolt + K4A-wrapper often crashes.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--drone-model",
        type=str,
        default=_DEFAULT_DRONE_MODEL,
        help="Crazyflow drone model name (e.g. cf21B_500, cf2x_L250).",
    )
    parser.add_argument(
        "--trail-every",
        type=int,
        default=_TRAIL_DRAW_EVERY_FRAMES,
        metavar="N",
        help="Trail: append path every frame when >0; draw_line every N frames (1=continuous; 0=off).",
    )
    parser.add_argument(
        "--led-every",
        type=int,
        default=_LED_APPLY_EVERY_FRAMES,
        metavar="N",
        help="LED material refresh every N camera frames (1 = highest refresh).",
    )
    parser.add_argument(
        "--sim-render-every",
        type=int,
        default=_DEFAULT_SIM_RENDER_EVERY,
        metavar="N",
        help="Render Crazyflow every N camera frames (0=disable sim render window).",
    )
    parser.add_argument(
        "--max-sim-substeps",
        type=int,
        default=_ONLINE_MAX_SIM_SUBSTEPS_PER_FRAME,
        help="Cap physics substeps per camera iteration (wall-clock dt * sim.freq).",
    )
    parser.add_argument(
        "--imshow-every",
        type=int,
        default=int(_DEFAULT_ONLINE_IMSHOW_EVERY),
        metavar="N",
        help="Show OpenCV window every N iterations (2+ reduces UI stalls / perceived dropped frames).",
    )
    parser.add_argument(
        "--raw-target-ema",
        type=float,
        default=_DEFAULT_RAW_TARGET_EMA,
        help="EMA on morph targets before spacing (0=off, ~0.3–0.5 smooths motion).",
    )
    parser.add_argument(
        "--open-jump-reset",
        type=float,
        default=_DEFAULT_OPEN_JUMP_RESET,
        help="If |Δopen| >= this in one frame, snap internal smooth target to collision-safe target (0=off). "
        "Reduces lag after plane↔sphere transitions.",
    )
    parser.add_argument(
        "--left-swarm-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Left-hand whole-formation move/rotate (default: on). 0 = arm, 0 again = smooth unwind; "
        "use --no-left-swarm-pose to disable.",
    )
    parser.add_argument(
        "--left-trans-scale",
        type=float,
        default=None,
        help="Wrist delta (mm in MP world) -> meters; default %.4f."
        % (_DEFAULT_LEFT_TRANS_SCALE_MM,),
    )
    parser.add_argument("--left-rot-scale", type=float, default=_DEFAULT_LEFT_ROT_SCALE)
    parser.add_argument("--left-trans-ema", type=float, default=_DEFAULT_LEFT_TRANS_EMA)
    parser.add_argument("--left-rot-ema", type=float, default=_DEFAULT_LEFT_ROT_EMA)
    parser.add_argument("--left-max-offset-m", type=float, default=_DEFAULT_LEFT_MAX_OFFSET_M)
    parser.add_argument("--left-max-rot-rad", type=float, default=_DEFAULT_LEFT_MAX_ROT_RAD)
    parser.add_argument("--left-lost-decay", type=float, default=1.0)
    parser.add_argument(
        "--left-unwind-seconds",
        type=float,
        default=_DEFAULT_LEFT_UNWIND_S,
        help="Second press of 0: seconds to ease left-hand rigid back to morph-only frame.",
    )
    parser.add_argument(
        "--left-swarm-depth-frame-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Orbbec: wrist delta in depth-camera mm → world (see --left-cam-preset, --left-cam-y-to-world-z). "
        "Uses palm-frame rotation (axis-angle). Webcam/legacy: turn off.",
    )
    parser.add_argument(
        "--left-cam-preset",
        type=str,
        default=_DEFAULT_LEFT_CAM_PRESET,
        choices=("camera", "legacy", "fwd_y", "flip_depth"),
        help="Embed depth camera (+X right,+Y down,+Z forward) into sim axes. "
        "camera: identity (sim axes = cam at arm). fwd_y: Z-up style (near/far→sim Y). "
        "With --left-world-frame camera_at_arm this matrix is frozen at press 0.",
    )
    parser.add_argument(
        "--left-world-frame",
        type=str,
        default=_DEFAULT_LEFT_WORLD_FRAME,
        choices=("camera_at_arm", "sim"),
        help="camera_at_arm: press 0 locks wrist+palm + cam→sim map (absolute ref). "
        "sim: cam→sim map from CLI every frame (legacy).",
    )
    parser.add_argument(
        "--left-axis-trans-deadzone-m",
        type=float,
        default=_DEFAULT_AXIS_TRANS_DEADZONE_M,
        help="axis_locked: ignore translation components smaller than this (m).",
    )
    parser.add_argument(
        "--left-axis-rot-deadzone-deg",
        type=float,
        default=float(np.degrees(_DEFAULT_AXIS_ROT_DEADZONE_RAD)),
        help="axis_locked: ignore rotation components smaller than this (degrees).",
    )
    parser.add_argument(
        "--left-axis-trans-on-m",
        type=float,
        default=_DEFAULT_AXIS_TRANS_ON_M,
        help="axis_locked: min world translation norm to count as translate-dominant.",
    )
    parser.add_argument(
        "--left-axis-rot-on-deg",
        type=float,
        default=float(np.degrees(_DEFAULT_AXIS_ROT_ON_RAD)),
        help="axis_locked: min intrinsic palm rotation (deg) to count as rotate-dominant.",
    )
    parser.add_argument(
        "--install-hotkey-deps",
        action="store_true",
        help="pip install pynput+keyboard into this Python before start (if missing).",
    )
    parser.add_argument(
        "--left-cam-y-to-world-z",
        type=float,
        default=_DEFAULT_LEFT_CAM_Y_TO_WORLD_Z,
        help="Scale image-plane dy (camera mm) → world altitude (0..1). Higher = more world-Z from vertical hand motion; 0 disables.",
    )
    parser.add_argument(
        "--left-plane-rot-scale-mul",
        type=float,
        default=_DEFAULT_LEFT_PLANE_ROT_SCALE_MUL,
        help="When right-hand open snap is plane (spread→plane branch), multiply left rot_scale by this (0..1). 1 = no extra cut.",
    )
    parser.add_argument(
        "--left-rot-pivot",
        type=str,
        default=_DEFAULT_LEFT_ROT_PIVOT,
        choices=("per_drone", "centroid"),
        help="centroid: whole formation spins about its center (default). "
        "per_drone: each drone pivots about its own arm-time position.",
    )
    parser.add_argument(
        "--no-left-dual-webcam-rot",
        action="store_true",
        help="Disable USB webcam palm rotation when Orbbec MP visibility is low (default: dual on for Orbbec).",
    )
    parser.add_argument(
        "--show-webcam-preview",
        action="store_true",
        help="Show USB webcam window with rot=depth|webcam|orbbec2d and vis_min (dual rotation debug).",
    )
    parser.add_argument(
        "--no-global-hotkeys",
        action="store_true",
        help="Only read keys from OpenCV windows (SPACE/0 require Orbbec window focus).",
    )
    parser.add_argument(
        "--mode-vis-min",
        type=float,
        default=_DEFAULT_MODE_VIS_MIN,
        help="Min MediaPipe left-hand visibility (0..1) to allow morph mode M1..M5 to change; below = hold mode.",
    )
    parser.add_argument(
        "--left-rot-webcam-vis-thresh",
        type=float,
        default=_DEFAULT_LEFT_ROT_WEBCAM_VIS_THRESH,
        help="If Orbbec hand visibility min < this, use webcam 2D palm basis for rotation (translation stays depth).",
    )
    parser.add_argument(
        "--left-rot-webcam-index",
        type=int,
        default=_DEFAULT_LEFT_ROT_WEBCAM_INDEX,
        help="OpenCV index for rotation webcam (-1=auto scan).",
    )
    parser.add_argument(
        "--left-rot-direct-follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Palm rotation follows hand directly: palm-local normal twist maps to world Z yaw; "
        "use --no-left-rot-direct-follow to restore calmer legacy damping.",
    )
    parser.add_argument(
        "--left-palm-basis",
        type=str,
        default=_DEFAULT_LEFT_PALM_BASIS,
        choices=tuple(sorted(LEFT_PALM_BASIS_PRESETS)),
        help="Two MCPs (minus wrist) define palm rotation frame: index_ring (default, wider span), index_middle, middle_ring.",
    )
    parser.add_argument(
        "--left-rot-gate-deg",
        type=float,
        default=float(np.degrees(_DEFAULT_LEFT_ROT_GATE_RAD)),
        help="Min |rotation vector| in degrees (axis-angle magnitude) before applying formation rotation.",
    )
    parser.add_argument(
        "--left-yaw-min-horiz",
        type=float,
        default=_DEFAULT_LEFT_YAW_MIN_HORIZ,
        help="Unused (legacy); kept for script compatibility.",
    )
    parser.add_argument(
        "--left-rot-gain",
        type=float,
        default=_DEFAULT_LEFT_ROT_GAIN,
        help="Multiplier on measured rotation (after rot-scale); lower = calmer formation.",
    )
    parser.add_argument(
        "--left-rot-trans-tau-mm",
        type=float,
        default=_DEFAULT_LEFT_ROT_TRANS_TAU_MM,
        help="While moving: rotation cmd *= exp(-wrist_step_mm / tau). 0 = off (default).",
    )
    parser.add_argument(
        "--left-rot-world-z-scale",
        type=float,
        default=_DEFAULT_LEFT_ROT_WORLD_Z_SCALE,
        help="0..1 multiplier on world axis–angle Z after R_to_rotvec(M R Mᵀ) (smaller = less spin about global Z). 1 = no damping.",
    )
    parser.add_argument(
        "--left-flip-x",
        action="store_true",
        help="Flip sign of left-hand world X translation (try if horizontal swarm motion feels inverted).",
    )
    parser.add_argument(
        "--left-flip-y",
        action="store_true",
        help="Flip world Y (forward/back). Try if push/pull feels reversed.",
    )
    parser.add_argument(
        "--left-flip-z",
        action="store_true",
        help="Use +1 on world Z (default is -1: palm up → sim altitude).",
    )
    parser.add_argument(
        "--left-pose-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print palm center + pose debug block to stdout while L-move armed (default on).",
    )
    parser.add_argument(
        "--left-pose-debug-every",
        type=int,
        default=15,
        metavar="N",
        help="Print left-pose debug every N frames.",
    )
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--morph-world-scale", type=float, default=_DEFAULT_MORPH_WORLD_SCALE)
    parser.add_argument("--spacing-audit-every", type=int, default=0, metavar="N")
    parser.add_argument("--prearm-hover-z", type=float, default=_DEFAULT_PREARM_HOVER_Z)
    parser.add_argument("--prearm-takeoff-z", type=float, default=_DEFAULT_PREARM_TAKEOFF_Z)
    parser.add_argument("--plot-every", type=int, default=int(ONLINE_PLOT_EVERY_N))
    parser.add_argument("--debug-3d-plot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--debug-report-viz",
        action="store_true",
        help="Open ALL report Matplotlib panels (heavy). Prefer individual --debug-report-* flags.",
    )
    parser.add_argument("--debug-report-morph", action="store_true", help="Report screenshot: morph 3D only.")
    parser.add_argument("--debug-report-hand", action="store_true", help="Report screenshot: hand landmarks + skeleton.")
    parser.add_argument("--debug-report-pca", action="store_true", help="Report screenshot: hand PCA panel.")
    parser.add_argument(
        "--debug-report-landmarks",
        action="store_true",
        help="Report screenshot: open/close/current landmark clouds.",
    )
    parser.add_argument("--debug-report-palm", action="store_true", help="Report screenshot: palm pose (cam mm).")
    parser.add_argument("--debug-webcam-pipeline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mp-detect-every", type=int, default=int(_ONLINE_MP_DETECT_EVERY))
    parser.add_argument("--draw-hand-debug", action="store_true")
    parser.add_argument("--mp-delegate", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--profile-frame", action="store_true")
    parser.add_argument("--profile-every", type=int, default=60)
    parser.add_argument("--open-vis-min", type=float, default=_DEFAULT_OPEN_VIS_MIN)
    parser.add_argument("--left-trans-rot-coupling", type=float, default=0.50)
    parser.add_argument("--planner", choices=("direct", "axswarm"), default="direct")
    parser.add_argument("--axswarm-settings", type=str, default=None)
    parser.add_argument("--axswarm-project-root", type=str, default=None)
    parser.add_argument("--axswarm-max-iters", type=int, default=None)
    parser.add_argument("--axswarm-max-solve-ms", type=float, default=90.0)
    parser.add_argument("--axswarm-max-deviation-m", type=float, default=0.2)
    parser.add_argument("--axswarm-pos-weight", type=float, default=None)
    parser.add_argument("--swarm-workspace-box-m", type=float, default=_DEFAULT_SWARM_WORKSPACE_BOX_M)
    parser.add_argument("--swarm-workspace-wall-margin-m", type=float, default=_DEFAULT_SWARM_WORKSPACE_WALL_MARGIN_M)
    parser.add_argument("--swarm-workspace-clear-margin-m", type=float, default=_DEFAULT_SWARM_WORKSPACE_CLEAR_MARGIN_M)
    parser.add_argument("--swarm-workspace-mode", choices=("clip", "freeze"), default=_DEFAULT_SWARM_WORKSPACE_MODE)
    parser.add_argument("--left-palm-depth-outlier-z-mm", type=float, default=_DEFAULT_LEFT_PALM_DEPTH_OUTLIER_Z_MM)
    parser.add_argument("--left-palm-depth-outlier-lat-ratio", type=float, default=_DEFAULT_LEFT_PALM_DEPTH_OUTLIER_LAT_RATIO)
    parser.add_argument("--left-palm-center-depth-ema", type=float, default=_DEFAULT_LEFT_PALM_CENTER_DEPTH_EMA)
    parser.add_argument("--center-trace", action="store_true")
    parser.add_argument("--center-trace-every", type=int, default=10)
    parser.add_argument(
        "--rigid-pose-trace",
        action="store_true",
        help="Record hand vs swarm rigid pose trajectories from each press-0 arm until disarm/exit.",
    )
    parser.add_argument(
        "--rigid-pose-trace-out",
        type=str,
        default=None,
        help="Output JSON path (default: logs/rigid_pose_trace_<timestamp>.json).",
    )
    parser.add_argument(
        "--rigid-pose-trace-every",
        type=int,
        default=1,
        help="Record one sample every N frames while armed (default: 1).",
    )
    parser.add_argument(
        "--drones-config",
        type=str,
        default=None,
        help="TOML file listing Crazyflie URIs (enables real-swarm mode; disables MuJoCo).",
    )
    parser.add_argument(
        "--real-lighthouse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Lighthouse deck localization (default: from drones.toml swarm.lighthouse).",
    )

    return parser


def report_debug_panels_from_args(args) -> ReportDebugPanels:
    from functions.display_sim.gesture_report_debug import ReportDebugPanels

    return ReportDebugPanels.from_cli(
        all_viz=bool(getattr(args, "debug_report_viz", False)),
        morph=bool(getattr(args, "debug_report_morph", False)),
        hand=bool(getattr(args, "debug_report_hand", False)),
        pca=bool(getattr(args, "debug_report_pca", False)),
        landmarks=bool(getattr(args, "debug_report_landmarks", False)),
        palm=bool(getattr(args, "debug_report_palm", False)),
    )
