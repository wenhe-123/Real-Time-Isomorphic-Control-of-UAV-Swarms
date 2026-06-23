"""Argparse definitions for online Crazyflow control.

Production tuning lives in ``config/online_defaults.yaml`` (loaded by
``online_defaults.py``). This module only defines session controls and debug flags.
"""

from __future__ import annotations

import argparse

from functions.runtime.online_defaults import ONLINE_DEFAULTS
from functions.runtime.pipeline_tuning import ONLINE_PLOT_EVERY_N


def build_online_control_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online Crazyflow control from Orbbec/Webcam morph targets."
    )
    # --- Session / morph workspace (not in online_defaults.yaml) ---
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
        default=int(ONLINE_DEFAULTS.camera.orbbec_fps),
        help="Nominal outer-loop rate (match Orbbec FPS; default from online_defaults.yaml).",
    )
    parser.add_argument("--xy-radius", type=float, default=3.00)
    parser.add_argument(
        "--z-amplitude",
        type=float,
        default=0.35,
        help="Max ±Z thickness around hover_z for morph mapping (m).",
    )
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
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mp-delegate", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--drones-config",
        type=str,
        default=None,
        help="TOML file listing Crazyflie URIs (enables real-swarm mode; disables MuJoCo).",
    )
    parser.add_argument(
        "--skip-real-connect",
        action="store_true",
        help="With --drones-config: load TOML layout but skip Crazyflie radio and ROS mocap.",
    )
    parser.add_argument(
        "--no-real-track-velocity",
        action="store_true",
        help="With --drones-config: feed axswarm mocap positions with zero velocity.",
    )
    parser.add_argument("--axswarm-settings", type=str, default=None)

    # --- Left-hand: yaml defaults; only sign fixes and opt-outs on CLI ---
    parser.add_argument(
        "--no-left-swarm-pose",
        action="store_true",
        help="Disable left-hand whole-formation rigid control (default on via yaml).",
    )
    parser.add_argument(
        "--no-left-dual-webcam-rot",
        action="store_true",
        help="Disable USB webcam palm rotation when Orbbec visibility is low.",
    )
    parser.add_argument(
        "--no-left-rot-direct-follow",
        action="store_true",
        help="Restore calmer legacy palm rotation damping (yaml default: direct follow).",
    )
    parser.add_argument(
        "--left-flip-x",
        action="store_true",
        help="Flip sign of left-hand world X translation.",
    )
    parser.add_argument(
        "--left-flip-y",
        action="store_true",
        help="Flip sign of left-hand world Y translation.",
    )
    parser.add_argument(
        "--left-flip-z",
        action="store_true",
        help="Flip sign of left-hand world Z translation.",
    )
    parser.add_argument(
        "--left-pose-frame-viz",
        action="store_true",
        help="Debug: draw palm RGB axes + arm ref on Orbbec overlay.",
    )

    # --- Hotkeys ---
    parser.add_argument(
        "--install-hotkey-deps",
        action="store_true",
        help="pip install pynput+keyboard into this Python before start (if missing).",
    )
    parser.add_argument(
        "--no-global-hotkeys",
        action="store_true",
        help="Only read keys from OpenCV windows (SPACE/0 require Orbbec window focus).",
    )

    # --- Debug / profiling ---
    parser.add_argument(
        "--debug-drone-targets-every",
        type=int,
        default=0,
        metavar="N",
        help="Print full target chain every N frames (0=off).",
    )
    parser.add_argument(
        "--debug-drone-pos-every",
        type=int,
        default=0,
        metavar="N",
        help="Print each drone cmd_target + sim/real xyz every N frames (0=off).",
    )
    parser.add_argument(
        "--debug-drone-pos",
        action="store_true",
        help="Shorthand for --debug-drone-pos-every 1.",
    )
    parser.add_argument(
        "--formation-rigid-3d-debug",
        action="store_true",
        help="Matplotlib topo: blue morph-only vs magenta after L-hand rigid.",
    )
    parser.add_argument(
        "--show-webcam-preview",
        action="store_true",
        help="Show USB webcam window (dual rotation debug).",
    )
    parser.add_argument("--draw-hand-debug", action="store_true")
    parser.add_argument("--spacing-audit-every", type=int, default=0, metavar="N")
    parser.add_argument("--center-trace", action="store_true")
    parser.add_argument("--center-trace-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=int(ONLINE_PLOT_EVERY_N))
    parser.add_argument("--debug-3d-plot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--debug-report-viz",
        action="store_true",
        help="Open ALL report Matplotlib panels (heavy). Prefer individual --debug-report-* flags.",
    )
    parser.add_argument("--debug-report-morph", action="store_true")
    parser.add_argument("--debug-report-hand", action="store_true")
    parser.add_argument("--debug-report-pca", action="store_true")
    parser.add_argument("--debug-report-landmarks", action="store_true")
    parser.add_argument("--debug-report-palm", action="store_true")
    parser.add_argument("--debug-webcam-pipeline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-frame", action="store_true")
    parser.add_argument("--profile-every", type=int, default=60)
    parser.add_argument(
        "--rigid-pose-trace",
        action="store_true",
        help="Record hand vs swarm rigid pose from each press-0 arm until disarm/exit.",
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
        help="Record one sample every N frames while armed.",
    )

    return parser


def report_debug_panels_from_args(args) -> ReportDebugPanels:
    from debug.gesture_report_debug import ReportDebugPanels

    return ReportDebugPanels.from_cli(
        all_viz=bool(getattr(args, "debug_report_viz", False)),
        morph=bool(getattr(args, "debug_report_morph", False)),
        hand=bool(getattr(args, "debug_report_hand", False)),
        pca=bool(getattr(args, "debug_report_pca", False)),
        landmarks=bool(getattr(args, "debug_report_landmarks", False)),
        palm=bool(getattr(args, "debug_report_palm", False)),
    )
