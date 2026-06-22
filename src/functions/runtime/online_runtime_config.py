"""Immutable per-run settings for the online control main loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from debug.gesture_report_debug import ReportDebugPanels
from functions.open_close.morph_world import ScaleConfig
from functions.runtime.online_defaults import ONLINE_DEFAULTS, OnlineDefaults
from functions.runtime.pipeline_tuning import PipelineTuning
from functions.swarm_motion.axswarm_runtime import load_axswarm_min_separation
from functions.swarm_motion.left_pose_tuning import LeftPoseRuntime, build_left_pose_runtime


@dataclass(frozen=True, slots=True)
class OnlineRuntimeConfig:
    point_count: int
    fps: int
    min_separation_m: float
    scale: ScaleConfig
    pipe: PipelineTuning
    drone_model: str
    prearm_hover_z: float
    prearm_takeoff_z: float
    axswarm_settings: str | None
    max_sim_substeps: int
    plot_every_n: int
    report_panels: ReportDebugPanels | None
    left: LeftPoseRuntime
    orbbec_flip_horizontal: bool
    orbbec_use_transformed_depth: bool
    orbbec_hand_swap: str
    center_trace: bool
    center_trace_every: int
    install_hotkey_deps: bool
    global_hotkeys: bool
    drones_config: str | None
    skip_real_connect: bool
    morph_radius_mm: float
    trail_every_n: int
    led_every_n: int
    sim_render_every: int
    imshow_every: int
    mp_detect_every: int
    debug_drone_targets_every: int
    debug_drone_pos_every: int
    spacing_audit_every: int
    webcam_rot_stride: int
    show_webcam_preview: bool
    mode_vis_min: float
    open_vis_min: float
    formation_rigid_3d_debug: bool
    draw_hand_debug: bool
    profile_frame: bool
    profile_every: int
    mp_input_scale: float


@dataclass
class OnlineWebcamState:
    cap: Any | None = None
    landmarker: Any | None = None
    frame_idx: int = 0
    rot_cache: dict[str, Any] = field(
        default_factory=lambda: {"B": None, "res": None, "fr": None, "idx": None}
    )
    rot_stride: int = 6


def build_online_runtime_config(
    args: argparse.Namespace,
    *,
    point_count: int,
    scale: ScaleConfig,
    pipeline: PipelineTuning,
    panels: ReportDebugPanels,
    defaults: OnlineDefaults | None = None,
) -> OnlineRuntimeConfig:
    """Map minimal CLI args + yaml defaults into a normalized runtime config."""
    d = ONLINE_DEFAULTS if defaults is None else defaults
    debug_drone_pos_every = max(0, int(args.debug_drone_pos_every))
    if bool(args.debug_drone_pos) and debug_drone_pos_every == 0:
        debug_drone_pos_every = 1
    settings_path = Path(args.axswarm_settings) if args.axswarm_settings else None
    min_separation_m = load_axswarm_min_separation(
        settings_path=settings_path,
    )
    show_webcam_preview = bool(getattr(args, "show_webcam_preview", False))
    left = build_left_pose_runtime(
        args,
        panels,
        defaults=d,
        fps=int(args.fps),
        show_webcam_preview=show_webcam_preview,
    )
    return OnlineRuntimeConfig(
        point_count=int(point_count),
        fps=int(args.fps),
        min_separation_m=min_separation_m,
        scale=scale,
        pipe=pipeline,
        drone_model=str(d.sim.drone_model),
        prearm_hover_z=float(d.prearm.prearm_hover_z),
        prearm_takeoff_z=float(d.prearm.prearm_takeoff_z),
        axswarm_settings=args.axswarm_settings,
        max_sim_substeps=int(d.sim.max_sim_substeps_per_frame),
        plot_every_n=max(0, int(pipeline.plot_every_n)),
        report_panels=panels if panels.any_enabled() else None,
        left=left,
        orbbec_flip_horizontal=bool(d.camera.orbbec_flip_horizontal),
        orbbec_use_transformed_depth=bool(d.camera.orbbec_use_transformed_depth),
        orbbec_hand_swap=str(d.camera.orbbec_hand_swap).strip().lower(),
        center_trace=bool(getattr(args, "center_trace", False)),
        center_trace_every=max(1, int(getattr(args, "center_trace_every", 10))),
        install_hotkey_deps=bool(getattr(args, "install_hotkey_deps", False)),
        global_hotkeys=not bool(getattr(args, "no_global_hotkeys", False)),
        drones_config=str(args.drones_config) if args.drones_config else None,
        skip_real_connect=bool(getattr(args, "skip_real_connect", False)),
        morph_radius_mm=float(args.radius_mm),
        trail_every_n=max(0, int(d.display.trail_draw_every_frames)),
        led_every_n=max(1, int(d.display.led_apply_every_frames)),
        sim_render_every=(
            0
            if args.drones_config
            else max(0, int(d.display.sim_render_every))
        ),
        imshow_every=max(1, int(d.display.online_imshow_every)),
        mp_detect_every=max(1, int(d.camera.mp_detect_every)),
        debug_drone_targets_every=max(0, int(args.debug_drone_targets_every)),
        debug_drone_pos_every=debug_drone_pos_every,
        spacing_audit_every=int(getattr(args, "spacing_audit_every", 0)),
        webcam_rot_stride=max(1, int(d.display.webcam_rot_stride)),
        show_webcam_preview=show_webcam_preview,
        mode_vis_min=float(d.morph.mode_vis_min),
        open_vis_min=float(d.morph.open_vis_min),
        formation_rigid_3d_debug=bool(getattr(args, "formation_rigid_3d_debug", False)),
        draw_hand_debug=bool(getattr(args, "draw_hand_debug", False)) or panels.hand,
        profile_frame=bool(getattr(args, "profile_frame", False)),
        profile_every=int(getattr(args, "profile_every", 60)),
        mp_input_scale=float(d.camera.mp_input_scale),
    )
