"""Immutable per-run settings for the online control main loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from functions.display_sim.gesture_report_debug import ReportDebugPanels
from functions.open_close.morph_world import ScaleConfig
from functions.runtime.online_defaults import (
    _DEFAULT_LEFT_TRANS_SCALE_MM,
    _ONLINE_MP_INPUT_SCALE,
)
from functions.runtime.pipeline_tuning import PipelineTuning
from functions.swarm_motion.axswarm_runtime import load_axswarm_min_separation


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
    left_swarm_pose: bool
    left_unwind_s: float
    left_rot_direct_follow: bool
    left_swarm_depth_frame_motion: bool
    left_world_frame: str
    left_cam_preset: str
    left_cam_y_to_world_z: float
    left_palm_basis: str
    left_plane_rot_scale_mul: float
    left_rot_pivot: str
    left_dual_webcam_rot: bool
    left_rot_webcam_vis_thresh: float
    left_rot_scale: float
    left_rot_gain: float
    left_rot_gate_rad: float
    left_rot_trans_tau_mm: float
    left_rot_world_z_scale: float
    orbbec_flip_horizontal: bool
    orbbec_use_transformed_depth: bool
    orbbec_hand_swap: str
    center_trace: bool
    center_trace_every: int
    install_hotkey_deps: bool
    global_hotkeys: bool
    drones_config: str | None
    morph_radius_mm: float
    trail_every_n: int
    led_every_n: int
    sim_render_every: int
    imshow_every: int
    mp_detect_every: int
    left_trans_scale: float
    left_trans_ema: float
    left_rot_ema: float
    left_max_offset_m: float
    left_max_rot_rad: float
    left_axis_sign: tuple[float, float, float]
    left_lost_decay: float
    debug_drone_targets_every: int
    debug_drone_pos_every: int
    spacing_audit_every: int
    left_axis_trans_deadzone_m: float
    left_axis_rot_deadzone_rad: float
    left_axis_trans_on_m: float
    left_axis_rot_on_rad: float
    left_axis_trans_rot_coupling: float
    left_palm_depth_outlier_z_mm: float
    left_palm_depth_outlier_lat_ratio: float
    left_palm_center_depth_ema: float
    left_pose_frame_viz: bool
    left_pose_frame_viz_every: int
    left_pose_debug: bool
    left_pose_debug_every: int
    webcam_rot_stride: int
    show_webcam_preview: bool
    mode_vis_min: float
    open_vis_min: float
    formation_rigid_3d_debug: bool
    left_rot_webcam_index: int
    draw_hand_debug: bool
    profile_frame: bool
    profile_every: int
    mp_input_scale: float = _ONLINE_MP_INPUT_SCALE


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
) -> OnlineRuntimeConfig:
    """Map CLI args + prepared scale/pipeline into a normalized runtime config."""
    left_trans_scale = args.left_trans_scale
    if left_trans_scale is None:
        left_trans_scale = _DEFAULT_LEFT_TRANS_SCALE_MM
    left_axis_sign = (
        -1.0 if args.left_flip_x else 1.0,
        -1.0 if args.left_flip_y else 1.0,
        -1.0 if args.left_flip_z else 1.0,
    )
    debug_drone_pos_every = max(0, int(args.debug_drone_pos_every))
    if bool(args.debug_drone_pos) and debug_drone_pos_every == 0:
        debug_drone_pos_every = 1
    settings_path = Path(args.axswarm_settings) if args.axswarm_settings else None
    min_separation_m = load_axswarm_min_separation(
        settings_path=settings_path,
    )
    return OnlineRuntimeConfig(
        point_count=int(point_count),
        fps=int(args.fps),
        min_separation_m=min_separation_m,
        scale=scale,
        pipe=pipeline,
        drone_model=str(args.drone_model),
        prearm_hover_z=float(args.prearm_hover_z),
        prearm_takeoff_z=float(args.prearm_takeoff_z),
        axswarm_settings=args.axswarm_settings,
        max_sim_substeps=int(args.max_sim_substeps),
        plot_every_n=max(0, int(pipeline.plot_every_n)),
        report_panels=panels if panels.any_enabled() else None,
        left_swarm_pose=bool(args.left_swarm_pose),
        left_unwind_s=float(args.left_unwind_seconds),
        left_rot_direct_follow=bool(args.left_rot_direct_follow),
        left_swarm_depth_frame_motion=bool(args.left_swarm_depth_frame_motion),
        left_world_frame=str(args.left_world_frame).strip().lower(),
        left_cam_preset=str(args.left_cam_preset).strip().lower(),
        left_cam_y_to_world_z=float(args.left_cam_y_to_world_z),
        left_palm_basis=str(args.left_palm_basis).strip().lower(),
        left_plane_rot_scale_mul=float(args.left_plane_rot_scale_mul),
        left_rot_pivot=str(args.left_rot_pivot).strip().lower(),
        left_dual_webcam_rot=not bool(args.no_left_dual_webcam_rot),
        left_rot_webcam_vis_thresh=float(args.left_rot_webcam_vis_thresh),
        left_rot_scale=float(args.left_rot_scale),
        left_rot_gain=float(args.left_rot_gain),
        left_rot_gate_rad=float(np.radians(float(args.left_rot_gate_deg))),
        left_rot_trans_tau_mm=float(args.left_rot_trans_tau_mm),
        left_rot_world_z_scale=float(args.left_rot_world_z_scale),
        orbbec_flip_horizontal=bool(args.orbbec_flip_horizontal),
        orbbec_use_transformed_depth=bool(args.orbbec_use_transformed_depth),
        orbbec_hand_swap=str(args.orbbec_hand_swap).strip().lower(),
        center_trace=bool(args.center_trace),
        center_trace_every=max(1, int(args.center_trace_every)),
        install_hotkey_deps=bool(args.install_hotkey_deps),
        global_hotkeys=not bool(args.no_global_hotkeys),
        drones_config=str(args.drones_config) if args.drones_config else None,
        morph_radius_mm=float(args.radius_mm),
        trail_every_n=max(0, int(args.trail_every)),
        led_every_n=max(1, int(args.led_every)),
        sim_render_every=max(0, int(args.sim_render_every)),
        imshow_every=max(1, int(args.imshow_every)),
        mp_detect_every=max(1, int(args.mp_detect_every)),
        left_trans_scale=float(left_trans_scale),
        left_trans_ema=float(args.left_trans_ema),
        left_rot_ema=float(args.left_rot_ema),
        left_max_offset_m=float(args.left_max_offset_m),
        left_max_rot_rad=float(args.left_max_rot_rad),
        left_axis_sign=left_axis_sign,
        left_lost_decay=float(args.left_lost_decay),
        debug_drone_targets_every=max(0, int(args.debug_drone_targets_every)),
        debug_drone_pos_every=debug_drone_pos_every,
        spacing_audit_every=int(args.spacing_audit_every),
        left_axis_trans_deadzone_m=float(args.left_axis_trans_deadzone_m),
        left_axis_rot_deadzone_rad=float(np.radians(args.left_axis_rot_deadzone_deg)),
        left_axis_trans_on_m=float(args.left_axis_trans_on_m),
        left_axis_rot_on_rad=float(np.radians(args.left_axis_rot_on_deg)),
        left_axis_trans_rot_coupling=float(args.left_trans_rot_coupling),
        left_palm_depth_outlier_z_mm=float(args.left_palm_depth_outlier_z_mm),
        left_palm_depth_outlier_lat_ratio=float(args.left_palm_depth_outlier_lat_ratio),
        left_palm_center_depth_ema=float(args.left_palm_center_depth_ema),
        left_pose_frame_viz=bool(args.left_pose_frame_viz) or panels.palm,
        left_pose_frame_viz_every=int(args.left_pose_frame_viz_every),
        left_pose_debug=bool(args.left_pose_debug),
        left_pose_debug_every=int(args.left_pose_debug_every),
        webcam_rot_stride=max(1, int(args.webcam_rot_stride)),
        show_webcam_preview=bool(args.show_webcam_preview),
        mode_vis_min=float(args.mode_vis_min),
        open_vis_min=float(args.open_vis_min),
        formation_rigid_3d_debug=bool(args.formation_rigid_3d_debug),
        left_rot_webcam_index=int(args.left_rot_webcam_index),
        draw_hand_debug=bool(args.draw_hand_debug) or panels.hand,
        profile_frame=bool(args.profile_frame),
        profile_every=int(args.profile_every),
    )
