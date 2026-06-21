"""Immutable left-hand swarm tuning and runtime bundle (yaml defaults + minimal CLI overrides)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from functions.runtime.online_defaults import ONLINE_DEFAULTS, OnlineDefaults

if TYPE_CHECKING:
    from debug.gesture_report_debug import ReportDebugPanels


@dataclass(frozen=True, slots=True)
class LeftPoseTuning:
    trans_scale: float
    rot_scale: float
    max_offset_m: float
    max_rot_rad: float
    axis_sign: tuple[float, float, float]
    lost_decay: float
    rot_gate_rad: float
    rot_gain: float
    rot_trans_tau_mm: float
    rot_world_z_scale: float
    plane_rot_scale_mul: float
    axis_trans_deadzone_m: float
    axis_rot_deadzone_rad: float
    axis_trans_on_m: float
    axis_rot_on_rad: float
    axis_trans_rot_coupling: float
    palm_depth_outlier_z_mm: float
    palm_depth_outlier_lat_ratio: float
    palm_center_depth_ema: float
    pose_debug: bool
    pose_debug_every: int
    pose_frame_viz: bool
    pose_frame_viz_every: int
    mode_vis_min: float
    mp_input_scale: float
    fps: int
    show_webcam_preview: bool
    cam_y_to_world_z: float


@dataclass(frozen=True, slots=True)
class LeftPoseRuntime:
    """All left-swarm pose settings for one online run."""

    enabled: bool
    unwind_s: float
    direct_follow: bool
    depth_frame_motion: bool
    world_frame: str
    cam_preset: str
    cam_y_to_world_z: float
    palm_basis: str
    dual_webcam_rot: bool
    rot_webcam_vis_thresh: float
    rot_webcam_index: int
    tuning: LeftPoseTuning


def _left_axis_sign_from_cli(
    args: argparse.Namespace, defaults: OnlineDefaults
) -> tuple[float, float, float]:
    base = defaults.left_hand_pose.left_axis_sign
    return (
        base[0] * (-1.0 if bool(getattr(args, "left_flip_x", False)) else 1.0),
        base[1] * (-1.0 if bool(getattr(args, "left_flip_y", False)) else 1.0),
        base[2] * (-1.0 if bool(getattr(args, "left_flip_z", False)) else 1.0),
    )


def _build_left_tuning(
    args: argparse.Namespace,
    panels: ReportDebugPanels,
    *,
    defaults: OnlineDefaults,
    fps: int,
    show_webcam_preview: bool,
    left_axis_sign: tuple[float, float, float],
    direct_follow: bool,
) -> LeftPoseTuning:
    lp = defaults.left_hand_pose
    rot_scale = float(lp.left_rot_scale)
    rot_gain = float(lp.left_rot_gain)
    rot_gate_rad = float(lp.left_rot_gate_rad)
    rot_trans_tau_mm = float(lp.left_rot_trans_tau_mm)
    rot_world_z_scale = float(lp.left_rot_world_z_scale)
    plane_rot_scale_mul = float(lp.left_plane_rot_scale_mul)
    if direct_follow:
        rot_world_z_scale = 1.0
        rot_trans_tau_mm = 0.0
        rot_gate_rad = 0.02
        plane_rot_scale_mul = 1.0
        rot_scale = float(max(rot_scale, 0.72))
        rot_gain = float(max(rot_gain, 0.92))
    pose_frame_viz = bool(lp.left_pose_frame_viz) or bool(
        getattr(args, "left_pose_frame_viz", False)
    ) or panels.palm
    return LeftPoseTuning(
        trans_scale=float(lp.left_trans_scale_mm),
        rot_scale=rot_scale,
        max_offset_m=float(lp.left_max_offset_m),
        max_rot_rad=float(lp.left_max_rot_rad),
        axis_sign=left_axis_sign,
        lost_decay=float(lp.left_lost_decay),
        rot_gate_rad=rot_gate_rad,
        rot_gain=rot_gain,
        rot_trans_tau_mm=rot_trans_tau_mm,
        rot_world_z_scale=rot_world_z_scale,
        plane_rot_scale_mul=plane_rot_scale_mul,
        axis_trans_deadzone_m=float(lp.axis_trans_deadzone_m),
        axis_rot_deadzone_rad=float(lp.axis_rot_deadzone_rad),
        axis_trans_on_m=float(lp.axis_trans_on_m),
        axis_rot_on_rad=float(lp.axis_rot_on_rad),
        axis_trans_rot_coupling=float(lp.axis_trans_rot_coupling),
        palm_depth_outlier_z_mm=float(lp.left_palm_depth_outlier_z_mm),
        palm_depth_outlier_lat_ratio=float(lp.left_palm_depth_outlier_lat_ratio),
        palm_center_depth_ema=float(lp.left_palm_center_depth_ema),
        pose_debug=bool(lp.left_pose_debug),
        pose_debug_every=int(lp.left_pose_debug_every),
        pose_frame_viz=pose_frame_viz,
        pose_frame_viz_every=int(defaults.display.left_pose_frame_viz_every),
        mode_vis_min=float(defaults.morph.mode_vis_min),
        mp_input_scale=float(defaults.camera.mp_input_scale),
        fps=int(fps),
        show_webcam_preview=bool(show_webcam_preview),
        cam_y_to_world_z=float(lp.left_cam_y_to_world_z),
    )


def build_left_pose_runtime(
    args: argparse.Namespace,
    panels: ReportDebugPanels,
    *,
    defaults: OnlineDefaults | None = None,
    fps: int,
    show_webcam_preview: bool,
) -> LeftPoseRuntime:
    """Build left pose runtime from yaml defaults and minimal CLI overrides."""
    d = ONLINE_DEFAULTS if defaults is None else defaults
    lp = d.left_hand_pose
    direct_follow = bool(lp.left_rot_direct_follow) and not bool(
        getattr(args, "no_left_rot_direct_follow", False)
    )
    left_axis_sign = _left_axis_sign_from_cli(args, d)
    tuning = _build_left_tuning(
        args,
        panels,
        defaults=d,
        fps=fps,
        show_webcam_preview=show_webcam_preview,
        left_axis_sign=left_axis_sign,
        direct_follow=direct_follow,
    )
    enabled = bool(lp.left_swarm_pose) and not bool(getattr(args, "no_left_swarm_pose", False))
    dual_webcam_rot = bool(lp.left_dual_webcam_rot) and not bool(
        getattr(args, "no_left_dual_webcam_rot", False)
    )
    return LeftPoseRuntime(
        enabled=enabled,
        unwind_s=float(lp.left_unwind_s),
        direct_follow=direct_follow,
        depth_frame_motion=bool(lp.left_swarm_depth_frame_motion),
        world_frame=str(lp.left_world_frame).strip().lower(),
        cam_preset=str(lp.left_cam_preset).strip().lower(),
        cam_y_to_world_z=float(lp.left_cam_y_to_world_z),
        palm_basis=str(lp.left_palm_basis).strip().lower(),
        dual_webcam_rot=dual_webcam_rot,
        rot_webcam_vis_thresh=float(lp.left_rot_webcam_vis_thresh),
        rot_webcam_index=int(lp.left_rot_webcam_index),
        tuning=tuning,
    )


@dataclass
class LeftPoseSensorInput:
    """Per-frame palm / depth / arm inputs for ``update_left_swarm_pose``."""

    pts_l_pose_mm: Any
    palm_center_depth_mm: Any = None
    palm_center_color_px: Any = None
    palm_calib: Any = None
    palm_frame_h: int = 0
    palm_frame_w: int = 0
    palm_depth_aligned: Any = None
    palm_depth_raw: Any = None
    palm_depth_patch_r: int = 2
    B_rot: Any = None
    cam_delta_to_world: np.ndarray | None = None
    cam_translation_to_world: np.ndarray | None = None
    arm_sim_from_cam: np.ndarray | None = None
    arm_sim_trans_from_cam: np.ndarray | None = None
    arm_cam_preset_label: str = ""
    ref_swarm_xyz: np.ndarray | None = None
    ref_basis_image: Any = None
    palm_basis: str = ""
    force_reset: bool = False
    plane_rot_mul: float = 1.0
