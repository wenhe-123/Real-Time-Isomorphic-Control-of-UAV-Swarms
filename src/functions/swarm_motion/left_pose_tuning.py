"""Immutable left-hand swarm tuning (from CLI config, adjusted once at boot)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from functions.runtime.online_runtime_config import OnlineRuntimeConfig


@dataclass(frozen=True, slots=True)
class LeftPoseTuning:
    trans_scale: float
    rot_scale: float
    trans_ema: float
    rot_ema: float
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

    @classmethod
    def from_config(
        cls,
        cfg: OnlineRuntimeConfig,
        *,
        direct_follow: bool = False,
    ) -> LeftPoseTuning:
        rot_scale = float(cfg.left_rot_scale)
        rot_gain = float(cfg.left_rot_gain)
        rot_gate_rad = float(cfg.left_rot_gate_rad)
        rot_trans_tau_mm = float(cfg.left_rot_trans_tau_mm)
        rot_world_z_scale = float(cfg.left_rot_world_z_scale)
        plane_rot_scale_mul = float(cfg.left_plane_rot_scale_mul)
        if direct_follow:
            rot_world_z_scale = 1.0
            rot_trans_tau_mm = 0.0
            rot_gate_rad = 0.02
            plane_rot_scale_mul = 1.0
            rot_scale = float(max(rot_scale, 0.72))
            rot_gain = float(max(rot_gain, 0.92))
        return cls(
            trans_scale=float(cfg.left_trans_scale),
            rot_scale=rot_scale,
            trans_ema=float(cfg.left_trans_ema),
            rot_ema=float(cfg.left_rot_ema),
            max_offset_m=float(cfg.left_max_offset_m),
            max_rot_rad=float(cfg.left_max_rot_rad),
            axis_sign=tuple(cfg.left_axis_sign),
            lost_decay=float(cfg.left_lost_decay),
            rot_gate_rad=rot_gate_rad,
            rot_gain=rot_gain,
            rot_trans_tau_mm=rot_trans_tau_mm,
            rot_world_z_scale=rot_world_z_scale,
            plane_rot_scale_mul=plane_rot_scale_mul,
            axis_trans_deadzone_m=float(cfg.left_axis_trans_deadzone_m),
            axis_rot_deadzone_rad=float(cfg.left_axis_rot_deadzone_rad),
            axis_trans_on_m=float(cfg.left_axis_trans_on_m),
            axis_rot_on_rad=float(cfg.left_axis_rot_on_rad),
            axis_trans_rot_coupling=float(cfg.left_axis_trans_rot_coupling),
            palm_depth_outlier_z_mm=float(cfg.left_palm_depth_outlier_z_mm),
            palm_depth_outlier_lat_ratio=float(cfg.left_palm_depth_outlier_lat_ratio),
            palm_center_depth_ema=float(cfg.left_palm_center_depth_ema),
            pose_debug=bool(cfg.left_pose_debug),
            pose_debug_every=int(cfg.left_pose_debug_every),
            pose_frame_viz=bool(cfg.left_pose_frame_viz),
            pose_frame_viz_every=int(cfg.left_pose_frame_viz_every),
            mode_vis_min=float(cfg.mode_vis_min),
            mp_input_scale=float(cfg.mp_input_scale),
            fps=int(cfg.fps),
            show_webcam_preview=bool(cfg.show_webcam_preview),
            cam_y_to_world_z=float(cfg.left_cam_y_to_world_z),
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
    ref_drone_xyz: np.ndarray | None = None
    ref_swarm_xyz: np.ndarray | None = None
    ref_basis_image: Any = None
    palm_basis: str = ""
    force_reset: bool = False
    plane_rot_mul: float = 1.0
