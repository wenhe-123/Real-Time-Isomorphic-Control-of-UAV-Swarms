"""Load production online control defaults from ``config/online_defaults.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SECTION_KEYS = ("Display", "Camera", "Morph", "Sim", "Prearm", "LeftHandPose")


@dataclass(frozen=True, slots=True)
class DisplayDefaults:
    left_pose_frame_viz_every: int
    webcam_rot_stride: int
    trail_draw_every_frames: int
    led_apply_every_frames: int
    sim_render_every: int
    online_imshow_every: int
    trail_buffer_maxlen: int
    wcam_preview_window: str


@dataclass(frozen=True, slots=True)
class CameraDefaults:
    orbbec_fps: str
    mp_input_scale: float
    mp_detect_every: int
    orbbec_flip_horizontal: bool
    orbbec_use_transformed_depth: bool
    orbbec_hand_swap: str


@dataclass(frozen=True, slots=True)
class MorphDefaults:
    target_alpha: float
    morph_world_scale: float
    mode_rot_freeze_latch: int
    mode_vis_min: float
    open_vis_min: float


@dataclass(frozen=True, slots=True)
class SimDefaults:
    drone_model: str
    control_freq_hz: float
    max_sim_substeps_per_frame: int
    ground_z: float


@dataclass(frozen=True, slots=True)
class PrearmDefaults:
    prearm_hover_z: float
    prearm_takeoff_z: float


@dataclass(frozen=True, slots=True)
class LeftHandPoseDefaults:
    left_swarm_pose: bool
    left_rot_direct_follow: bool
    left_swarm_depth_frame_motion: bool
    left_pose_debug: bool
    left_pose_debug_every: int
    left_pose_frame_viz: bool
    left_unwind_s: float
    left_cam_preset: str
    left_world_frame: str
    left_cam_y_to_world_z: float
    left_palm_basis: str
    left_axis_sign: tuple[float, float, float]
    left_trans_scale_mm: float
    left_max_offset_m: float
    left_lost_decay: float
    left_rot_scale: float
    left_rot_gain: float
    left_rot_gate_rad: float
    left_max_rot_rad: float
    left_rot_trans_tau_mm: float
    left_rot_world_z_scale: float
    left_plane_rot_scale_mul: float
    left_palm_depth_outlier_z_mm: float
    left_palm_depth_outlier_lat_ratio: float
    left_palm_center_depth_ema: float
    axis_trans_deadzone_m: float
    axis_rot_deadzone_rad: float
    axis_trans_on_m: float
    axis_rot_on_rad: float
    axis_trans_rot_coupling: float
    left_dual_webcam_rot: bool
    left_rot_webcam_vis_thresh: float
    left_rot_webcam_index: int


@dataclass(frozen=True, slots=True)
class OnlineDefaults:
    display: DisplayDefaults
    camera: CameraDefaults
    morph: MorphDefaults
    sim: SimDefaults
    prearm: PrearmDefaults
    left_hand_pose: LeftHandPoseDefaults

    @classmethod
    def load(cls, path: Path | None = None) -> OnlineDefaults:
        settings_path = default_online_defaults_path() if path is None else Path(path)
        with open(settings_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"online defaults must be a mapping, got {type(raw).__name__}")
        return cls.from_yaml(raw)

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> OnlineDefaults:
        unknown = sorted(set(raw) - set(_SECTION_KEYS))
        if unknown:
            raise KeyError(f"unknown online_defaults.yaml sections: {', '.join(unknown)}")
        missing = sorted(set(_SECTION_KEYS) - set(raw))
        if missing:
            raise KeyError(f"missing online_defaults.yaml sections: {', '.join(missing)}")
        for section in _SECTION_KEYS:
            if not isinstance(raw[section], dict):
                raise TypeError(f"section {section!r} must be a mapping")
        d = raw["Display"]
        c = raw["Camera"]
        m = raw["Morph"]
        s = raw["Sim"]
        p = raw["Prearm"]
        l = raw["LeftHandPose"]
        return cls(
            display=DisplayDefaults(
                left_pose_frame_viz_every=int(d["left_pose_frame_viz_every"]),
                webcam_rot_stride=int(d["webcam_rot_stride"]),
                trail_draw_every_frames=int(d["trail_draw_every_frames"]),
                led_apply_every_frames=int(d["led_apply_every_frames"]),
                sim_render_every=int(d["sim_render_every"]),
                online_imshow_every=int(d["online_imshow_every"]),
                trail_buffer_maxlen=int(d["trail_buffer_maxlen"]),
                wcam_preview_window=str(d["wcam_preview_window"]),
            ),
            camera=CameraDefaults(
                orbbec_fps=str(c["orbbec_fps"]),
                mp_input_scale=float(c["mp_input_scale"]),
                mp_detect_every=int(c["mp_detect_every"]),
                orbbec_flip_horizontal=bool(c["orbbec_flip_horizontal"]),
                orbbec_use_transformed_depth=bool(c["orbbec_use_transformed_depth"]),
                orbbec_hand_swap=str(c["orbbec_hand_swap"]),
            ),
            morph=MorphDefaults(
                target_alpha=float(m["target_alpha"]),
                morph_world_scale=float(m["morph_world_scale"]),
                mode_rot_freeze_latch=int(m["mode_rot_freeze_latch"]),
                mode_vis_min=float(m["mode_vis_min"]),
                open_vis_min=float(m["open_vis_min"]),
            ),
            sim=SimDefaults(
                drone_model=str(s["drone_model"]),
                control_freq_hz=float(s.get("control_freq_hz", 10.0)),
                max_sim_substeps_per_frame=int(s["max_sim_substeps_per_frame"]),
                ground_z=float(s["ground_z"]),
            ),
            prearm=PrearmDefaults(
                prearm_hover_z=float(p["prearm_hover_z"]),
                prearm_takeoff_z=float(p["prearm_takeoff_z"]),
            ),
            left_hand_pose=LeftHandPoseDefaults(
                left_swarm_pose=bool(l["left_swarm_pose"]),
                left_rot_direct_follow=bool(l["left_rot_direct_follow"]),
                left_swarm_depth_frame_motion=bool(l["left_swarm_depth_frame_motion"]),
                left_pose_debug=bool(l["left_pose_debug"]),
                left_pose_debug_every=int(l["left_pose_debug_every"]),
                left_pose_frame_viz=bool(l["left_pose_frame_viz"]),
                left_unwind_s=float(l["left_unwind_s"]),
                left_cam_preset=str(l["left_cam_preset"]),
                left_world_frame=str(l["left_world_frame"]),
                left_cam_y_to_world_z=float(l["left_cam_y_to_world_z"]),
                left_palm_basis=str(l["left_palm_basis"]),
                left_axis_sign=tuple(float(v) for v in l["left_axis_sign"]),
                left_trans_scale_mm=float(l["left_trans_scale_mm"]),
                left_max_offset_m=float(l["left_max_offset_m"]),
                left_lost_decay=float(l["left_lost_decay"]),
                left_rot_scale=float(l["left_rot_scale"]),
                left_rot_gain=float(l["left_rot_gain"]),
                left_rot_gate_rad=float(l["left_rot_gate_rad"]),
                left_max_rot_rad=float(l["left_max_rot_rad"]),
                left_rot_trans_tau_mm=float(l["left_rot_trans_tau_mm"]),
                left_rot_world_z_scale=float(l["left_rot_world_z_scale"]),
                left_plane_rot_scale_mul=float(l["left_plane_rot_scale_mul"]),
                left_palm_depth_outlier_z_mm=float(l["left_palm_depth_outlier_z_mm"]),
                left_palm_depth_outlier_lat_ratio=float(l["left_palm_depth_outlier_lat_ratio"]),
                left_palm_center_depth_ema=float(l["left_palm_center_depth_ema"]),
                axis_trans_deadzone_m=float(l["axis_trans_deadzone_m"]),
                axis_rot_deadzone_rad=float(l["axis_rot_deadzone_rad"]),
                axis_trans_on_m=float(l["axis_trans_on_m"]),
                axis_rot_on_rad=float(l["axis_rot_on_rad"]),
                axis_trans_rot_coupling=float(l["axis_trans_rot_coupling"]),
                left_dual_webcam_rot=bool(l["left_dual_webcam_rot"]),
                left_rot_webcam_vis_thresh=float(l["left_rot_webcam_vis_thresh"]),
                left_rot_webcam_index=int(l["left_rot_webcam_index"]),
            ),
        )


def default_online_defaults_path() -> Path:
    """Bundled ``config/online_defaults.yaml``."""
    return Path(__file__).resolve().parents[3] / "config" / "online_defaults.yaml"


ONLINE_DEFAULTS = OnlineDefaults.load()

__all__ = [
    "CameraDefaults",
    "DisplayDefaults",
    "LeftHandPoseDefaults",
    "MorphDefaults",
    "ONLINE_DEFAULTS",
    "OnlineDefaults",
    "PrearmDefaults",
    "SimDefaults",
    "default_online_defaults_path",
]
