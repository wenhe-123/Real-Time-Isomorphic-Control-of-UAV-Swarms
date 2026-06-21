"""Load production online control defaults from ``config/online_defaults.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_YAML_KEY_TO_ATTR: dict[str, str] = {
    "left_pose_frame_viz_every": "_DEFAULT_LEFT_POSE_FRAME_VIZ_EVERY",
    "webcam_rot_stride": "_DEFAULT_WEBCAM_ROT_STRIDE",
    "trail_draw_every_frames": "_TRAIL_DRAW_EVERY_FRAMES",
    "led_apply_every_frames": "_LED_APPLY_EVERY_FRAMES",
    "sim_render_every": "_DEFAULT_SIM_RENDER_EVERY",
    "orbbec_fps": "_ONLINE_ORBBEC_FPS",
    "mp_input_scale": "_ONLINE_MP_INPUT_SCALE",
    "mp_detect_every": "_ONLINE_MP_DETECT_EVERY",
    "online_imshow_every": "_DEFAULT_ONLINE_IMSHOW_EVERY",
    "target_alpha": "_DEFAULT_TARGET_ALPHA",
    "drone_model": "_DEFAULT_DRONE_MODEL",
    "min_separation_m": "_DEFAULT_MIN_SEPARATION_M",
    "morph_world_scale": "_DEFAULT_MORPH_WORLD_SCALE",
    "mode_rot_freeze_latch": "_ONLINE_MODE_ROT_FREEZE_LATCH",
    "mode_vis_min": "_DEFAULT_MODE_VIS_MIN",
    "open_vis_min": "_DEFAULT_OPEN_VIS_MIN",
    "trail_buffer_maxlen": "_TRAIL_BUFFER_MAXLEN",
    "max_sim_substeps_per_frame": "_ONLINE_MAX_SIM_SUBSTEPS_PER_FRAME",
    "raw_target_ema": "_DEFAULT_RAW_TARGET_EMA",
    "max_target_step_m": "_DEFAULT_MAX_TARGET_STEP_M",
    "open_jump_reset": "_DEFAULT_OPEN_JUMP_RESET",
    "ground_z": "_DEFAULT_GROUND_Z",
    "prearm_hover_z": "_DEFAULT_PREARM_HOVER_Z",
    "prearm_takeoff_z": "_DEFAULT_PREARM_TAKEOFF_Z",
    "left_axis_sign": "_DEFAULT_LEFT_AXIS_SIGN",
    "left_trans_scale_mm": "_DEFAULT_LEFT_TRANS_SCALE_MM",
    "left_rot_scale": "_DEFAULT_LEFT_ROT_SCALE",
    "left_trans_ema": "_DEFAULT_LEFT_TRANS_EMA",
    "left_rot_ema": "_DEFAULT_LEFT_ROT_EMA",
    "left_max_offset_m": "_DEFAULT_LEFT_MAX_OFFSET_M",
    "left_lost_decay": "_DEFAULT_LEFT_LOST_DECAY",
    "left_max_rot_rad": "_DEFAULT_LEFT_MAX_ROT_RAD",
    "left_unwind_s": "_DEFAULT_LEFT_UNWIND_S",
    "left_palm_depth_outlier_z_mm": "_DEFAULT_LEFT_PALM_DEPTH_OUTLIER_Z_MM",
    "left_palm_depth_outlier_lat_ratio": "_DEFAULT_LEFT_PALM_DEPTH_OUTLIER_LAT_RATIO",
    "left_palm_center_depth_ema": "_DEFAULT_LEFT_PALM_CENTER_DEPTH_EMA",
    "left_rot_gate_rad": "_DEFAULT_LEFT_ROT_GATE_RAD",
    "left_yaw_min_horiz": "_DEFAULT_LEFT_YAW_MIN_HORIZ",
    "left_rot_gain": "_DEFAULT_LEFT_ROT_GAIN",
    "left_rot_trans_tau_mm": "_DEFAULT_LEFT_ROT_TRANS_TAU_MM",
    "left_rot_world_z_scale": "_DEFAULT_LEFT_ROT_WORLD_Z_SCALE",
    "left_cam_y_to_world_z": "_DEFAULT_LEFT_CAM_Y_TO_WORLD_Z",
    "left_cam_preset": "_DEFAULT_LEFT_CAM_PRESET",
    "left_world_frame": "_DEFAULT_LEFT_WORLD_FRAME",
    "axis_trans_deadzone_m": "_DEFAULT_AXIS_TRANS_DEADZONE_M",
    "axis_rot_deadzone_rad": "_DEFAULT_AXIS_ROT_DEADZONE_RAD",
    "axis_trans_on_m": "_DEFAULT_AXIS_TRANS_ON_M",
    "axis_rot_on_rad": "_DEFAULT_AXIS_ROT_ON_RAD",
    "left_palm_basis": "_DEFAULT_LEFT_PALM_BASIS",
    "left_plane_rot_scale_mul": "_DEFAULT_LEFT_PLANE_ROT_SCALE_MUL",
    "left_rot_pivot": "_DEFAULT_LEFT_ROT_PIVOT",
    "left_dual_webcam_rot": "_DEFAULT_LEFT_DUAL_WEBCAM_ROT",
    "left_rot_webcam_vis_thresh": "_DEFAULT_LEFT_ROT_WEBCAM_VIS_THRESH",
    "left_rot_webcam_index": "_DEFAULT_LEFT_ROT_WEBCAM_INDEX",
    "wcam_preview_window": "_WCAM_PREVIEW_WINDOW",
    "orbbec_flip_horizontal": "_DEFAULT_ORBBEC_FLIP_HORIZONTAL",
    "orbbec_use_transformed_depth": "_DEFAULT_ORBBEC_USE_TRANSFORMED_DEPTH",
    "orbbec_hand_swap": "_DEFAULT_ORBBEC_HAND_SWAP",
}


def default_online_defaults_path() -> Path:
    """Bundled ``config/online_defaults.yaml``."""
    return Path(__file__).resolve().parents[3] / "config" / "online_defaults.yaml"


def load_online_defaults_yaml(path: Path | None = None) -> dict[str, Any]:
    settings_path = default_online_defaults_path() if path is None else Path(path)
    with open(settings_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"online defaults must be a mapping, got {type(raw).__name__}")
    return raw


def _coerce_value(key: str, value: Any) -> Any:
    if key == "left_axis_sign":
        return tuple(float(v) for v in value)
    if key == "orbbec_fps":
        return str(value)
    return value


def _apply_online_defaults(raw: dict[str, Any]) -> None:
    unknown = sorted(set(raw) - set(_YAML_KEY_TO_ATTR))
    if unknown:
        raise KeyError(f"unknown online_defaults.yaml keys: {', '.join(unknown)}")
    missing = sorted(set(_YAML_KEY_TO_ATTR) - set(raw))
    if missing:
        raise KeyError(f"missing online_defaults.yaml keys: {', '.join(missing)}")
    for yaml_key, attr in _YAML_KEY_TO_ATTR.items():
        globals()[attr] = _coerce_value(yaml_key, raw[yaml_key])


_apply_online_defaults(load_online_defaults_yaml())

__all__ = list(_YAML_KEY_TO_ATTR.values())
