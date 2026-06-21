"""Mutable runtime state for left-hand swarm rigid pose."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from functions.mode_switch.hand_constants import WRIST_ID
from functions.swarm_motion.left_pose_config import DEFAULT_LEFT_PALM_BASIS
from functions.swarm_motion.left_palm_geom import _enforce_thumb_positive_x, palm_orthonormal_basis


class LeftSwarmPoseState:
    """Tracks reference palm frame and outputs smoothed world offset + full rotation."""

    enabled: bool = True
    initialized: bool = False
    ref_basis: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    ema_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ema_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Smooth return-to-morph-frame after disarm (seconds); 0 = not unwinding
    unwind_end_t: float = 0.0
    unwind_duration: float = 0.0
    unwind_off0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    unwind_rv0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Previous palm center (camera mm), used for frame-to-frame translation delta.
    prev_palm_mm: np.ndarray | None = None
    #: When set (``camera_at_arm``), maps gated camera mm → sim m; frozen at press-0, not updated per frame.
    frozen_M_rot: np.ndarray | None = None
    frozen_M_trans: np.ndarray | None = None
    frozen_cam_preset: str = ""
    #: Swarm XYZ at arm (sim m); reserved for arm-time diagnostics / traces.
    ref_swarm_targets: np.ndarray | None = None
    #: 2D/webcam palm basis at arm when dual-rotation fallback is enabled.
    ref_basis_image: np.ndarray | None = None
    #: Wrist (camera mm) at arm — debug / overlay only.
    ref_wrist_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Palm centroid (camera mm) at arm — translation origin.
    ref_palm_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Palm basis ΔR this frame (world rad); for HUD — may differ from applied ``rv_cmd``.
    last_rv_pose_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_rv_cmd_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_h_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_cam_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_cam_arm_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_h_raw_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_palm_center_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: EMA-smoothed palm center (camera mm) after depth outlier rejection.
    filtered_palm_mm: np.ndarray | None = None
    last_depth_outlier: bool = False
    last_depth_outlier_prev: bool = False
    last_wrist_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_mcp_center_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_mcp_valid_count: int = 0
    last_palm_center_color_px: tuple[int, int] | None = None
    last_palm_color_u: float | None = None
    last_palm_color_v: float | None = None
    last_rv_cam_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Previous Orbbec MP min visibility (dual-rot); used to hold morph mode when occluded.
    last_orbbec_vis_min: float = 1.0
    last_dual_rot_source: str = "depth"
    last_dual_vis_min: float = 1.0
    last_dual_vis_thresh: float = 0.0
    last_pose_rejected: bool = False
    last_reject_reason: str = ""
    #: Previous basis from the rotation source actually used (depth vs image/webcam).
    prev_rot_basis: np.ndarray | None = None
    prev_rot_source: str = "depth"
    #: Last classified motion: ``translate`` | ``rotate`` | ``none`` (mode-switch freeze).
    last_axis_motion: str = "none"
    last_rot_source: str = "depth"
    last_rot_blend_w: float = 0.0
    last_trans_blend_w: float = 0.0

    def reset_to_current(
        self,
        h: np.ndarray,
        *,
        palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
        sim_from_cam: np.ndarray | None = None,
        sim_trans_from_cam: np.ndarray | None = None,
        cam_preset_label: str = "",
        ref_swarm_targets: np.ndarray | None = None,
        ref_basis_image: np.ndarray | None = None,
        palm_center_override: np.ndarray | None = None,
        palm_pose: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        wrist_mm = np.asarray(h[WRIST_ID, :3], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(wrist_mm)):
            return False
        if palm_pose is not None:
            pc, B = palm_pose
            pc = np.asarray(pc, dtype=np.float64).reshape(3)
            B = np.asarray(B, dtype=np.float64).reshape(3, 3)
        else:
            out = palm_orthonormal_basis(
                h, palm_basis=palm_basis, palm_center_override=palm_center_override
            )
            if out is None:
                return False
            origin, B = out
            pc = np.asarray(origin, dtype=np.float64).reshape(3)
        B = _enforce_thumb_positive_x(B, h, wrist_mm)
        if palm_center_override is not None:
            pc_ov = np.asarray(palm_center_override, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(pc_ov)):
                pc = pc_ov
        self.ref_wrist_mm = wrist_mm.copy()
        self.ref_palm_center = np.asarray(pc, dtype=np.float64).reshape(3).copy()
        self.ref_basis = B.copy()
        self.initialized = True
        self.ema_offset[:] = 0.0
        self.ema_rotvec[:] = 0.0
        self.prev_palm_mm = pc.copy()
        self.prev_rot_basis = B.copy()
        self.prev_rot_source = "depth"
        self.last_rot_source = "depth"
        self.last_palm_center_mm = pc.copy()
        self.last_delta_cam_mm[:] = 0.0
        self.last_delta_cam_arm_mm[:] = 0.0
        self.last_delta_h_world[:] = 0.0
        self.last_delta_h_raw_m[:] = 0.0
        self.last_rv_pose_world[:] = 0.0
        self.last_rv_cmd_world[:] = 0.0
        if sim_from_cam is not None:
            self.frozen_M_rot = np.asarray(sim_from_cam, dtype=np.float64).reshape(3, 3).copy()
            if sim_trans_from_cam is not None:
                self.frozen_M_trans = np.asarray(sim_trans_from_cam, dtype=np.float64).reshape(3, 3).copy()
            else:
                self.frozen_M_trans = self.frozen_M_rot.copy()
            self.frozen_cam_preset = str(cam_preset_label)
        else:
            self.frozen_M_rot = None
            self.frozen_M_trans = None
            self.frozen_cam_preset = ""
        if ref_swarm_targets is not None:
            rs = np.asarray(ref_swarm_targets, dtype=np.float64)
            if rs.ndim == 2 and rs.shape[1] >= 3:
                self.ref_swarm_targets = rs.astype(np.float32, copy=True)
        else:
            self.ref_swarm_targets = None
        if ref_basis_image is not None:
            self.ref_basis_image = np.asarray(ref_basis_image, dtype=np.float64).reshape(3, 3).copy()
        else:
            self.ref_basis_image = None
        self.filtered_palm_mm = np.asarray(pc, dtype=np.float64).reshape(3).copy()
        self.last_depth_outlier = False
        return True

    def is_unwinding(self) -> bool:
        return float(self.unwind_end_t) > 0.0 and time.monotonic() < float(self.unwind_end_t)

    def begin_unwind(self, duration_s: float) -> None:
        """Fade rigid offset/rotation to identity over ``duration_s`` (smoothstep)."""
        d = float(max(duration_s, 1e-3))
        self.unwind_off0 = np.asarray(self.ema_offset, dtype=np.float64).copy()
        self.unwind_rv0 = np.asarray(self.ema_rotvec, dtype=np.float64).copy()
        self.unwind_duration = d
        self.unwind_end_t = time.monotonic() + d

    def cancel_unwind(self) -> None:
        """Abort smooth restore (e.g. user re-arms); clears offset like disarm."""
        self.unwind_end_t = 0.0
        self.unwind_duration = 0.0
        self.unwind_off0[:] = 0.0
        self.unwind_rv0[:] = 0.0
        self.initialized = False
        self.ema_offset[:] = 0.0
        self.ema_rotvec[:] = 0.0
        self.prev_palm_mm = None
        self.prev_rot_basis = None
        self.prev_rot_source = "depth"
        self.last_axis_motion = "none"
        self.last_rot_source = "depth"
        self.last_dual_rot_source = "depth"
        self.last_dual_vis_min = 1.0
        self.last_dual_vis_thresh = 0.0
        self.last_rot_blend_w = 0.0
        self.last_trans_blend_w = 0.0
        self.frozen_M_rot = None
        self.frozen_M_trans = None
        self.frozen_cam_preset = ""
        self.ref_swarm_targets = None
        self.ref_basis_image = None
