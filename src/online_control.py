"""Online Crazyflow control using the same morph target logic as webcam_main.

The first test mode intentionally keeps the target fixed at mode=1, open=1.0.
This validates Crazyflow startup and the initial target mapping before wiring in
live MediaPipe hand updates.
"""

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable

# Prevent OpenCV-Qt font lookup spam that can flood stdout/stall UI on some envs.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_UI_BACKEND", "GTK3")

import cv2
import jax.numpy as jnp
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import runtime.hand_tracking_orbbec as ob

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from shared.common_utils import resolve_model_path
from shared.hand_constants import MCP_IDS, WRIST_ID
from shared.hand_draw_utils import draw_all_hands
from shared.mode_gesture_utils import classify_mode_from_fingers as classify_mode_from_fingers_common
from shared.modes_runtime import (
    ModeState,
    RightHandState,
    process_left_mode,
    process_right_open,
    update_mode_state as shared_update_mode_state,
    update_open_state as shared_update_open_state,
)
from shared.morph_lp_plot import MORPH_PLANE_RADIUS_A, MORPH_PLANE_RADIUS_B, mode_epsilon_pair, update_3d_plot_lp
from shared.mp_hand_utils import (
    extract_world_points_mm_result,
    find_left_right_indices,
    orbbec_resolve_swap_mp_hands,
)
from shared.morph_led_materials import apply_morph_led_theme
from shared.left_hand_swarm_pose import (
    LEFT_PALM_BASIS_PRESETS,
    LeftSwarmPoseState,
    R_to_rotvec,
    apply_rigid_to_targets,
    build_sim_from_cam_matrices,
    disarm_left_swarm_pose,
    left_hand_pose_matrix_depth_mm,
    left_cam_preset_rotation,
    make_cam_translation_matrix,
    mp_hand_visibility_scores,
    palm_basis_from_mp_image_plane,
    palm_basis_pair_indices,
    print_left_swarm_pose_debug,
    update_left_swarm_pose,
)
from shared.dual_view_utils import draw_hand_webcam, open_webcam_capture
from shared.left_hand_rotation_dual import _detect_webcam_hand, resolve_dual_left_rotation
from shared.left_pose_frame_viz import draw_left_pose_frame_overlay
from shared.online_input_keys import (
    OnlineKeyQueue,
    format_hotkey_install_hint,
    probe_global_hotkey_backends,
    process_online_control_keys,
    try_install_hotkey_dependencies,
)
from shared.trajectory_record import TrajectoryRecorder

# Matplotlib 3D: update every N camera frames (1 = every frame, smoother preview; 0 = off).
_DEFAULT_PLOT_EVERY_ONLINE = 4
_DEFAULT_LEFT_POSE_FRAME_VIZ_EVERY = 3
_DEFAULT_WEBCAM_ROT_STRIDE = 3
_TRAIL_DRAW_EVERY_FRAMES = 0
_LED_APPLY_EVERY_FRAMES = 3
_PLOT_PAUSE_SEC = 0.0
_DEFAULT_SIM_RENDER_EVERY = 1
_ONLINE_ORBBEC_FPS = "30"
_ONLINE_ORBBEC_USE_DEPTH_FUSION = True
_ONLINE_MP_INPUT_SCALE = 0.5
_DEFAULT_TARGET_ALPHA = 0.44
_DEFAULT_DRONE_MODEL = "cf21B_500"
_STATUS_PRINT_EVERY_SEC = 0
_DEFAULT_MIN_SEPARATION_M = 0.32
_DEFAULT_INPUT_BACKEND = "orbbec"
# Faster mode switching in online control: use fewer stable frames than runtime default.
_ONLINE_MODE_DEBOUNCE_FRAMES = 1
# Left-hand MP visibility min before morph mode (M1..M5) may change; blocks occlusion false counts.
_DEFAULT_MODE_VIS_MIN = 0.55
# Keep online-control open response aligned with webcam/orbbec runtime behavior.
_ONLINE_OPEN_SMOOTH = 0.18
_ONLINE_SNAP_SOFT_K = 0.34
_ONLINE_SNAP_SOFT_MAX_STEP = 0.16
# History length for MuJoCo trail polylines (smaller = shorter tail on screen).
_TRAIL_BUFFER_MAXLEN = 6
# Integrated loop: advance sim by wall-clock dt each frame and smooth targets per physics step
# so slow vision loops do not starve control (reduces collisions when the hand moves quickly).
_ONLINE_MAX_SIM_SUBSTEPS_PER_FRAME = 160
_DEFAULT_STATE_FREQ = 320
# EMA on morph targets before separation (0 = off). Reduces discrete jumps from gesture/vision.
_DEFAULT_RAW_TARGET_EMA = 0.52
# Max |Δposition| per drone per physics substep toward blended target (m); 0 = no cap.
_DEFAULT_MAX_TARGET_STEP_M = 0.038
# When |Δopen| >= this after a frame's gesture update, snap smooth_target to safe_target (reduces plane↔sphere chase lag).
_DEFAULT_OPEN_JUMP_RESET = 0.34
# Left hand: rigid motion of whole swarm (translation + rotation about formation centroid)
# Pose uses MediaPipe world landmarks in mm (wrist is not fixed at origin unlike HAND_FRAME_SCALED).
# fwd_y: dz→sim Y (in/out). Flip Y so hand-toward-camera matches expected swarm direction.
# middle_thumb palm embed: +palm Z (toward camera) → world +Y forward; use +1 on Y.
# palm X→world X (lateral), palm Z→world Y (fwd/back), palm Y→world Z (up). Default Z=-1 matches camera Y-down.
_DEFAULT_LEFT_AXIS_SIGN: tuple[float, float, float] = (1.0, 1.0, -1.0)
# Palm/camera mm → world meters (× on palm components before axis_sign). ~15 mm ≈ 0.18 m.
_DEFAULT_LEFT_TRANS_SCALE_MM = 0.012
_DEFAULT_LEFT_ROT_SCALE = 1.0
_DEFAULT_LEFT_TRANS_EMA = 0.42
_DEFAULT_LEFT_ROT_EMA = 0.28
_DEFAULT_LEFT_MAX_OFFSET_M = 1.8

_DEFAULT_LEFT_MAX_ROT_RAD = 3.14
_DEFAULT_LEFT_UNWIND_S = 2.6
# Ignore |rotation vector| below this (rad axis-angle) after rot_scale*rot_gain — higher = calmer tilt.
_DEFAULT_LEFT_ROT_GATE_RAD = 0.07
_DEFAULT_LEFT_YAW_MIN_HORIZ = 0.17
_DEFAULT_LEFT_ROT_GAIN = 1.00
# While wrist moves: rotation cmd *= exp(-Δmm / tau). Larger tau = less suppression during pans.
_DEFAULT_LEFT_ROT_TRANS_TAU_MM = 32.0
# Damp world-up component of axis–angle (formation spin about global Z).
# 1.0: do not damp ω_world Z (0.12 made palm twist ≈ dead with ``camera`` / fwd_y).
_DEFAULT_LEFT_ROT_WORLD_Z_SCALE = 1.0
# Image dy → sim altitude (fwd_y third row). 1.0 = full −cam Y → sim Z.
_DEFAULT_LEFT_CAM_Y_TO_WORLD_Z = 1.0
# Z-up sim embedding: X=lateral, Y=in-out, Z=altitude. ``camera`` (identity) maps dy→sim Y (wrong).
_DEFAULT_LEFT_CAM_PRESET = "fwd_y"
# ``camera_at_arm``: press 0 locks depth-camera → sim map; baseline wrist + palm = origin.
_DEFAULT_LEFT_WORLD_FRAME = "camera_at_arm"
_DEFAULT_LEFT_CONTROL_STYLE = "axis_locked"
_DEFAULT_AXIS_TRANS_DEADZONE_M = 0.006
_DEFAULT_AXIS_ROT_DEADZONE_RAD = 0.016
_DEFAULT_AXIS_CAM_DEADZONE_XYZ_MM = (5.0, 5.0, 14.0)
_DEFAULT_AXIS_CAM_SNAP_MIN_RATIO = 1.12
_DEFAULT_AXIS_TRANS_ON_M = 0.006
_DEFAULT_AXIS_ROT_ON_RAD = 0.010
_DEFAULT_AXIS_ROT_EXCL_RATIO = 1.32
_DEFAULT_AXIS_TRANS_EXCL_RATIO = 1.85
_DEFAULT_AXIS_ROT_BOOST = 2.0
_DEFAULT_AXIS_FORWARD_BOOST = 1.38
_DEFAULT_AXIS_LATERAL_MIN_DX_MM = 7.0
_DEFAULT_AXIS_FORWARD_MIN_DZ_MM = 3.0
# When |dz| dominates in-plane motion: translation uses only optical-axis mm (reduces bogus lateral
# when hand moves mainly toward/away from camera). 0 ratio = off.
_DEFAULT_LEFT_DEPTH_DOM_RATIO = 1.82
_DEFAULT_LEFT_DEPTH_DOM_MIN_MM = 4.0
# Palm rotation vs translation: small intrinsic angle + wrist moving → zero rotation that frame.
# Off by default: coex zeroed rotation whenever the wrist moved (felt “rotation dead”).
_DEFAULT_LEFT_ROT_COEX_TRANS_MIN_MM = 0.0
_DEFAULT_LEFT_ROT_COEX_MAX_ANGLE_DEG = 7.5
# Wider MCP baseline than index/middle for palm frame (see shared.left_hand_swarm_pose).
_DEFAULT_LEFT_PALM_BASIS = "middle_thumb"
# Lateral pan: when |dx| dominates hypot(dy,dz), translation uses only [dx,0,0] (camera mm).
_DEFAULT_LEFT_TRANS_LATERAL_DOM_RATIO = 0.92
_DEFAULT_LEFT_TRANS_LATERAL_DOM_MIN_MM = 0.32
# Zero dz when image-plane motion dominates |dz| (before depth_dom).
_DEFAULT_LEFT_TRANS_XY_DOM_STRIP_DZ_RATIO = 0.88
_DEFAULT_LEFT_TRANS_XY_DOM_STRIP_DZ_MIN_H_MM = 0.22
# Slow lateral: per-frame |dx| can be small; still strip dz when |dx| beats |dz|.
_DEFAULT_LEFT_TRANS_DX_DOM_STRIP_DZ_RATIO = 0.88
_DEFAULT_LEFT_TRANS_DX_DOM_STRIP_DZ_MIN_DX_MM = 0.22
# When |dy| dominates hypot(dx,dz): zero camera dz for translation (reduces vertical→forward/back leak).
_DEFAULT_LEFT_VERTICAL_DOM_RATIO = 1.38
_DEFAULT_LEFT_VERTICAL_DOM_MIN_MM = 2.0
# Optional: skip vertical dz-strip when |dz| dominates hypot(dx,dy) (0 = off by default).
_DEFAULT_LEFT_VERTICAL_OPTICAL_PRESERVE_RATIO = 0.0
_DEFAULT_LEFT_VERTICAL_OPTICAL_PRESERVE_MIN_MM = 0.0
# Multiplier on camera dz after translation gates (fwd_y → world Y); lower = less noisy near/far.
_DEFAULT_LEFT_TRANS_BOOST_OPTICAL = 1.08
# When hypot(dx,dy) dominates |dz| in translation mm: zero palm rotation that frame (plane pan).
# Off by default: planar_dom zeroed rotation on most pans (including small wrist motion).
_DEFAULT_LEFT_ROT_PLANAR_DOM_RATIO = 0.0
_DEFAULT_LEFT_ROT_PLANAR_DOM_MIN_MM = 1.15
# Palm ≈ image plane (|palm normal · optical|) + twist about +Z_cam → world ω ≈ [0, θ, 0] (fwd_y depth path).
# Off by default: palm twist injects world-Y rotation and skips planar/coex/gate — noisy during pans.
_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_WORLD_Y = False
_DEFAULT_LEFT_ROT_PALM_FACE_COS_ALIGN_MIN = 0.78
_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_DOM_RATIO = 1.0
_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_MIN_RAD = 0.006
_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_WORLD_Y_SIGN = 1.0
# Right-hand open snap_state == "plane" (spread→plane morph branch): extra left rotation attenuation.
_DEFAULT_LEFT_PLANE_ROT_SCALE_MUL = 0.50
# centroid: whole formation rotates about its centroid (整体自转). per_drone: each slot pivots separately.
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
# Gesture-to-mode parameters (same policy as webcam pipeline).
_MODE_COUNT_TIP_IDS = [8, 12, 16, 20, 4]
_MODE_EXTEND_MIN = 0.42
_MODE_TIER_GAP = 0.21
from shared.morph_shape_control import (
    LpShapePipelineState,
    advance_lp_shape_p,
    index_mcp_tip_segment_norm,
)
from shared.morph_renderers import (
    init_fixed_surface_points,
    mapped_fixed_surface_points,
    prompt_and_init_fixed_surface_points,
)


@dataclass
class ScaleConfig:
    xy_radius: float
    z_center: float
    z_amplitude: float
    z_min: float
    z_max: float
    reference_xy_extent_mm: float
    reference_z_extent_mm: float
    #: Extra factor on Z mm→m (1.0 matches topo plot when reference_z_extent_mm == reference_xy_extent_mm).
    z_mm_scale: float = 1.0


class LiveTargetState:
    """Thread-safe latest Crazyflow target from webcam recognition."""

    def __init__(self, initial_target: np.ndarray):
        self._lock = Lock()
        self._target = np.asarray(initial_target, dtype=np.float32).copy()
        self.mode = 1
        self.open_alpha = 1.0

    def get(self) -> np.ndarray:
        with self._lock:
            return self._target.copy()

    def set(self, target: np.ndarray, mode: int, open_alpha: float) -> None:
        with self._lock:
            self._target = np.asarray(target, dtype=np.float32).copy()
            self.mode = int(mode)
            self.open_alpha = float(open_alpha)


def summarize_target_workspace(points_m: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (max_xy_radius_m, xyz_min, xyz_max) for a target set in meters."""
    pts = np.asarray(points_m, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        z = np.zeros((3,), dtype=np.float32)
        return 0.0, z, z
    xy_r = np.linalg.norm(pts[:, :2], axis=1)
    return float(np.max(xy_r)), np.min(pts[:, :3], axis=0), np.max(pts[:, :3], axis=0)


def init_3d_plot(plot_every_n: int, title: str):
    """Create 3D figure/axes only when enabled."""
    plot_enabled = int(plot_every_n) > 0
    if not plot_enabled:
        return False, None, None, None
    plt.ion()
    fig = plt.figure(title)
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")
    try:
        fig.tight_layout()
    except Exception:
        pass
    plt.show(block=False)
    return True, fig, ax_hand, ax_topo


def close_3d_plot(fig) -> None:
    if fig is None:
        return
    plt.ioff()
    plt.close(fig)


def refresh_3d_plot_nonblocking(fig) -> None:
    """Refresh matplotlib canvas without blocking in plt.pause()."""
    if fig is None:
        return
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        pass


def _clear_formation_rigid_debug(ax_topo) -> None:
    for art in getattr(ax_topo, "_iso_swarm_debug_artists", None) or []:
        try:
            art.remove()
        except Exception:
            pass
    ax_topo._iso_swarm_debug_artists = []
    tx = getattr(ax_topo, "_iso_swarm_debug_text", None)
    if tx is not None:
        try:
            tx.remove()
        except Exception:
            pass
    ax_topo._iso_swarm_debug_text = None


def draw_formation_rigid_debug_on_topo(
    ax_topo,
    p_before_m: np.ndarray,
    p_after_m: np.ndarray,
    *,
    off_m: np.ndarray | None = None,
    R_pose: np.ndarray | None = None,
) -> None:
    """Overlay morph-only vs L-move targets on the topo axis (scaled to ~mm span for visibility).

    Blue dots: morph-only targets before left-hand rigid; magenta: after ``apply_rigid_to_targets``.
    Used only when ``--formation-rigid-3d-debug`` is passed (default: overlay off).
    """
    _clear_formation_rigid_debug(ax_topo)
    pb = np.asarray(p_before_m, dtype=np.float64)
    pa = np.asarray(p_after_m, dtype=np.float64)
    if pb.ndim != 2 or pa.ndim != 2 or pb.shape != pa.shape or pb.shape[1] < 3:
        return
    c = np.mean(pb[:, :3], axis=0)
    d0 = pb[:, :3] - c
    d1 = pa[:, :3] - c
    r = max(
        float(np.max(np.linalg.norm(d0, axis=1))),
        float(np.max(np.linalg.norm(d1, axis=1))),
        0.12,
    )
    s = 85.0 / r
    q0 = d0 * s
    q1 = d1 * s
    art0 = ax_topo.scatter(
        q0[:, 0],
        q0[:, 1],
        q0[:, 2],
        c="tab:blue",
        s=22,
        alpha=0.88,
        depthshade=False,
        label="morph cmd",
    )
    art1 = ax_topo.scatter(
        q1[:, 0],
        q1[:, 1],
        q1[:, 2],
        c="tab:magenta",
        s=40,
        alpha=0.92,
        marker="^",
        depthshade=False,
        edgecolors="k",
        linewidths=0.45,
        label="+ L-move",
    )
    ax_topo._iso_swarm_debug_artists = [art0, art1]
    if off_m is not None and R_pose is not None:
        offv = np.asarray(off_m, dtype=np.float64).ravel()
        R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
        rv = R_to_rotvec(R)
        ang = float(np.linalg.norm(rv))
        if ang > 1e-8:
            ax = rv / ang
            rot_s = f"rot≈{float(np.degrees(ang)):+.1f}°  n̂≈({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"
        else:
            rot_s = "rot≈0°"
        t = (
            f"L-swarm: |Δ|={float(np.linalg.norm(offv)):.3f}m  "
            f"Δ=({offv[0]:+.2f},{offv[1]:+.2f},{offv[2]:+.2f})  {rot_s}"
        )
        try:
            tx = ax_topo.text2D(
                0.02,
                0.98,
                t,
                transform=ax_topo.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="tab:purple",
            )
            ax_topo._iso_swarm_debug_text = tx
        except Exception:
            ax_topo._iso_swarm_debug_text = None


def update_online_3d_plot(
    *,
    ax_hand,
    ax_topo,
    hands_3d,
    morph_mode: int,
    open_out: float | None,
    lp_shape: "LpShapePipelineState",
    topo_radius_override_mm: float | None = None,
):
    """Shared 3D update config aligned with Orbbec runtime.

    By default the topo morph uses the same ``r_vis = max(140, 2.2 * r_topo)`` rule as
    ``hand_tracking_orbbec`` (readable point spread). Set ``topo_radius_override_mm`` only if
    you need a fixed millimetre radius for the plot; plane-state cache keys already include
    ``R`` so mixed radii no longer corrupt ``mapped_fixed_surface_points``.
    """
    return update_3d_plot_lp(
        ax_hand,
        ax_topo,
        hands_3d,
        morph_mode=morph_mode,
        morph_alpha_smoothed=open_out,
        control_label="online open+p",
        analyze_hand_topology_fn=ob.analyze_hand_topology,
        clamp01_fn=ob.clamp01,
        shape_normalized=True,
        hand_frame=ob.HAND_FRAME_SCALED,
        hand_3d_source=ob.HAND_3D_SOURCE_MP,
        hand_frame_palm_plane=ob.HAND_FRAME_PALM_PLANE,
        norm_axis_halflim=ob.NORM_AXIS_HALFLIM,
        morph_axis_lim_mm=ob.MORPH_AXIS_LIM_MM,
        mode_shape_t=lp_shape.left_shape_t_ema,
        epsilon_pair_display=lp_shape.epsilon_pair_display,
        topo_radius_override_mm=topo_radius_override_mm,
    )


def update_live_target_from_state(
    *,
    live_target: "LiveTargetState",
    mode_state: ModeState,
    right_state: RightHandState,
    lp_shape: "LpShapePipelineState",
    scale: "ScaleConfig",
    radius_mm: float,
    open_out: float | None,
) -> None:
    open_v = float(
        open_out
        if open_out is not None
        else (
            float(right_state.last_open_out)
            if right_state.last_open_out is not None
            else float(live_target.open_alpha)
        )
    )
    epsilon1, epsilon2 = mode_epsilon_pair(
        int(mode_state.morph_mode),
        lp_shape.left_shape_t_ema,
    )
    points_mm = mapped_fixed_surface_points(
        radius=float(radius_mm),
        open_alpha=open_v,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        plane_radius_a=MORPH_PLANE_RADIUS_A,
        plane_radius_b=MORPH_PLANE_RADIUS_B,
        morph_mode=int(mode_state.morph_mode),
    )
    target = normalize_morph_points(points_mm, scale)
    # Guard against pathological collapsed frames (all points near origin).
    xy_r_max, _xyz_min, _xyz_max = summarize_target_workspace(target)
    if not np.isfinite(xy_r_max) or xy_r_max < 0.35:
        return
    live_target.set(target, mode=int(mode_state.morph_mode), open_alpha=open_v)


def classify_mode_from_fingers(hand_points):
    """Compatibility wrapper using shared mode-gesture classifier."""
    return classify_mode_from_fingers_common(
        hand_points,
        mode_count_tip_ids=_MODE_COUNT_TIP_IDS,
        mode_extend_min=_MODE_EXTEND_MIN,
        mode_tier_gap=_MODE_TIER_GAP,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
    )


def closest_pair(points: np.ndarray) -> tuple[float, int, int]:
    """Smallest pairwise distance among morph target points (normalized Crazyflow coords).

    Used for **spacing / near-collision diagnostics**: which two drones (point IDs) are
    closest and how far apart they are. Startup prints initial spacing; integrated loop
    logs ``target_min=(i,j)`` about once per second when debugging formation quality.
    Vectorized upper triangle, O(n²) memory.
    """
    p = np.asarray(points, dtype=np.float64)
    n = int(p.shape[0])
    if n < 2:
        return float("inf"), -1, -1
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        return float("inf"), -1, -1
    flat = d2[iu, ju]
    k = int(np.argmin(flat))
    return float(np.sqrt(float(flat[k]))), int(iu[k]), int(ju[k])


def normalize_morph_points(points_mm: np.ndarray, scale: ScaleConfig) -> np.ndarray:
    """Map morph-renderer millimeter targets into the Crazyflow workspace.

    Uses the **same mm semantics as the Matplotlib topo plot** (isotropic axes in mm):
    X and Y scale by ``xy_radius / reference_xy_extent_mm``; Z scales by
    ``xy_radius / reference_z_extent_mm`` (default equals XY reference so proportions match
    the 3D sample view). ``z_mm_scale`` applies only as an extra Z factor. Overall size is
    controlled by ``--xy-radius``; ``s_fit`` scales XY and Z **together** so the formation
    fits ``[z_min, z_max]`` without non-uniform clipping.
    """
    pts = np.asarray(points_mm, dtype=np.float32)
    xy_den = max(float(scale.reference_xy_extent_mm), 1.0)
    z_den = max(float(scale.reference_z_extent_mm), 1.0)
    xy_s0 = float(scale.xy_radius) / xy_den
    z_s0 = (float(scale.xy_radius) / z_den) * float(scale.z_mm_scale)
    raw_z = pts[:, 2] * z_s0
    max_d = float(np.max(raw_z)) if raw_z.size else 0.0
    min_d = float(np.min(raw_z)) if raw_z.size else 0.0
    z_top = float(scale.z_max) - float(scale.z_center)
    z_bot = float(scale.z_center) - float(scale.z_min)
    margin = 5e-3
    s_up = (z_top - margin) / max(max_d, 1e-9) if max_d > 1e-9 else 1e9
    s_dn = (z_bot - margin) / max(-min_d, 1e-9) if min_d < -1e-9 else 1e9
    s_fit = float(min(1.0, s_up, s_dn))
    xy_s = xy_s0 * s_fit
    z_s = z_s0 * s_fit
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = pts[:, 0] * xy_s
    out[:, 1] = pts[:, 1] * xy_s
    out[:, 2] = np.clip(
        float(scale.z_center) + pts[:, 2] * z_s,
        float(scale.z_min),
        float(scale.z_max),
    )
    return out


def debug_print_drone_targets(
    pts_m: np.ndarray,
    *,
    frame_idx: int,
    morph_mode: int,
    open_v: float | None,
    label: str,
    compare_to: np.ndarray | None = None,
) -> None:
    p = np.asarray(pts_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 3:
        print(f"[debug {label}] f={frame_idx} invalid shape {getattr(p, 'shape', None)}")
        return
    op = f"{float(open_v):.3f}" if open_v is not None else "-"
    z0, z1 = float(np.min(p[:, 2])), float(np.max(p[:, 2]))
    xy_r = np.linalg.norm(p[:, :2], axis=1)
    line = (
        f"[debug {label}] frame={frame_idx} M={int(morph_mode)} open={op} n={p.shape[0]} "
        f"z_span={z1 - z0:.3f}m z[{z0:.3f},{z1:.3f}] xy_r_max={float(np.max(xy_r)):.3f}m"
    )
    if compare_to is not None:
        c = np.asarray(compare_to, dtype=np.float64)
        if c.shape == p.shape:
            err = np.linalg.norm(p - c, axis=1)
            line += f" | vs_cmd mean={float(np.mean(err)):.3f}m max={float(np.max(err)):.3f}m"
    print(line)
    for i in range(p.shape[0]):
        print(f"  drone {i:2d}: x={p[i, 0]:8.4f} y={p[i, 1]:8.4f} z={p[i, 2]:8.4f}")


def clamp_targets_step(prev: np.ndarray, nxt: np.ndarray, max_step_m: float) -> np.ndarray:
    """Per-drone clamp: move from prev toward nxt by at most max_step_m (L2 per row)."""
    if max_step_m <= 0:
        return np.asarray(nxt, dtype=np.float32)
    p = np.asarray(prev, dtype=np.float64)
    x = np.asarray(nxt, dtype=np.float64)
    d = x - p
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_step_m) / np.maximum(dist, 1e-9))
    out = p + d * s
    return out.astype(np.float32)


def enforce_min_separation(points: np.ndarray, min_sep: float, *, iters: int = 6) -> np.ndarray:
    """Lightweight collision guard: iteratively repel pairs closer than ``min_sep``."""
    pts = np.asarray(points, dtype=np.float64).copy()
    n = int(pts.shape[0])
    if n < 2 or min_sep <= 0:
        return pts.astype(np.float32)
    ms = float(min_sep)
    for _ in range(max(1, int(iters))):
        disp = np.zeros_like(pts)
        moved = False
        for i in range(n - 1):
            d = pts[i + 1 :] - pts[i]
            dist = np.linalg.norm(d, axis=1)
            mask = dist < ms
            if not np.any(mask):
                continue
            moved = True
            idx = np.where(mask)[0]
            dist_m = np.maximum(dist[idx], 1e-6)
            dir_m = d[idx] / dist_m[:, None]
            push = 0.5 * (ms - dist_m)[:, None] * dir_m
            disp[i] -= np.sum(push, axis=0)
            disp[i + 1 + idx] += push
        pts += disp
        if not moved:
            break
    return pts.astype(np.float32)


def fixed_morph_points(
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
) -> np.ndarray:
    """Generate the same fixed indexed morph points used by webcam_main."""
    init_fixed_surface_points(point_count)
    epsilon1, epsilon2 = mode_epsilon_pair(int(morph_mode), shape_t)
    return mapped_fixed_surface_points(
        radius=float(radius_mm),
        open_alpha=float(open_alpha),
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        plane_radius_a=MORPH_PLANE_RADIUS_A,
        plane_radius_b=MORPH_PLANE_RADIUS_B,
        morph_mode=int(morph_mode),
    )


def make_initial_target_provider(
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
    scale: ScaleConfig,
) -> Callable[[], np.ndarray]:
    """Return a provider for the initial fixed mode/open target."""
    points_mm = fixed_morph_points(
        point_count=point_count,
        radius_mm=radius_mm,
        morph_mode=morph_mode,
        open_alpha=open_alpha,
        shape_t=shape_t,
    )
    target = normalize_morph_points(points_mm, scale)
    dist, i, j = closest_pair(target)
    print(
        f"Initial target: mode={morph_mode}, open={open_alpha:.2f}, n={point_count}, "
        f"radius_mm={radius_mm:.1f}"
    )
    xyz_min = np.min(points_mm, axis=0)
    xyz_max = np.max(points_mm, axis=0)
    print(
        "raw_mm_range="
        f"x[{xyz_min[0]:.1f},{xyz_max[0]:.1f}] "
        f"y[{xyz_min[1]:.1f},{xyz_max[1]:.1f}] "
        f"z[{xyz_min[2]:.1f},{xyz_max[2]:.1f}]"
    )
    print(f"Closest initial target spacing: pair=({i},{j}), dist={dist:.2f}m")

    def provider() -> np.ndarray:
        return target

    return provider


def make_initial_live_target(
    point_count: int,
    radius_mm: float,
    morph_mode: int,
    open_alpha: float,
    shape_t: float | None,
    scale: ScaleConfig,
) -> LiveTargetState:
    """Create a live target state initialized from mode/open defaults."""
    points_mm = fixed_morph_points(
        point_count=point_count,
        radius_mm=radius_mm,
        morph_mode=morph_mode,
        open_alpha=open_alpha,
        shape_t=shape_t,
    )
    target = normalize_morph_points(points_mm, scale)
    dist, i, j = closest_pair(target)
    print(
        f"Initial target: mode={morph_mode}, open={open_alpha:.2f}, n={point_count}, "
        f"radius_mm={radius_mm:.1f}"
    )
    xyz_min = np.min(points_mm, axis=0)
    xyz_max = np.max(points_mm, axis=0)
    print(
        "raw_mm_range="
        f"x[{xyz_min[0]:.1f},{xyz_max[0]:.1f}] "
        f"y[{xyz_min[1]:.1f},{xyz_max[1]:.1f}] "
        f"z[{xyz_min[2]:.1f},{xyz_max[2]:.1f}]"
    )
    print(f"Closest initial target spacing: pair=({i},{j}), dist={dist:.2f}m")
    state = LiveTargetState(target)
    state.mode = int(morph_mode)
    state.open_alpha = float(open_alpha)
    r_xy, xyz_min_m, xyz_max_m = summarize_target_workspace(target)
    print(
        "target_range_m="
        f"x[{xyz_min_m[0]:.2f},{xyz_max_m[0]:.2f}] "
        f"y[{xyz_min_m[1]:.2f},{xyz_max_m[1]:.2f}] "
        f"z[{xyz_min_m[2]:.2f},{xyz_max_m[2]:.2f}] "
        f"xy_radius_max={r_xy:.2f}"
    )
    return state


def run_webcam_mediapipe(
    camera_index: int,
    stop_event: Event,
    camera_buffer: int,
    model_path: str | None,
    plot_every_n: int,
    live_target: LiveTargetState,
    point_count: int,
    scale: ScaleConfig,
) -> None:
    """Run the same webcam/MediaPipe recognition path used by webcam_main."""
    resolved_model = resolve_model_path(model_path, __file__)
    options = ob.HandLandmarkerOptions(
        base_options=ob.BaseOptions(model_asset_path=resolved_model),
        running_mode=ob.VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        print(f"[WARN] Cannot open webcam index {camera_index}; Crazyflow will continue.")
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(camera_buffer))
    except Exception:
        pass

    mode_state = ModeState()
    right_state = RightHandState()
    lp_shape = LpShapePipelineState()
    plot_every_n = max(0, int(plot_every_n))

    plot_enabled, fig, ax_hand, ax_topo = init_3d_plot(plot_every_n, "Online Control Webcam + 3D")
    render_enabled = True

    print(
        f"MediaPipe webcam started on camera {camera_index}. "
        "Left hand = MODE, right hand = OPEN. Press q/Enter in webcam window to stop."
    )
    try:
        with ob.HandLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_ms = int(frame_idx * (1000 / 30))
                try:
                    result = landmarker.detect_for_video(mp_image, t_ms)
                except Exception as exc:
                    print(f"[WARN] detect_for_video: {exc}")
                    continue

                idx_l, idx_r = find_left_right_indices(result, invert_handedness=False)
                pts_l = extract_world_points_mm_result(result, idx_l) if idx_l is not None else None
                dist_norm = (
                    index_mcp_tip_segment_norm(pts_l, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
                    if pts_l is not None
                    else None
                )

                mode_raw, tier_count = shared_update_mode_state(
                    pts_l,
                    mode_state=mode_state,
                    classify_mode_fn=classify_mode_from_fingers,
                    debounce_frames=_ONLINE_MODE_DEBOUNCE_FRAMES,
                    mode_smooth=0.22,
                )
                active_mode = int(mode_state.morph_mode)
                advance_lp_shape_p(dist_norm, active_mode, lp_shape)

                pts_r = extract_world_points_mm_result(result, idx_r) if idx_r is not None else None
                hands_3d = []
                topo_r = None
                if pts_r is not None:
                    topo_r = ob.analyze_hand_topology(pts_r)
                    right_state.last_right_pts = list(pts_r)
                    hands_3d = [pts_r]
                elif right_state.last_right_pts is not None:
                    hands_3d = [right_state.last_right_pts]

                open_out = shared_update_open_state(
                    pts_r,
                    right_state=right_state,
                    analyze_topology_fn=ob.analyze_hand_topology,
                    open_smooth=_ONLINE_OPEN_SMOOTH,
                    plane_snap_on=ob.PLANE_SNAP_ON,
                    plane_snap_off=ob.PLANE_SNAP_OFF,
                    sphere_snap_on=ob.SPHERE_SNAP_ON,
                    sphere_snap_off=ob.SPHERE_SNAP_OFF,
                    topology_analysis=topo_r,
                    snap_soft_k=_ONLINE_SNAP_SOFT_K,
                    snap_soft_max_step=_ONLINE_SNAP_SOFT_MAX_STEP,
                )

                frame, _kp_map = draw_all_hands(
                    frame,
                    result,
                    mode_hand_idx=idx_l,
                    morph_hand_idx=idx_r,
                    morph_mode=mode_state.morph_mode,
                    open_value=open_out,
                    depth_map=None,
                    print_depth=False,
                )

                hands_3d_plot = hands_3d
                if plot_enabled and not hands_3d_plot and pts_l is not None:
                    hands_3d_plot = [pts_l]

                analyses = None
                if plot_enabled and hands_3d_plot and (frame_idx % plot_every_n) == 0:
                    try:
                        analyses = update_online_3d_plot(
                            ax_hand=ax_hand,
                            ax_topo=ax_topo,
                            hands_3d=hands_3d_plot,
                            morph_mode=mode_state.morph_mode,
                            open_out=open_out,
                            lp_shape=lp_shape,
                        )
                        refresh_3d_plot_nonblocking(fig)
                    except Exception as exc:
                        plot_enabled = False
                        print(f"[WARN] Disabled Matplotlib 3D updates after plotting error: {exc}")

                if topo_r is not None:
                    update_live_target_from_state(
                        live_target=live_target,
                        mode_state=mode_state,
                        right_state=right_state,
                        lp_shape=lp_shape,
                        scale=scale,
                        radius_mm=float(topo_r["radius"]),
                        open_out=(open_out if open_out is not None else float(topo_r["morph_alpha"])),
                    )

                cv2.putText(
                    frame,
                    f"ONLINE M{mode_state.morph_mode} raw:{mode_raw} open:{open_out if open_out is not None else '-'} "
                    f"tier:{tier_count if tier_count >= 0 else '-'}",
                    (16, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Online Control Webcam", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 13):
                    stop_event.set()
                    break
                frame_idx += 1
    finally:
        cap.release()
        close_3d_plot(fig)
        cv2.destroyWindow("Online Control Webcam")


def run_online_crazyflow(
    target_provider: Callable[[], np.ndarray],
    point_count: int,
    duration: float,
    fps: int,
    target_alpha: float,
    min_separation_m: float = _DEFAULT_MIN_SEPARATION_M,
    stop_event: Event | None = None,
    morph_mode: int = 1,
    drone_model: str = _DEFAULT_DRONE_MODEL,
) -> None:
    """Run Crazyflow and continuously track targets from target_provider."""
    n_worlds = 1
    n_drones = int(point_count)
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        control=Control.state,
        drone_model=str(drone_model),
    )
    sim.reset()

    pos_buffer = deque(maxlen=_TRAIL_BUFFER_MAXLEN)
    formation_radius = max(0.90, 0.35 * n_drones / np.pi)
    formation_angles = np.linspace(0.0, 2.0 * np.pi, n_drones, endpoint=False)
    takeoff_start = np.stack(
        [
            formation_radius * np.cos(formation_angles),
            formation_radius * np.sin(formation_angles),
            np.full((n_drones,), 0.60, dtype=float),
        ],
        axis=1,
    ).astype(np.float32)
    takeoff_hover = np.stack(
        [
            formation_radius * np.cos(formation_angles),
            formation_radius * np.sin(formation_angles),
            np.full((n_drones,), 1.20, dtype=float),
        ],
        axis=1,
    ).astype(np.float32)
    first_target = np.asarray(target_provider(), dtype=np.float32).copy()
    first_target[:, 2] = np.maximum(first_target[:, 2], 1.10)
    smooth_target = takeoff_start.copy()
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))
    trail_rgba_cf = [colors[d].tolist() for d in range(n_drones)]
    cmd_cf = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)

    zeros = jnp.zeros_like(sim.data.states.pos)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=jnp.asarray(takeoff_start[None, :, :], device=sim.device),
            vel=zeros,
            ang_vel=zeros,
        )
    )
    print(
        f"Takeoff formation radius={formation_radius:.2f}m, "
        f"z={takeoff_start[0, 2]:.2f}->{takeoff_hover[0, 2]:.2f}m"
    )

    takeoff_duration = 2.0
    transition_duration = 2.0
    try:
        total_duration = max(float(duration), 0.1) + takeoff_duration + transition_duration
        for i in range(int(total_duration * sim.control_freq)):
            if stop_event is not None and stop_event.is_set():
                break
            t = i / sim.control_freq
            if t < takeoff_duration:
                alpha = t / max(takeoff_duration, 1e-6)
                alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0))
                raw_target = (1.0 - alpha) * takeoff_start + alpha * takeoff_hover
            elif t < takeoff_duration + transition_duration:
                alpha = (t - takeoff_duration) / max(transition_duration, 1e-6)
                alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0))
                raw_target = (1.0 - alpha) * takeoff_hover + alpha * first_target
            else:
                raw_target = np.asarray(target_provider(), dtype=np.float32)

            safe_target = enforce_min_separation(raw_target, min_separation_m, iters=10)
            smooth_target = target_alpha * safe_target + (1.0 - target_alpha) * smooth_target
            # Keep a safety margin after smoothing so filtering does not re-introduce near-collisions.
            smooth_target = enforce_min_separation(
                smooth_target,
                min_separation_m * 1.15,
                iters=8,
            )

            cmd_cf[..., :3] = smooth_target
            cmd_cf[..., 9] = 0.0

            sim.state_control(cmd_cf)
            sim.step(sim.freq // sim.control_freq)

            if ((i * fps) % sim.control_freq) < fps:
                apply_morph_led_theme(sim, int(morph_mode))
                pos_buffer.append(np.asarray(sim.data.states.pos[0], dtype=np.float64))
                if len(pos_buffer) > 1:
                    lines = np.asarray(pos_buffer)
                    for d in range(n_drones):
                        draw_line(
                            sim,
                            lines[:, d, :],
                            rgba=trail_rgba_cf[d],
                            start_size=0.5,
                            end_size=2.0,
                        )
                sim.render()
    finally:
        sim.close()


def run_integrated_online_control(
    live_target: LiveTargetState,
    point_count: int,
    duration: float,
    fps: int,
    target_alpha: float,
    min_separation_m: float,
    camera_index: int,
    camera_buffer: int,
    model_path: str | None,
    plot_every_n: int,
    scale: ScaleConfig,
    *,
    trail_every_n: int = _TRAIL_DRAW_EVERY_FRAMES,
    led_every_n: int = _LED_APPLY_EVERY_FRAMES,
    sim_render_every: int = _DEFAULT_SIM_RENDER_EVERY,
    morph_radius_mm: float = 50.0,
    drone_model: str = _DEFAULT_DRONE_MODEL,
    input_backend: str = _DEFAULT_INPUT_BACKEND,
    state_freq: int = _DEFAULT_STATE_FREQ,
    max_sim_substeps: int = _ONLINE_MAX_SIM_SUBSTEPS_PER_FRAME,
    imshow_every: int = 1,
    raw_target_ema: float = _DEFAULT_RAW_TARGET_EMA,
    max_target_step_m: float = _DEFAULT_MAX_TARGET_STEP_M,
    left_swarm_pose: bool = True,
    left_trans_scale: float = _DEFAULT_LEFT_TRANS_SCALE_MM,
    left_rot_scale: float = _DEFAULT_LEFT_ROT_SCALE,
    left_trans_ema: float = _DEFAULT_LEFT_TRANS_EMA,
    left_rot_ema: float = _DEFAULT_LEFT_ROT_EMA,
    left_max_offset_m: float = _DEFAULT_LEFT_MAX_OFFSET_M,
    left_max_rot_rad: float = _DEFAULT_LEFT_MAX_ROT_RAD,
    left_axis_sign: tuple[float, float, float] = _DEFAULT_LEFT_AXIS_SIGN,
    left_lost_decay: float = 0.92,
    debug_drone_targets_every: int = 0,
    open_jump_reset: float = _DEFAULT_OPEN_JUMP_RESET,
    left_unwind_s: float = _DEFAULT_LEFT_UNWIND_S,
    left_swarm_depth_frame_motion: bool = True,
    left_rot_gate_rad: float = _DEFAULT_LEFT_ROT_GATE_RAD,
    left_yaw_min_horiz: float = _DEFAULT_LEFT_YAW_MIN_HORIZ,
    left_rot_gain: float = _DEFAULT_LEFT_ROT_GAIN,
    left_rot_trans_tau_mm: float = _DEFAULT_LEFT_ROT_TRANS_TAU_MM,
    left_rot_world_z_scale: float = _DEFAULT_LEFT_ROT_WORLD_Z_SCALE,
    left_cam_y_to_world_z: float = _DEFAULT_LEFT_CAM_Y_TO_WORLD_Z,
    left_cam_preset: str = _DEFAULT_LEFT_CAM_PRESET,
    left_world_frame: str = _DEFAULT_LEFT_WORLD_FRAME,
    left_control_style: str = _DEFAULT_LEFT_CONTROL_STYLE,
    left_axis_trans_deadzone_m: float = _DEFAULT_AXIS_TRANS_DEADZONE_M,
    left_axis_rot_deadzone_rad: float = _DEFAULT_AXIS_ROT_DEADZONE_RAD,
    left_axis_cam_deadzone_xyz_mm: tuple[float, float, float] = _DEFAULT_AXIS_CAM_DEADZONE_XYZ_MM,
    left_axis_cam_snap_min_ratio: float = _DEFAULT_AXIS_CAM_SNAP_MIN_RATIO,
    left_axis_trans_on_m: float = _DEFAULT_AXIS_TRANS_ON_M,
    left_axis_rot_on_rad: float = _DEFAULT_AXIS_ROT_ON_RAD,
    left_axis_rot_excl_ratio: float = _DEFAULT_AXIS_ROT_EXCL_RATIO,
    left_axis_trans_excl_ratio: float = _DEFAULT_AXIS_TRANS_EXCL_RATIO,
    left_axis_rot_boost: float = _DEFAULT_AXIS_ROT_BOOST,
    left_axis_forward_boost: float = _DEFAULT_AXIS_FORWARD_BOOST,
    left_axis_lateral_min_dx_mm: float = _DEFAULT_AXIS_LATERAL_MIN_DX_MM,
    left_axis_forward_min_dz_mm: float = _DEFAULT_AXIS_FORWARD_MIN_DZ_MM,
    install_hotkey_deps: bool = False,
    left_depth_dom_ratio: float = _DEFAULT_LEFT_DEPTH_DOM_RATIO,
    left_depth_dom_min_mm: float = _DEFAULT_LEFT_DEPTH_DOM_MIN_MM,
    left_palm_basis: str = _DEFAULT_LEFT_PALM_BASIS,
    left_rot_coex_trans_min_mm: float = _DEFAULT_LEFT_ROT_COEX_TRANS_MIN_MM,
    left_rot_coex_max_angle_deg: float = _DEFAULT_LEFT_ROT_COEX_MAX_ANGLE_DEG,
    left_vertical_dom_ratio: float = _DEFAULT_LEFT_VERTICAL_DOM_RATIO,
    left_vertical_dom_min_mm: float = _DEFAULT_LEFT_VERTICAL_DOM_MIN_MM,
    left_vertical_optical_preserve_ratio: float = _DEFAULT_LEFT_VERTICAL_OPTICAL_PRESERVE_RATIO,
    left_vertical_optical_preserve_min_mm: float = _DEFAULT_LEFT_VERTICAL_OPTICAL_PRESERVE_MIN_MM,
    left_trans_lateral_dom_ratio: float = _DEFAULT_LEFT_TRANS_LATERAL_DOM_RATIO,
    left_trans_lateral_dom_min_mm: float = _DEFAULT_LEFT_TRANS_LATERAL_DOM_MIN_MM,
    left_trans_xy_dom_strip_dz_ratio: float = _DEFAULT_LEFT_TRANS_XY_DOM_STRIP_DZ_RATIO,
    left_trans_xy_dom_strip_dz_min_h_mm: float = _DEFAULT_LEFT_TRANS_XY_DOM_STRIP_DZ_MIN_H_MM,
    left_trans_dx_dom_strip_dz_ratio: float = _DEFAULT_LEFT_TRANS_DX_DOM_STRIP_DZ_RATIO,
    left_trans_dx_dom_strip_dz_min_dx_mm: float = _DEFAULT_LEFT_TRANS_DX_DOM_STRIP_DZ_MIN_DX_MM,
    left_trans_boost_optical: float = _DEFAULT_LEFT_TRANS_BOOST_OPTICAL,
    left_rot_planar_dom_ratio: float = _DEFAULT_LEFT_ROT_PLANAR_DOM_RATIO,
    left_rot_planar_dom_min_mm: float = _DEFAULT_LEFT_ROT_PLANAR_DOM_MIN_MM,
    left_rot_palm_face_twist_world_y: bool = _DEFAULT_LEFT_ROT_PALM_FACE_TWIST_WORLD_Y,
    left_rot_palm_face_cos_align_min: float = _DEFAULT_LEFT_ROT_PALM_FACE_COS_ALIGN_MIN,
    left_rot_palm_face_twist_dom_ratio: float = _DEFAULT_LEFT_ROT_PALM_FACE_TWIST_DOM_RATIO,
    left_rot_palm_face_twist_min_rad: float = _DEFAULT_LEFT_ROT_PALM_FACE_TWIST_MIN_RAD,
    left_rot_palm_face_twist_world_y_sign: float = _DEFAULT_LEFT_ROT_PALM_FACE_TWIST_WORLD_Y_SIGN,
    left_plane_rot_scale_mul: float = _DEFAULT_LEFT_PLANE_ROT_SCALE_MUL,
    left_rot_direct_follow: bool = False,
    left_rot_pivot: str = _DEFAULT_LEFT_ROT_PIVOT,
    left_dual_webcam_rot: bool = _DEFAULT_LEFT_DUAL_WEBCAM_ROT,
    left_rot_webcam_vis_thresh: float = _DEFAULT_LEFT_ROT_WEBCAM_VIS_THRESH,
    left_rot_webcam_index: int = _DEFAULT_LEFT_ROT_WEBCAM_INDEX,
    orbbec_flip_horizontal: bool = _DEFAULT_ORBBEC_FLIP_HORIZONTAL,
    orbbec_use_transformed_depth: bool = _DEFAULT_ORBBEC_USE_TRANSFORMED_DEPTH,
    orbbec_hand_swap: str = _DEFAULT_ORBBEC_HAND_SWAP,
    formation_rigid_3d_debug: bool = False,
    left_pose_frame_viz: bool = True,
    left_pose_frame_viz_every: int = _DEFAULT_LEFT_POSE_FRAME_VIZ_EVERY,
    left_pose_debug: bool = True,
    left_pose_debug_every: int = 2,
    webcam_rot_stride: int = _DEFAULT_WEBCAM_ROT_STRIDE,
    show_webcam_preview: bool = False,
    global_hotkeys: bool = True,
    mode_vis_min: float = _DEFAULT_MODE_VIS_MIN,
    record_trajectory: str | None = None,
) -> None:
    """Run MediaPipe, Matplotlib, OpenCV, and Crazyflow in the main thread.

    All stages run sequentially on one thread (lowest end-to-end latency for a single
    pipeline). True overlap of MediaPipe vs MuJoCo would need multiprocessing and adds
    sync/jitter cost; hot paths here use vectorized NumPy + reused buffers instead.
    """
    input_backend = str(input_backend).strip().lower()
    use_orbbec = input_backend == "orbbec"
    resolved_model = resolve_model_path(model_path, __file__)
    options = ob.HandLandmarkerOptions(
        base_options=ob.BaseOptions(model_asset_path=resolved_model),
        running_mode=ob.VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = None
    k4a = None
    calib = None
    ema_3d = None
    if use_orbbec:
        orbbec_fps = ob.FPS.FPS_15 if _ONLINE_ORBBEC_FPS == "15" else ob.FPS.FPS_30
        k4a = ob.PyK4A(
            ob.Config(
                color_resolution=1,
                depth_mode=2,
                synchronized_images_only=False,
                camera_fps=orbbec_fps,
            )
        )
        k4a.start()
        calib = k4a.calibration
    else:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {camera_index}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(camera_buffer))
        except Exception:
            pass

    n_worlds = 1
    n_drones = int(point_count)
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        control=Control.state,
        drone_model=str(drone_model),
        state_freq=max(1, int(state_freq)),
    )
    sim.reset()

    pos_buffer = deque(maxlen=_TRAIL_BUFFER_MAXLEN)
    first_target = live_target.get()
    first_target[:, 2] = np.maximum(first_target[:, 2], 1.10)
    smooth_target = first_target.copy()
    raw_target_filt = first_target.copy()
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))
    trail_rgba = [colors[d].tolist() for d in range(n_drones)]
    cmd_buf = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)

    traj_recorder: TrajectoryRecorder | None = None
    if record_trajectory:
        traj_recorder = TrajectoryRecorder(
            record_trajectory,
            meta={
                "source": "iso_swarm.online_control",
                "n_drones": n_drones,
                "fps": int(fps),
                "state_freq": int(state_freq),
                "min_separation_m": float(min_separation_m),
                "input_backend": str(input_backend),
                "drone_model": str(drone_model),
            },
        )
        print(f"Recording trajectory to {traj_recorder.path} (one sample per camera frame).")

    zeros = jnp.zeros_like(sim.data.states.pos)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=jnp.asarray(first_target[None, :, :], device=sim.device),
            vel=zeros,
            ang_vel=zeros,
        )
    )

    mode_state = ModeState()
    right_state = RightHandState()
    lp_shape = LpShapePipelineState()
    plot_every_n = max(0, int(plot_every_n))
    trail_every_n = max(0, int(trail_every_n))
    led_every_n = max(1, int(led_every_n))
    sim_render_every = max(0, int(sim_render_every))
    max_sim_substeps = max(1, int(max_sim_substeps))
    imshow_every = max(1, int(imshow_every))
    raw_target_ema = float(max(0.0, min(raw_target_ema, 1.0)))
    max_target_step_m = float(max(0.0, max_target_step_m))
    debug_drone_targets_every = max(0, int(debug_drone_targets_every))
    open_jump_reset = float(open_jump_reset)
    left_unwind_s = float(max(0.05, left_unwind_s))
    left_rot_gate_rad = float(max(0.0, left_rot_gate_rad))
    left_yaw_min_horiz = float(max(0.06, left_yaw_min_horiz))
    left_rot_gain = float(max(0.0, left_rot_gain))
    left_rot_trans_tau_mm = float(max(0.0, left_rot_trans_tau_mm))
    left_rot_world_z_scale = float(np.clip(left_rot_world_z_scale, 0.0, 1.0))
    left_cam_preset = str(left_cam_preset).strip().lower()
    left_cam_y_to_world_z = float(np.clip(left_cam_y_to_world_z, 0.0, 1.0))
    left_palm_basis = str(left_palm_basis).strip().lower()
    palm_basis_pair_indices(left_palm_basis)
    left_rot_coex_trans_min_mm = float(max(0.0, left_rot_coex_trans_min_mm))
    left_rot_coex_max_angle_rad = float(np.radians(max(0.0, left_rot_coex_max_angle_deg)))
    if left_rot_coex_trans_min_mm > 0.0 and left_rot_coex_max_angle_rad <= 0.0:
        left_rot_coex_max_angle_rad = float(np.radians(_DEFAULT_LEFT_ROT_COEX_MAX_ANGLE_DEG))
    left_vertical_dom_ratio = float(max(0.0, left_vertical_dom_ratio))
    left_vertical_dom_min_mm = float(max(0.0, left_vertical_dom_min_mm))
    left_vertical_optical_preserve_ratio = float(max(0.0, left_vertical_optical_preserve_ratio))
    left_vertical_optical_preserve_min_mm = float(max(0.0, left_vertical_optical_preserve_min_mm))
    left_trans_lateral_dom_ratio = float(max(0.0, left_trans_lateral_dom_ratio))
    left_trans_lateral_dom_min_mm = float(max(0.0, left_trans_lateral_dom_min_mm))
    left_trans_xy_dom_strip_dz_ratio = float(max(0.0, left_trans_xy_dom_strip_dz_ratio))
    left_trans_xy_dom_strip_dz_min_h_mm = float(max(0.0, left_trans_xy_dom_strip_dz_min_h_mm))
    left_trans_dx_dom_strip_dz_ratio = float(max(0.0, left_trans_dx_dom_strip_dz_ratio))
    left_trans_dx_dom_strip_dz_min_dx_mm = float(max(0.0, left_trans_dx_dom_strip_dz_min_dx_mm))
    left_trans_boost_optical = float(max(0.25, left_trans_boost_optical))
    left_rot_planar_dom_ratio = float(max(0.0, left_rot_planar_dom_ratio))
    left_rot_planar_dom_min_mm = float(max(0.0, left_rot_planar_dom_min_mm))
    left_rot_palm_face_cos_align_min = float(np.clip(left_rot_palm_face_cos_align_min, 0.0, 1.0))
    left_rot_palm_face_twist_dom_ratio = float(max(0.0, left_rot_palm_face_twist_dom_ratio))
    left_rot_palm_face_twist_min_rad = float(max(0.0, left_rot_palm_face_twist_min_rad))
    _sgy = float(left_rot_palm_face_twist_world_y_sign)
    left_rot_palm_face_twist_world_y_sign = (
        1.0 if (not np.isfinite(_sgy) or _sgy == 0.0) else float(np.sign(_sgy))
    )
    left_plane_rot_scale_mul = float(np.clip(left_plane_rot_scale_mul, 0.0, 1.0))
    if bool(left_rot_direct_follow):
        # Palm rotation R_cam = B @ refᵀ still mapped to world as M R Mᵀ; remove heuristics that
        # zeroed or heavily damped rotation so the swarm follows the hand frame visibly.
        left_rot_world_z_scale = 1.0
        left_trans_lateral_dom_ratio = 0.0
        left_trans_xy_dom_strip_dz_ratio = 0.0
        left_trans_dx_dom_strip_dz_ratio = 0.0
        left_trans_boost_optical = 1.0
        left_rot_trans_tau_mm = 0.0
        left_rot_gate_rad = 0.02
        left_rot_planar_dom_ratio = 0.0
        left_rot_planar_dom_min_mm = 0.0
        left_rot_coex_trans_min_mm = 0.0
        left_rot_coex_max_angle_rad = 0.0
        left_plane_rot_scale_mul = 1.0
        left_rot_scale = float(max(left_rot_scale, 0.72))
        left_rot_gain = float(max(left_rot_gain, 0.92))
    left_cam_motion = bool(left_swarm_depth_frame_motion) and use_orbbec and calib is not None
    left_world_frame_key = str(left_world_frame).strip().lower()
    left_control_style_key = str(left_control_style).strip().lower()
    left_use_camera_at_arm = left_world_frame_key == "camera_at_arm" and left_cam_motion
    left_M_rot = left_cam_preset_rotation(left_cam_preset) if left_cam_motion else None
    left_M_trans = (
        make_cam_translation_matrix(left_M_rot, image_y_to_world_z=left_cam_y_to_world_z)
        if left_M_rot is not None
        else None
    )
    left_rot_palm_face_twist_world_y_eff = (
        bool(left_rot_palm_face_twist_world_y)
        and bool(left_cam_motion)
        and (left_cam_preset == "fwd_y")
        and not bool(left_rot_direct_follow)
    )

    orbbec_swap_mp_hands = orbbec_resolve_swap_mp_hands(
        hand_swap=orbbec_hand_swap,
        flip_horizontal=bool(orbbec_flip_horizontal),
        use_orbbec=use_orbbec,
    )

    plot_enabled, fig, ax_hand, ax_topo = init_3d_plot(plot_every_n, "Online Control Webcam + 3D")
    render_enabled = True
    orbbec_flip_depth_warned = False

    if use_orbbec:
        print("Orbbec input started. Left hand = MODE, right hand = OPEN. Press q/Enter to stop.")
        if bool(orbbec_flip_horizontal):
            print(
                "Orbbec horizontal flip is ON (ego view vs mirror). "
                "Use --no-orbbec-flip-horizontal if depth/3D looks wrong."
            )
        if bool(orbbec_use_transformed_depth):
            print(
                "Orbbec transformed_depth is ON (K4A alignment). If the process aborts, use "
                "--no-orbbec-use-transformed-depth (default off for Femto Bolt)."
            )
        if orbbec_swap_mp_hands:
            print(
                "Orbbec: swapping MediaPipe left/right for mode vs open hand "
                f"(policy {str(orbbec_hand_swap).strip().lower()!r}; auto follows horizontal flip). "
                "Override: --orbbec-hand-swap off | on | auto."
            )
        if left_cam_motion:
            wf = (
                f"world_frame={left_world_frame_key!r} (cam→sim map locked at press 0)"
                if left_use_camera_at_arm
                else f"world_frame={left_world_frame_key!r} (cam→sim map from CLI each frame)"
            )
            print(
                f"Left-swarm depth: cam→sim preset={left_cam_preset!r} "
                f"(camera|legacy|fwd_y|flip_depth), {wf}, "
                f"palm_basis={left_palm_basis!r}."
            )
    else:
        print(
            f"MediaPipe webcam started on camera {camera_index}. "
            "Left hand = MODE, right hand = OPEN. Press q/Enter in webcam window to stop."
        )
    print("Holding default M1/open=1 target until SPACE is pressed.")
    left_pose_state = LeftSwarmPoseState(enabled=bool(left_swarm_pose))
    left_pose_reset_req = False
    left_pose_runtime_armed = False
    left_rot_pivot_key = str(left_rot_pivot).strip().lower()
    if left_rot_pivot_key not in ("per_drone", "centroid"):
        left_rot_pivot_key = "per_drone"
    left_dual_webcam_rot_eff = bool(left_dual_webcam_rot) and use_orbbec
    left_rot_webcam_vis_thresh = float(np.clip(left_rot_webcam_vis_thresh, 0.05, 0.99))
    if left_pose_state.enabled:
        print(
            "Left-hand whole formation: press 0 to START (zero pose = current hand; "
            "current palm center + pose are tracked relative to press-0), "
            f"0 again to restore morph frame (~{float(left_unwind_s):.1f}s)."
        )
        if bool(left_rot_direct_follow):
            print(
                "Left rotation direct-follow: palm vs baseline in camera frame → world via "
                "--left-cam-preset (M R Mᵀ); planar/coex/tau damp and strong Z damp disabled."
            )

    frame_idx = 0
    last_status_second = -1
    prev_open_for_snap: float | None = None
    gesture_control_enabled = False
    last_radius_mm = float(morph_radius_mm)
    start_time = time.monotonic()
    _hk_probe = (
        try_install_hotkey_dependencies()
        if bool(install_hotkey_deps)
        else probe_global_hotkey_backends()
    )
    key_queue = OnlineKeyQueue()
    if bool(global_hotkeys):
        key_queue.start(use_global=True, use_stdin=True)
        _hk = key_queue.mode
        if "pynput" not in _hk and "keyboard" not in _hk:
            print("[WARN] Global hotkeys not available in this Python environment.")
            print(format_hotkey_install_hint(_hk_probe))
        elif _hk == "off":
            print("[WARN] No hotkey backends started.")
            print(format_hotkey_install_hint(_hk_probe))
        else:
            print(f"Hotkeys: {_hk} — SPACE/0/q without Orbbec focus.")
        if left_control_style_key == "axis_locked":
            print(
                "Left control: axis_locked — hand at 0 is the position/pose reference; "
                "commanded motion is relative ±world X/Y/Z translation or rotation. "
                f"World ±XYZ trans/rot exclusive (rot×{left_axis_rot_excl_ratio:.2f} "
                f"trans×{left_axis_trans_excl_ratio:.2f}); fwd boost×{left_axis_forward_boost:.2f}."
            )
    gesture_control_enabled_box = [gesture_control_enabled]
    left_pose_reset_req_box = [left_pose_reset_req]
    left_pose_runtime_armed_box = [left_pose_runtime_armed]

    def _poll_keys(cv_key: int | None = None) -> bool:
        return process_online_control_keys(
            key_queue if bool(global_hotkeys) else None,
            global_hotkeys=bool(global_hotkeys),
            cv_key=cv_key,
            gesture_control_enabled=gesture_control_enabled_box,
            left_pose_reset_req=left_pose_reset_req_box,
            left_pose_runtime_armed=left_pose_runtime_armed_box,
            left_pose_state=left_pose_state,
            left_unwind_s=float(left_unwind_s),
            left_swarm_enabled=bool(left_pose_state.enabled),
        )

    try:
        with ob.HandLandmarker.create_from_options(options) as landmarker:
            webcam_cap = None
            webcam_landmarker = None
            webcam_frame_idx = 0
            _wcam_rot_stride = max(1, int(webcam_rot_stride))
            _wcam_rot_cache: dict = {"B": None, "res": None, "fr": None}
            if left_dual_webcam_rot_eff:
                try:
                    webcam_cap, _widx, _wb = open_webcam_capture(
                        int(left_rot_webcam_index), 0, 0, 8
                    )
                    webcam_landmarker = ob.HandLandmarker.create_from_options(options)
                    print(
                        f"Dual rotation: USB webcam index {_widx} ({_wb}) for palm pose when "
                        f"Orbbec visibility < {left_rot_webcam_vis_thresh:.2f}; "
                        "translation stays on depth wrist."
                    )
                except Exception as exc:
                    print(f"[WARN] Dual webcam rotation disabled: {exc}")
                    left_dual_webcam_rot_eff = False
                    webcam_cap = None
                    if webcam_landmarker is not None:
                        try:
                            webcam_landmarker.close()
                        except Exception:
                            pass
                        webcam_landmarker = None
            ocv_window_title = "Online Control Orbbec" if use_orbbec else "Online Control Webcam"
            try:
                cv2.namedWindow(ocv_window_title, cv2.WINDOW_NORMAL)
            except Exception:
                pass
            if bool(show_webcam_preview) and left_dual_webcam_rot_eff:
                try:
                    cv2.namedWindow(_WCAM_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
                except Exception:
                    pass
            while True:
                elapsed = time.monotonic() - start_time
                if float(duration) > 0.0 and elapsed > float(duration):
                    break
                if _poll_keys():
                    break

                t_ms = int(frame_idx * (1000 / max(float(fps), 1.0)))
                if use_orbbec:
                    capture = ob.safe_get_capture(k4a, warn_prefix="online_control get_capture")
                    got = ob.capture_orbbec_frame(capture)
                    if got is None:
                        try:
                            _ck = cv2.waitKey(1) & 0xFF
                        except Exception:
                            _ck = 255
                        if _poll_keys(_ck):
                            break
                        time.sleep(0.004)
                        continue
                    frame, depth_raw, capture = got
                    # transformed_depth can crash on Femto Bolt + Orbbec K4A-wrapper (see --orbbec-use-transformed-depth).
                    depth_aligned = ob.get_aligned_depth(capture, frame, bool(orbbec_use_transformed_depth))
                    if bool(orbbec_flip_horizontal):
                        fh, fw = int(frame.shape[0]), int(frame.shape[1])
                        frame = cv2.flip(frame, 1)
                        if depth_aligned is not None and depth_aligned.shape[:2] == (fh, fw):
                            depth_aligned = cv2.flip(depth_aligned, 1)
                        elif depth_raw is not None and depth_raw.shape[:2] == (fh, fw):
                            depth_raw = cv2.flip(depth_raw, 1)
                        elif not orbbec_flip_depth_warned:
                            print(
                                "[WARN] orbbec_flip_horizontal: depth resolution != color; flipped color only. "
                                "Try disabling flip or ensure transformed_depth matches color."
                            )
                            orbbec_flip_depth_warned = True
                    mp_frame = frame
                    if 0.0 < _ONLINE_MP_INPUT_SCALE < 1.0:
                        h0, w0 = frame.shape[:2]
                        sw = max(64, int(round(w0 * _ONLINE_MP_INPUT_SCALE)))
                        sh = max(48, int(round(h0 * _ONLINE_MP_INPUT_SCALE)))
                        mp_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
                    mp_image = ob.make_mp_image_from_bgr(mp_frame)
                    result = ob.detect_for_video_safe(
                        landmarker,
                        mp_image,
                        t_ms,
                        warn_prefix="online_control detect_for_video",
                    )
                    if result is None:
                        try:
                            _ck = cv2.waitKey(1) & 0xFF
                        except Exception:
                            _ck = 255
                        if _poll_keys(_ck):
                            break
                        time.sleep(0.002)
                        continue
                    depth_raw_for_draw = depth_raw if _ONLINE_ORBBEC_USE_DEPTH_FUSION else None
                    depth_aligned_for_draw = depth_aligned if _ONLINE_ORBBEC_USE_DEPTH_FUSION else None
                    calib_for_draw = calib if _ONLINE_ORBBEC_USE_DEPTH_FUSION else None
                    fusion_w = ob.DEPTH_FUSION_WEIGHT if _ONLINE_ORBBEC_USE_DEPTH_FUSION else 0.0
                    frame, hands_3d_all, ema_3d = ob.draw_hand(
                        frame,
                        result,
                        depth_raw=depth_raw_for_draw,
                        depth_aligned=depth_aligned_for_draw,
                        print_depth=False,
                        calibration=calib_for_draw,
                        fusion_weight=fusion_w,
                        ema_alpha=ob.POINT_EMA_ALPHA,
                        ema_points=ema_3d,
                        depth_patch_radius=(ob.DEPTH_MEDIAN_PATCH_RADIUS if _ONLINE_ORBBEC_USE_DEPTH_FUSION else 0),
                        hand_frame=ob.HAND_FRAME_SCALED,
                        filter_depth_outliers=_ONLINE_ORBBEC_USE_DEPTH_FUSION,
                        depth_max_delta_mm=ob.DEPTH_MAX_DELTA_FROM_WRIST_MM,
                        depth_median_max_delta_mm=ob.DEPTH_MEDIAN_MAX_DELTA_MM,
                        hand_3d_source=ob.HAND_3D_SOURCE_MP,
                        depth_unproject_rigid_T=None,
                    )
                    if orbbec_swap_mp_hands:
                        idx_l = ob.find_hand_index_by_side(result, "right")
                        idx_r = ob.find_hand_index_by_side(result, "left")
                    else:
                        idx_l = ob.find_hand_index_by_side(result, "left")
                        idx_r = ob.find_hand_index_by_side(result, "right")
                    pts_l = (
                        hands_3d_all[idx_l]
                        if idx_l is not None and idx_l < len(hands_3d_all)
                        else None
                    )
                    pts_r = (
                        hands_3d_all[idx_r]
                        if idx_r is not None and idx_r < len(hands_3d_all)
                        else None
                    )
                    mode_raw, tier_count = process_left_mode(
                        hands_3d_all,
                        idx_l,
                        mode_state,
                        mp_result=result,
                        mode_vis_min=float(mode_vis_min),
                    )
                    hands_3d, open_out = process_right_open(hands_3d_all, idx_r, right_state)
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        try:
                            _ck = cv2.waitKey(1) & 0xFF
                        except Exception:
                            _ck = 255
                        if _poll_keys(_ck):
                            break
                        time.sleep(0.004)
                        continue
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    mp_frame = frame
                    if 0.0 < _ONLINE_MP_INPUT_SCALE < 1.0:
                        h0, w0 = frame.shape[:2]
                        sw = max(64, int(round(w0 * _ONLINE_MP_INPUT_SCALE)))
                        sh = max(48, int(round(h0 * _ONLINE_MP_INPUT_SCALE)))
                        mp_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    try:
                        result = landmarker.detect_for_video(mp_image, t_ms)
                    except Exception as exc:
                        print(f"[WARN] detect_for_video: {exc}")
                        try:
                            _ck = cv2.waitKey(1) & 0xFF
                        except Exception:
                            _ck = 255
                        if _poll_keys(_ck):
                            break
                        time.sleep(0.002)
                        continue

                    idx_l, idx_r = find_left_right_indices(result, invert_handedness=False)
                    pts_l = extract_world_points_mm_result(result, idx_l) if idx_l is not None else None
                    pts_r = extract_world_points_mm_result(result, idx_r) if idx_r is not None else None
                    _mode_vis_min = None
                    if idx_l is not None and result is not None and float(mode_vis_min) > 0.0:
                        _, _mode_vis_min = mp_hand_visibility_scores(result, idx_l)
                    mode_raw, tier_count = shared_update_mode_state(
                        pts_l,
                        mode_state=mode_state,
                        classify_mode_fn=classify_mode_from_fingers,
                        debounce_frames=_ONLINE_MODE_DEBOUNCE_FRAMES,
                        mode_smooth=0.22,
                        mode_vis_min=float(mode_vis_min),
                        hand_visibility_min=_mode_vis_min,
                    )
                    hands_3d = []
                    topo_r = None
                    if pts_r is not None:
                        topo_r = ob.analyze_hand_topology(pts_r)
                        right_state.last_right_pts = list(pts_r)
                        hands_3d = [pts_r]
                    elif right_state.last_right_pts is not None:
                        hands_3d = [right_state.last_right_pts]
                    open_out = shared_update_open_state(
                        pts_r,
                        right_state=right_state,
                        analyze_topology_fn=ob.analyze_hand_topology,
                        open_smooth=_ONLINE_OPEN_SMOOTH,
                        plane_snap_on=ob.PLANE_SNAP_ON,
                        plane_snap_off=ob.PLANE_SNAP_OFF,
                        sphere_snap_on=ob.SPHERE_SNAP_ON,
                        sphere_snap_off=ob.SPHERE_SNAP_OFF,
                        topology_analysis=topo_r,
                        snap_soft_k=_ONLINE_SNAP_SOFT_K,
                        snap_soft_max_step=_ONLINE_SNAP_SOFT_MAX_STEP,
                    )

                # Orbbec: translation from depth-unprojected wrist (MP world is wrist-centric → no pan).
                mph, mpw = int(mp_frame.shape[0]), int(mp_frame.shape[1])
                if use_orbbec and calib is not None and idx_l is not None:
                    pts_l_pose_mm = left_hand_pose_matrix_depth_mm(
                        result,
                        idx_l,
                        int(frame.shape[0]),
                        int(frame.shape[1]),
                        mph,
                        mpw,
                        calibration=calib,
                        depth_aligned=depth_aligned,
                        depth_raw=depth_raw,
                        patch_r=int(ob.DEPTH_MEDIAN_PATCH_RADIUS),
                        palm_basis=left_palm_basis,
                    )
                else:
                    pts_l_pose_mm = (
                        extract_world_points_mm_result(result, idx_l) if idx_l is not None else None
                    )

                dist_norm = (
                    index_mcp_tip_segment_norm(pts_l, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
                    if pts_l is not None
                    else None
                )
                active_mode = int(mode_state.morph_mode)
                advance_lp_shape_p(dist_norm, active_mode, lp_shape)

                if not use_orbbec:
                    frame, _kp_map = draw_all_hands(
                        frame,
                        result,
                        mode_hand_idx=idx_l,
                        morph_hand_idx=idx_r,
                        morph_mode=mode_state.morph_mode,
                        open_value=open_out,
                        depth_map=None,
                        print_depth=False,
                    )

                if gesture_control_enabled and hands_3d:
                    topo_target = ob.analyze_hand_topology(hands_3d[0])
                    if topo_target is not None:
                        # Orbbec online path uses HAND_FRAME_SCALED (normalized units), so
                        # topology radius is not in millimeters and must not drive morph scale.
                        # Keep drone target scale in real metric mm using the configured radius.
                        last_radius_mm = float(morph_radius_mm)
                        open_target = (
                            open_out
                            if open_out is not None
                            else float(topo_target.get("morph_alpha", right_state.last_open_out or 0.0))
                        )
                        update_live_target_from_state(
                            live_target=live_target,
                            mode_state=mode_state,
                            right_state=right_state,
                            lp_shape=lp_shape,
                            scale=scale,
                            radius_mm=float(last_radius_mm),
                            open_out=float(open_target),
                        )

                raw_target = live_target.get()
                morph_targets_before_left_m = raw_target.copy()
                left_swarm_off: np.ndarray | None = None
                left_swarm_R: np.ndarray | None = None
                left_pose_dbg = ""
                _rot_dbg = ""
                _viz_B_rot = None
                if left_pose_state.enabled and (left_pose_runtime_armed or left_pose_state.is_unwinding()):
                    _plane_rot_mul = (
                        float(left_plane_rot_scale_mul)
                        if right_state.snap_state == "plane"
                        else 1.0
                    )
                    _rot_gate_eff = (
                        0.0
                        if left_control_style_key == "axis_locked"
                        else float(left_rot_gate_rad)
                    )
                    _do_arm = bool(left_pose_reset_req and left_pose_runtime_armed)
                    _arm_M_rot = None
                    _arm_M_trans = None
                    if left_use_camera_at_arm and _do_arm:
                        _arm_M_rot, _arm_M_trans = build_sim_from_cam_matrices(
                            left_cam_preset,
                            image_y_to_world_z=float(left_cam_y_to_world_z),
                        )
                    _arm_ref_drone = None
                    if _do_arm and left_rot_pivot_key == "per_drone":
                        _arm_ref_drone = np.asarray(morph_targets_before_left_m[:, :3], dtype=np.float64)
                    _prefetch_B = None
                    _prefetch_res = None
                    _prefetch_wfr = None
                    if left_dual_webcam_rot_eff and webcam_cap is not None and webcam_landmarker is not None:
                        _need_wcam_read = bool(show_webcam_preview)
                        if not _need_wcam_read and idx_l is not None and result is not None:
                            _, _vis_min_pre = mp_hand_visibility_scores(result, idx_l)
                            _need_wcam_read = float(_vis_min_pre) < float(left_rot_webcam_vis_thresh)
                        _prefetch_B = None
                        _prefetch_res = None
                        _prefetch_wfr = None
                        if _need_wcam_read:
                            _poll_wcam = bool(show_webcam_preview) or (
                                (frame_idx % _wcam_rot_stride) == 0
                            )
                            if _poll_wcam:
                                _prefetch_B, _prefetch_res, _prefetch_wfr, webcam_frame_idx = (
                                    _detect_webcam_hand(
                                        webcam_cap,
                                        webcam_landmarker,
                                        fps=float(fps),
                                        webcam_frame_idx=webcam_frame_idx,
                                        mp_input_scale=float(_ONLINE_MP_INPUT_SCALE),
                                        palm_basis=left_palm_basis,
                                        prefer_hand_idx=idx_l,
                                    )
                                )
                                _wcam_rot_cache["B"] = _prefetch_B
                                _wcam_rot_cache["res"] = _prefetch_res
                                _wcam_rot_cache["fr"] = _prefetch_wfr
                            else:
                                _prefetch_B = _wcam_rot_cache.get("B")
                                _prefetch_res = _wcam_rot_cache.get("res")
                                _prefetch_wfr = _wcam_rot_cache.get("fr")
                    _dual_rot, webcam_frame_idx = resolve_dual_left_rotation(
                        enabled=bool(left_dual_webcam_rot_eff),
                        orbbec_result=result,
                        orbbec_idx_l=idx_l,
                        do_arm=_do_arm,
                        palm_basis=left_palm_basis,
                        vis_thresh=float(left_rot_webcam_vis_thresh),
                        webcam_cap=webcam_cap,
                        webcam_landmarker=webcam_landmarker,
                        fps=float(fps),
                        webcam_frame_idx=webcam_frame_idx,
                        mp_input_scale=float(_ONLINE_MP_INPUT_SCALE),
                        prefetch_B=_prefetch_B,
                        prefetch_result=_prefetch_res,
                        prefetch_frame_bgr=_prefetch_wfr,
                    )
                    _B_rot = _dual_rot.B_rot
                    _viz_B_rot = _B_rot
                    _arm_ref_img = _dual_rot.arm_ref_img
                    _rot_dbg = _dual_rot.rot_dbg
                    if bool(show_webcam_preview) and _dual_rot.webcam_frame_bgr is not None:
                        _wdisp = _dual_rot.webcam_frame_bgr.copy()
                        if _dual_rot.webcam_result is not None:
                            _wdisp, _ = draw_hand_webcam(_wdisp, _dual_rot.webcam_result)
                        _wc = (60, 220, 80) if _dual_rot.rot_source == "depth" else (80, 200, 255)
                        cv2.putText(
                            _wdisp,
                            f"rot={_dual_rot.rot_source} vis_min={_dual_rot.vis_min:.2f} "
                            f"thresh={left_rot_webcam_vis_thresh:.2f}",
                            (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            _wc,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(_WCAM_PREVIEW_WINDOW, _wdisp)
                    off, R_pose = update_left_swarm_pose(
                        pts_l_pose_mm,
                        left_pose_state,
                        trans_scale=float(left_trans_scale),
                        rot_scale=float(left_rot_scale) * _plane_rot_mul,
                        trans_ema=float(left_trans_ema),
                        rot_ema=float(left_rot_ema),
                        max_offset_m=float(left_max_offset_m),
                        max_rot_rad=float(left_max_rot_rad),
                        axis_sign=tuple(left_axis_sign),
                        hand_lost_decay=float(left_lost_decay),
                        force_reset=_do_arm,
                        cam_delta_to_world=left_M_rot,
                        cam_translation_to_world=left_M_trans,
                        arm_sim_from_cam=_arm_M_rot,
                        arm_sim_trans_from_cam=_arm_M_trans,
                        arm_cam_preset_label=str(left_cam_preset) if _arm_M_rot is not None else "",
                        ref_drone_xyz=_arm_ref_drone,
                        ref_basis_image=_arm_ref_img,
                        B_rot=_B_rot,
                        rot_gate_rad=float(_rot_gate_eff),
                        yaw_min_horiz_norm=float(left_yaw_min_horiz),
                        rot_gain=float(left_rot_gain),
                        rot_trans_tau_mm=float(left_rot_trans_tau_mm),
                        rot_world_z_scale=float(left_rot_world_z_scale),
                        depth_dom_ratio=float(left_depth_dom_ratio),
                        depth_dom_min_mm=float(left_depth_dom_min_mm),
                        palm_basis=left_palm_basis,
                        rot_coex_trans_min_mm=float(left_rot_coex_trans_min_mm),
                        rot_coex_max_angle_rad=float(left_rot_coex_max_angle_rad),
                        vertical_dom_ratio=float(left_vertical_dom_ratio),
                        vertical_dom_min_mm=float(left_vertical_dom_min_mm),
                        vertical_optical_preserve_ratio=float(left_vertical_optical_preserve_ratio),
                        vertical_optical_preserve_min_mm=float(left_vertical_optical_preserve_min_mm),
                        lateral_dom_ratio=float(left_trans_lateral_dom_ratio),
                        lateral_dom_min_mm=float(left_trans_lateral_dom_min_mm),
                        trans_xy_dom_strip_dz_ratio=float(left_trans_xy_dom_strip_dz_ratio),
                        trans_xy_dom_strip_dz_min_h_mm=float(left_trans_xy_dom_strip_dz_min_h_mm),
                        trans_dx_dom_strip_dz_ratio=float(left_trans_dx_dom_strip_dz_ratio),
                        trans_dx_dom_strip_dz_min_dx_mm=float(left_trans_dx_dom_strip_dz_min_dx_mm),
                        trans_boost_optical=float(left_trans_boost_optical),
                        rot_planar_dom_ratio=float(left_rot_planar_dom_ratio),
                        rot_planar_dom_min_mm=float(left_rot_planar_dom_min_mm),
                        rot_palm_face_twist_world_y=bool(left_rot_palm_face_twist_world_y_eff),
                        rot_palm_face_cos_align_min=float(left_rot_palm_face_cos_align_min),
                        rot_palm_face_twist_dom_ratio=float(left_rot_palm_face_twist_dom_ratio),
                        rot_palm_face_twist_min_rad=float(left_rot_palm_face_twist_min_rad),
                        rot_palm_face_twist_world_y_sign=float(left_rot_palm_face_twist_world_y_sign),
                        control_style=left_control_style_key,
                        axis_trans_deadzone_m=float(left_axis_trans_deadzone_m),
                        axis_rot_deadzone_rad=float(left_axis_rot_deadzone_rad),
                        axis_cam_deadzone_xyz_mm=tuple(left_axis_cam_deadzone_xyz_mm),
                        axis_cam_snap_min_ratio=float(left_axis_cam_snap_min_ratio),
                        axis_trans_on_m=float(left_axis_trans_on_m),
                        axis_rot_on_rad=float(left_axis_rot_on_rad),
                        axis_rot_excl_ratio=float(left_axis_rot_excl_ratio),
                        axis_trans_excl_ratio=float(left_axis_trans_excl_ratio),
                        axis_rot_boost=float(left_axis_rot_boost),
                        axis_forward_boost=float(left_axis_forward_boost),
                        axis_lateral_min_dx_mm=float(left_axis_lateral_min_dx_mm),
                        axis_forward_min_dz_mm=float(left_axis_forward_min_dz_mm),
                    )
                    if _do_arm and left_use_camera_at_arm and left_pose_state.frozen_cam_preset:
                        print(
                            "Left swarm armed: depth-camera origin (wrist) + palm basis; "
                            f"frozen cam→sim preset={left_pose_state.frozen_cam_preset!r}."
                        )
                    left_pose_reset_req_box[0] = False
                    left_swarm_off = np.asarray(off, dtype=np.float64).copy()
                    left_swarm_R = np.asarray(R_pose, dtype=np.float64).copy()
                    raw_target = apply_rigid_to_targets(
                        morph_targets_before_left_m,
                        off,
                        R_pose,
                        ref_drone_xyz=left_pose_state.ref_drone_xyz,
                        pivot=left_rot_pivot_key,
                    )
                    _mot = getattr(left_pose_state, "last_axis_motion", "")
                    if _mot and _mot != "none":
                        _wr = float(getattr(left_pose_state, "last_rot_blend_w", 1.0))
                        _wt = float(getattr(left_pose_state, "last_trans_blend_w", 1.0))
                        _mot_s = f" [{_mot} R×{_wr:.0%} T×{_wt:.0%}]"
                    else:
                        _mot_s = ""
                    _dc = getattr(left_pose_state, "last_delta_cam_mm", None)
                    _dh0 = getattr(left_pose_state, "last_delta_h_raw_m", None)
                    _pan_mm = float(np.linalg.norm(_dc)) if _dc is not None else 0.0
                    _raw_m = float(np.linalg.norm(_dh0)) if _dh0 is not None else 0.0
                    left_pose_dbg = (
                        f" d={float(np.linalg.norm(off)):.2f}m"
                        f" pan={_pan_mm:.0f}mm raw={_raw_m:.3f}m{_rot_dbg}{_mot_s}"
                    )
                    _dbg_every = max(1, int(left_pose_debug_every))
                    if (
                        bool(left_pose_debug)
                        and left_pose_state.enabled
                        and (left_pose_runtime_armed or left_pose_state.is_unwinding())
                        and left_pose_state.initialized
                        and (frame_idx % _dbg_every) == 0
                    ):
                        print_left_swarm_pose_debug(
                            left_pose_state,
                            frame_idx=int(frame_idx),
                            axis_sign=tuple(left_axis_sign),
                            trans_scale=float(left_trans_scale),
                        )

                _viz_every = max(1, int(left_pose_frame_viz_every))
                if (
                    bool(left_pose_frame_viz)
                    and left_pose_state.enabled
                    and (frame_idx % _viz_every) == 0
                ):
                    draw_left_pose_frame_overlay(
                        frame,
                        calibration=calib if use_orbbec else None,
                        pts_l_pose_mm=pts_l_pose_mm,
                        result=result,
                        idx_l=idx_l,
                        left_pose_state=left_pose_state,
                        left_runtime_armed=bool(left_pose_runtime_armed),
                        B_rot=_viz_B_rot,
                        R_pose=left_swarm_R,
                        off_m=left_swarm_off,
                        palm_basis=left_palm_basis,
                        use_depth_projection=bool(use_orbbec and calib is not None),
                        motion=str(getattr(left_pose_state, "last_axis_motion", "none")),
                        rv_pose_world=np.asarray(left_pose_state.last_rv_pose_world, dtype=np.float64).copy()
                        if left_pose_runtime_armed or left_pose_state.is_unwinding()
                        else None,
                        rv_cmd_world=np.asarray(left_pose_state.last_rv_cmd_world, dtype=np.float64).copy()
                        if left_pose_runtime_armed or left_pose_state.is_unwinding()
                        else None,
                        pose_rotate_rad=float(left_axis_rot_on_rad),
                    )

                hands_3d_plot = hands_3d
                if plot_enabled and not hands_3d_plot and idx_l is not None and idx_l < len(hands_3d_all):
                    hands_3d_plot = [hands_3d_all[idx_l]]

                analyses = None
                if plot_enabled and hands_3d_plot and (frame_idx % plot_every_n) == 0:
                    try:
                        analyses = update_online_3d_plot(
                            ax_hand=ax_hand,
                            ax_topo=ax_topo,
                            hands_3d=hands_3d_plot,
                            morph_mode=mode_state.morph_mode,
                            open_out=open_out,
                            lp_shape=lp_shape,
                        )
                        if (
                            formation_rigid_3d_debug
                            and left_pose_state.enabled
                            and (left_pose_runtime_armed or left_pose_state.is_unwinding())
                            and left_swarm_R is not None
                        ):
                            draw_formation_rigid_debug_on_topo(
                                ax_topo,
                                morph_targets_before_left_m,
                                raw_target,
                                off_m=left_swarm_off,
                                R_pose=left_swarm_R,
                            )
                        else:
                            _clear_formation_rigid_debug(ax_topo)
                        refresh_3d_plot_nonblocking(fig)
                    except Exception as exc:
                        plot_enabled = False
                        print(f"[WARN] Disabled Matplotlib 3D updates after plotting error: {exc}")

                if _STATUS_PRINT_EVERY_SEC > 0:
                    status_second = int(elapsed)
                    if (status_second // _STATUS_PRINT_EVERY_SEC) != last_status_second:
                        last_status_second = status_second // _STATUS_PRINT_EVERY_SEC
                        has_l = idx_l is not None
                        has_r = idx_r is not None
                        if gesture_control_enabled or has_l or has_r:
                            min_dist, min_i, min_j = closest_pair(raw_target)
                            open_txt = f"{float(open_out):.2f}" if open_out is not None else "-"
                            print(
                                f"online t={elapsed:.1f}s armed={'yes' if gesture_control_enabled else 'no'} "
                                f"mode={int(mode_state.morph_mode)} "
                                f"raw={mode_raw} open={open_txt} "
                                f"L={'yes' if has_l else 'no'} "
                                f"R={'yes' if has_r else 'no'} "
                                f"target_min=({min_i},{min_j}) {min_dist:.2f}m"
                            )

                # Fixed substeps from nominal FPS so blend depth does not jitter frame-to-frame
                # (wall-clock dt caused large jumps when one frame had few steps and the next many).
                n_sim_steps = max(
                    1,
                    min(int(round(float(sim.freq) / max(float(fps), 1.0))), max_sim_substeps),
                )
                ta = float(min(max(target_alpha, 1e-6), 0.999999))
                blend_per_step = 1.0 - (1.0 - ta) ** (1.0 / float(n_sim_steps))

                if raw_target_ema > 0.0:
                    b = raw_target_ema
                    raw_target_filt = b * raw_target + (1.0 - b) * raw_target_filt
                    safe_src = raw_target_filt
                else:
                    safe_src = raw_target
                safe_target = enforce_min_separation(safe_src, min_separation_m, iters=10)
                if (
                    open_jump_reset > 0.0
                    and gesture_control_enabled
                    and open_out is not None
                    and prev_open_for_snap is not None
                    and abs(float(open_out) - float(prev_open_for_snap)) >= open_jump_reset
                ):
                    smooth_target = np.asarray(safe_target, dtype=np.float32).copy()
                if open_out is not None:
                    prev_open_for_snap = float(open_out)

                for _ in range(n_sim_steps):
                    blended = blend_per_step * safe_target + (1.0 - blend_per_step) * smooth_target
                    smooth_target = clamp_targets_step(smooth_target, blended, max_target_step_m)
                    smooth_target = enforce_min_separation(
                        smooth_target,
                        min_separation_m * 1.15,
                        iters=6,
                    )
                    cmd_buf[..., :3] = smooth_target
                    cmd_buf[..., 9] = 0.0
                    sim.state_control(cmd_buf)
                    sim.step(1)
                if debug_drone_targets_every > 0 and (frame_idx % debug_drone_targets_every) == 0:
                    cmd_snapshot = np.asarray(smooth_target, dtype=np.float64).copy()
                    debug_print_drone_targets(
                        cmd_snapshot,
                        frame_idx=frame_idx,
                        morph_mode=int(mode_state.morph_mode),
                        open_v=open_out,
                        label="cmd_target",
                    )
                    try:
                        pos0 = np.asarray(sim.data.states.pos[0], dtype=np.float64)
                        debug_print_drone_targets(
                            pos0,
                            frame_idx=frame_idx,
                            morph_mode=int(mode_state.morph_mode),
                            open_v=open_out,
                            label="sim_pos",
                            compare_to=cmd_snapshot,
                        )
                    except Exception:
                        pass
                if frame_idx % led_every_n == 0:
                    apply_morph_led_theme(sim, int(mode_state.morph_mode))

                if trail_every_n > 0:
                    pos_buffer.append(np.asarray(sim.data.states.pos[0], dtype=np.float64))
                if render_enabled and trail_every_n > 0 and len(pos_buffer) > 1 and (frame_idx % trail_every_n == 0):
                    lines = np.asarray(pos_buffer)
                    for d in range(n_drones):
                        try:
                            draw_line(
                                sim,
                                lines[:, d, :],
                                rgba=trail_rgba[d],
                                start_size=0.5,
                                end_size=2.0,
                            )
                        except Exception as exc:
                            render_enabled = False
                            print(f"[WARN] Disabled Crazyflow trail drawing after render error: {exc}")
                            break
                if render_enabled and sim_render_every > 0 and (frame_idx % sim_render_every == 0):
                    try:
                        sim.render()
                    except Exception as exc:
                        render_enabled = False
                        print(f"[WARN] Disabled Crazyflow rendering after render error: {exc}")

                if traj_recorder is not None:
                    pos0 = np.asarray(sim.data.states.pos[0], dtype=np.float32)
                    vel0 = np.asarray(sim.data.states.vel[0], dtype=np.float32)
                    traj_recorder.append(
                        elapsed,
                        setpoint=np.asarray(safe_target, dtype=np.float32),
                        raw_target=np.asarray(raw_target, dtype=np.float32),
                        cmd_target=np.asarray(smooth_target, dtype=np.float32),
                        sim_pos=pos0,
                        sim_vel=vel0,
                        gesture_armed=bool(gesture_control_enabled),
                    )

                if left_pose_state.enabled:
                    if left_pose_state.is_unwinding():
                        pose_hint = f" | L-move:restore{left_pose_dbg}"
                    elif left_pose_runtime_armed:
                        pose_hint = f" | L-move:ON{left_pose_dbg}"
                    else:
                        pose_hint = " | L-move:[0]"
                else:
                    pose_hint = ""
                cv2.putText(
                    frame,
                    f"ONLINE {'ARMED' if gesture_control_enabled_box[0] else 'HOLD DEFAULT - press SPACE'} "
                    f"M{mode_state.morph_mode} raw:{mode_raw} open:{open_out if open_out is not None else '-'} "
                    f"tier:{tier_count if tier_count >= 0 else '-'}{pose_hint}",
                    (16, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if frame_idx % imshow_every == 0:
                    cv2.imshow(ocv_window_title, frame)
                if _poll_keys(cv2.waitKey(1) & 0xFF):
                    break
                gesture_control_enabled = bool(gesture_control_enabled_box[0])
                left_pose_reset_req = bool(left_pose_reset_req_box[0])
                left_pose_runtime_armed = bool(left_pose_runtime_armed_box[0])
                frame_idx += 1
            if webcam_landmarker is not None:
                try:
                    webcam_landmarker.close()
                except Exception:
                    pass
            if webcam_cap is not None:
                webcam_cap.release()
            key_queue.stop()
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, stopping online control...")
    finally:
        if traj_recorder is not None:
            try:
                out = traj_recorder.save()
                print(f"Saved trajectory ({traj_recorder.n_frames} frames) to {out}")
            except RuntimeError as exc:
                print(f"[WARN] Trajectory not saved: {exc}")
        key_queue.stop()
        if cap is not None:
            cap.release()
        if k4a is not None:
            try:
                k4a.stop()
            except Exception:
                pass
            k4a = None
        sim.close()
        close_3d_plot(fig)
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point for online Crazyflow control."""
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
        "--target-alpha",
        type=float,
        default=0.14,
        help="Target tracking gain per outer frame (lower = slower/smoother; used with physics substeps).",
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
        "--no-left-pose-frame-viz",
        action="store_true",
        help="Disable Orbbec overlay: hand palm RGB axes (hX/hY/hZ) + arm ref (0*) + swarm centroid inset.",
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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-buffer", type=int, default=1)
    parser.add_argument(
        "--input-backend",
        choices=("orbbec", "webcam"),
        default=_DEFAULT_INPUT_BACKEND,
        help="Gesture input backend (default: orbbec).",
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
        "--plot-every",
        type=int,
        default=int(_DEFAULT_PLOT_EVERY_ONLINE),
        help=f"Matplotlib 3D every N camera frames (default {int(_DEFAULT_PLOT_EVERY_ONLINE)}; 1=every frame; 0=off).",
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
        "--state-freq",
        type=int,
        default=_DEFAULT_STATE_FREQ,
        help="Crazyflow position setpoint update rate (Hz); higher tracks fast target changes better.",
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
        default=1,
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
        "--max-target-step-m",
        type=float,
        default=_DEFAULT_MAX_TARGET_STEP_M,
        help="Cap per-drone target motion per physics substep (m); lower = slower max speed; 0 disables.",
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
    parser.add_argument("--left-lost-decay", type=float, default=0.92)
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
        "--left-control-style",
        type=str,
        default=_DEFAULT_LEFT_CONTROL_STYLE,
        choices=("axis_locked", "full"),
        help="axis_locked (default): after 0, current hand position/pose relative to the 0-reference "
        "drives one ±world X/Y/Z translation or rotation; full: legacy multi-axis 6DoF with dom gates.",
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
        "--left-axis-cam-deadzone-mm",
        type=float,
        nargs=3,
        default=list(_DEFAULT_AXIS_CAM_DEADZONE_XYZ_MM),
        metavar=("DX", "DY", "DZ"),
        help="axis_locked: legacy; gating now uses lateral/forward rules + world snap.",
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
        "--left-axis-rot-excl-ratio",
        type=float,
        default=_DEFAULT_AXIS_ROT_EXCL_RATIO,
        help="axis_locked: rotation wins if rot_score >= this × trans_score.",
    )
    parser.add_argument(
        "--left-axis-trans-excl-ratio",
        type=float,
        default=_DEFAULT_AXIS_TRANS_EXCL_RATIO,
        help="axis_locked: translation wins if trans_score >= this × rot_score.",
    )
    parser.add_argument(
        "--left-axis-forward-boost",
        type=float,
        default=_DEFAULT_AXIS_FORWARD_BOOST,
        help="axis_locked: gain on camera dz-only pans (maps to world Y).",
    )
    parser.add_argument(
        "--left-axis-forward-min-dz-mm",
        type=float,
        default=_DEFAULT_AXIS_FORWARD_MIN_DZ_MM,
        help="axis_locked: min |dz| mm to treat as near/far (lower = more sensitive).",
    )
    parser.add_argument(
        "--left-axis-rot-boost",
        type=float,
        default=_DEFAULT_AXIS_ROT_BOOST,
        help="axis_locked: multiply rotation when rotate-dominant (more sensitive spin).",
    )
    parser.add_argument(
        "--left-axis-lateral-min-dx-mm",
        type=float,
        default=_DEFAULT_AXIS_LATERAL_MIN_DX_MM,
        help="axis_locked: camera mm; lateral pan zeros spurious dz above this |dx|.",
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
        "--left-depth-dom-ratio",
        type=float,
        default=_DEFAULT_LEFT_DEPTH_DOM_RATIO,
        help="If >0 and |dz| dominates hypot(dx,dy), translation uses only camera [0,0,dz] mm (perspective coupling). 0 = off.",
    )
    parser.add_argument(
        "--left-depth-dom-min-mm",
        type=float,
        default=_DEFAULT_LEFT_DEPTH_DOM_MIN_MM,
        help="Minimum |dz| (mm) before depth-dominant translation gating can apply (used with --left-depth-dom-ratio).",
    )
    parser.add_argument(
        "--left-vertical-dom-ratio",
        type=float,
        default=_DEFAULT_LEFT_VERTICAL_DOM_RATIO,
        help="If >0 and |dy| dominates hypot(dx,dz), translation uses only [0,dy,0] mm (reduces vertical→horizontal leak). 0 = off.",
    )
    parser.add_argument(
        "--left-vertical-dom-min-mm",
        type=float,
        default=_DEFAULT_LEFT_VERTICAL_DOM_MIN_MM,
        help="Minimum |dy| (mm) before vertical-dominant dz gating can apply.",
    )
    parser.add_argument(
        "--left-vertical-optical-preserve-ratio",
        type=float,
        default=_DEFAULT_LEFT_VERTICAL_OPTICAL_PRESERVE_RATIO,
        help="If >0 and |dz| dominates hypot(dx,dy), skip vertical dz stripping (keeps near/far). 0 = off.",
    )
    parser.add_argument(
        "--left-vertical-optical-preserve-min-mm",
        type=float,
        default=_DEFAULT_LEFT_VERTICAL_OPTICAL_PRESERVE_MIN_MM,
        help="Minimum |dz| (mm) before optical-preserve can skip vertical dz stripping.",
    )
    parser.add_argument(
        "--left-trans-lateral-dom-ratio",
        type=float,
        default=_DEFAULT_LEFT_TRANS_LATERAL_DOM_RATIO,
        help="If >0 and |dx| dominates hypot(dy,dz), translation uses only [dx,0,0] mm (horizontal pan). 0 = off.",
    )
    parser.add_argument(
        "--left-trans-lateral-dom-min-mm",
        type=float,
        default=_DEFAULT_LEFT_TRANS_LATERAL_DOM_MIN_MM,
        help="Minimum |dx| (mm) before lateral-dominant translation gating can apply.",
    )
    parser.add_argument(
        "--left-trans-xy-dom-strip-dz-ratio",
        type=float,
        default=_DEFAULT_LEFT_TRANS_XY_DOM_STRIP_DZ_RATIO,
        help="If >0 and hypot(dx,dy) dominates |dz|, zero camera dz for translation (0=off). Reduces Y on lateral pan.",
    )
    parser.add_argument(
        "--left-trans-xy-dom-strip-dz-min-h-mm",
        type=float,
        default=_DEFAULT_LEFT_TRANS_XY_DOM_STRIP_DZ_MIN_H_MM,
        help="Minimum hypot(dx,dy) (mm) before xy-dominated dz strip; lower catches slow pans.",
    )
    parser.add_argument(
        "--left-trans-dx-dom-strip-dz-ratio",
        type=float,
        default=_DEFAULT_LEFT_TRANS_DX_DOM_STRIP_DZ_RATIO,
        help="If >0 and |dx|>=ratio*|dz| with |dx|>=min, zero dz before depth_dom (slow lateral). 0=off.",
    )
    parser.add_argument(
        "--left-trans-dx-dom-strip-dz-min-dx-mm",
        type=float,
        default=_DEFAULT_LEFT_TRANS_DX_DOM_STRIP_DZ_MIN_DX_MM,
        help="Minimum |dx| (mm) before dx-vs-dz dz strip can apply.",
    )
    parser.add_argument(
        "--left-trans-boost-optical",
        type=float,
        default=_DEFAULT_LEFT_TRANS_BOOST_OPTICAL,
        help="Multiplier on camera dz (mm) after translation gates, before world map (fwd_y: stronger near/far).",
    )
    parser.add_argument(
        "--left-rot-planar-dom-ratio",
        type=float,
        default=_DEFAULT_LEFT_ROT_PLANAR_DOM_RATIO,
        help="If >0 and hypot(dx,dy) dominates |dz| in translation mm, palm rotation is zeroed that frame. 0 = off.",
    )
    parser.add_argument(
        "--left-rot-planar-dom-min-mm",
        type=float,
        default=_DEFAULT_LEFT_ROT_PLANAR_DOM_MIN_MM,
        help="Minimum image-plane hypot(dx,dy) (mm) before planar rotation suppression can apply.",
    )
    parser.add_argument(
        "--left-rot-palm-face-twist-world-y",
        action="store_true",
        help="Enable palm-facing + in-plane twist → world Y (fwd_y + Orbbec depth). Off by default (noisy during pans).",
    )
    parser.add_argument(
        "--left-rot-palm-face-cos-align-min",
        type=float,
        default=_DEFAULT_LEFT_ROT_PALM_FACE_COS_ALIGN_MIN,
        help="Min |palm normal · optical| (0..1) for palm–image-plane alignment before twist→world-Y.",
    )
    parser.add_argument(
        "--left-rot-palm-face-twist-dom-ratio",
        type=float,
        default=_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_DOM_RATIO,
        help="Require |ω_cam,z| ≥ ratio * hypot(ω_cam,x, ω_cam,y) for twist path; 0 = alignment only.",
    )
    parser.add_argument(
        "--left-rot-palm-face-twist-min-rad",
        type=float,
        default=_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_MIN_RAD,
        help="Min ‖R_to_rotvec(R_cam)‖ (rad) before twist path can apply.",
    )
    parser.add_argument(
        "--left-rot-palm-face-twist-world-y-sign",
        type=float,
        default=_DEFAULT_LEFT_ROT_PALM_FACE_TWIST_WORLD_Y_SIGN,
        help="±1 flip on in-plane twist → world ``Y`` angle if CW/CCW feels inverted.",
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
        help="centroid: whole formation spins about its center (整体自转, default). "
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
        action="store_true",
        help="Palm rotation follows hand more visibly: relax Z damp, gate, planar/coex/tau, plane snap cut; "
        "still uses R_cam=B@refᵀ and ω=R_to_rotvec(M R Mᵀ) from --left-cam-preset.",
    )
    parser.add_argument(
        "--left-palm-basis",
        type=str,
        default=_DEFAULT_LEFT_PALM_BASIS,
        choices=tuple(sorted(LEFT_PALM_BASIS_PRESETS)),
        help="Two MCPs (minus wrist) define palm rotation frame: index_ring (default, wider span), index_middle, middle_ring.",
    )
    parser.add_argument(
        "--left-rot-coex-trans-min-mm",
        type=float,
        default=_DEFAULT_LEFT_ROT_COEX_TRANS_MIN_MM,
        help="If wrist step (mm/frame) >= this and intrinsic palm angle < coex max, rotation is zeroed that frame. 0 = off.",
    )
    parser.add_argument(
        "--left-rot-coex-max-angle-deg",
        type=float,
        default=_DEFAULT_LEFT_ROT_COEX_MAX_ANGLE_DEG,
        help="Max intrinsic palm angle (deg, axis-angle norm) still treated as translation noise when coex trans min is on.",
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
        help="Print left-pose debug every N frames (default 2).",
    )
    parser.add_argument("--no-webcam", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--record-trajectory",
        type=str,
        default=None,
        metavar="PATH",
        help="Save per-frame setpoint/cmd/sim state to NPZ for offline axswarm replay.",
    )
    args = parser.parse_args()

    n_default = int(max(8, int(args.point_count)))
    if sys.stdin.isatty():
        point_count = int(prompt_and_init_fixed_surface_points(default_n=n_default))
    else:
        init_fixed_surface_points(n_default)
        point_count = n_default
        print(f"Fixed surface samples initialized (non-interactive): n={point_count}")
    scale = ScaleConfig(
        xy_radius=float(args.xy_radius),
        z_center=float(args.z_center),
        z_amplitude=float(args.z_amplitude),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        reference_xy_extent_mm=float(args.reference_xy_extent_mm),
        reference_z_extent_mm=float(args.reference_z_extent_mm),
        z_mm_scale=float(args.z_mm_scale),
    )
    live_target = make_initial_live_target(
        point_count=point_count,
        radius_mm=float(args.radius_mm),
        morph_mode=int(args.mode),
        open_alpha=float(args.open_alpha),
        shape_t=args.shape_t,
        scale=scale,
    )
    if args.print_only:
        return

    left_trans_scale = args.left_trans_scale
    if left_trans_scale is None:
        left_trans_scale = _DEFAULT_LEFT_TRANS_SCALE_MM
    # Default matches _DEFAULT_LEFT_AXIS_SIGN (Z flipped for altitude). Use --no-left-flip-z for +Z.
    left_axis_sign = (
        -1.0 if args.left_flip_x else 1.0,
        -1.0 if args.left_flip_y else 1.0,
        1.0 if args.left_flip_z else -1.0,
    )

    if not args.no_webcam:
        run_integrated_online_control(
            live_target=live_target,
            point_count=point_count,
            duration=float(args.duration),
            fps=int(args.fps),
            target_alpha=float(args.target_alpha),
            min_separation_m=float(args.min_separation_m),
            camera_index=int(args.camera),
            camera_buffer=int(args.camera_buffer),
            model_path=args.model,
            plot_every_n=int(args.plot_every),
            scale=scale,
            trail_every_n=int(args.trail_every),
            led_every_n=int(args.led_every),
            sim_render_every=int(args.sim_render_every),
            morph_radius_mm=float(args.radius_mm),
            drone_model=str(args.drone_model),
            input_backend=str(args.input_backend),
            state_freq=int(args.state_freq),
            max_sim_substeps=int(args.max_sim_substeps),
            imshow_every=int(args.imshow_every),
            raw_target_ema=float(args.raw_target_ema),
            max_target_step_m=float(args.max_target_step_m),
            left_swarm_pose=bool(args.left_swarm_pose),
            left_trans_scale=float(left_trans_scale),
            left_rot_scale=float(args.left_rot_scale),
            left_trans_ema=float(args.left_trans_ema),
            left_rot_ema=float(args.left_rot_ema),
            left_max_offset_m=float(args.left_max_offset_m),
            left_max_rot_rad=float(args.left_max_rot_rad),
            left_axis_sign=left_axis_sign,
            left_lost_decay=float(args.left_lost_decay),
            debug_drone_targets_every=int(args.debug_drone_targets_every),
            open_jump_reset=float(args.open_jump_reset),
            left_unwind_s=float(args.left_unwind_seconds),
            left_swarm_depth_frame_motion=bool(args.left_swarm_depth_frame_motion),
            left_rot_gate_rad=float(np.radians(float(args.left_rot_gate_deg))),
            left_yaw_min_horiz=float(args.left_yaw_min_horiz),
            left_rot_gain=float(args.left_rot_gain),
            left_rot_trans_tau_mm=float(args.left_rot_trans_tau_mm),
            left_rot_world_z_scale=float(args.left_rot_world_z_scale),
            left_cam_y_to_world_z=float(args.left_cam_y_to_world_z),
            left_cam_preset=str(args.left_cam_preset).strip().lower(),
            left_world_frame=str(args.left_world_frame).strip().lower(),
            left_control_style=str(args.left_control_style).strip().lower(),
            left_axis_trans_deadzone_m=float(args.left_axis_trans_deadzone_m),
            left_axis_rot_deadzone_rad=float(np.radians(args.left_axis_rot_deadzone_deg)),
            left_axis_cam_deadzone_xyz_mm=tuple(float(x) for x in args.left_axis_cam_deadzone_mm),
            left_axis_cam_snap_min_ratio=float(_DEFAULT_AXIS_CAM_SNAP_MIN_RATIO),
            left_axis_trans_on_m=float(args.left_axis_trans_on_m),
            left_axis_rot_on_rad=float(np.radians(args.left_axis_rot_on_deg)),
            left_axis_rot_excl_ratio=float(args.left_axis_rot_excl_ratio),
            left_axis_trans_excl_ratio=float(args.left_axis_trans_excl_ratio),
            left_axis_rot_boost=float(args.left_axis_rot_boost),
            left_axis_forward_boost=float(args.left_axis_forward_boost),
            left_axis_lateral_min_dx_mm=float(args.left_axis_lateral_min_dx_mm),
            left_axis_forward_min_dz_mm=float(args.left_axis_forward_min_dz_mm),
            install_hotkey_deps=bool(args.install_hotkey_deps),
            left_depth_dom_ratio=float(args.left_depth_dom_ratio),
            left_depth_dom_min_mm=float(args.left_depth_dom_min_mm),
            left_palm_basis=str(args.left_palm_basis).strip().lower(),
            left_rot_coex_trans_min_mm=float(args.left_rot_coex_trans_min_mm),
            left_rot_coex_max_angle_deg=float(args.left_rot_coex_max_angle_deg),
            left_vertical_dom_ratio=float(args.left_vertical_dom_ratio),
            left_vertical_dom_min_mm=float(args.left_vertical_dom_min_mm),
            left_vertical_optical_preserve_ratio=float(args.left_vertical_optical_preserve_ratio),
            left_vertical_optical_preserve_min_mm=float(args.left_vertical_optical_preserve_min_mm),
            left_trans_lateral_dom_ratio=float(args.left_trans_lateral_dom_ratio),
            left_trans_lateral_dom_min_mm=float(args.left_trans_lateral_dom_min_mm),
            left_trans_xy_dom_strip_dz_ratio=float(args.left_trans_xy_dom_strip_dz_ratio),
            left_trans_xy_dom_strip_dz_min_h_mm=float(args.left_trans_xy_dom_strip_dz_min_h_mm),
            left_trans_dx_dom_strip_dz_ratio=float(args.left_trans_dx_dom_strip_dz_ratio),
            left_trans_dx_dom_strip_dz_min_dx_mm=float(args.left_trans_dx_dom_strip_dz_min_dx_mm),
            left_trans_boost_optical=float(args.left_trans_boost_optical),
            left_rot_planar_dom_ratio=float(args.left_rot_planar_dom_ratio),
            left_rot_planar_dom_min_mm=float(args.left_rot_planar_dom_min_mm),
            left_rot_palm_face_twist_world_y=bool(args.left_rot_palm_face_twist_world_y),
            left_rot_palm_face_cos_align_min=float(args.left_rot_palm_face_cos_align_min),
            left_rot_palm_face_twist_dom_ratio=float(args.left_rot_palm_face_twist_dom_ratio),
            left_rot_palm_face_twist_min_rad=float(args.left_rot_palm_face_twist_min_rad),
            left_rot_palm_face_twist_world_y_sign=float(args.left_rot_palm_face_twist_world_y_sign),
            left_plane_rot_scale_mul=float(args.left_plane_rot_scale_mul),
            left_rot_direct_follow=bool(args.left_rot_direct_follow),
            left_rot_pivot=str(args.left_rot_pivot).strip().lower(),
            left_dual_webcam_rot=not bool(args.no_left_dual_webcam_rot),
            left_rot_webcam_vis_thresh=float(args.left_rot_webcam_vis_thresh),
            left_rot_webcam_index=int(args.left_rot_webcam_index),
            orbbec_flip_horizontal=bool(args.orbbec_flip_horizontal),
            orbbec_use_transformed_depth=bool(args.orbbec_use_transformed_depth),
            orbbec_hand_swap=str(args.orbbec_hand_swap).strip().lower(),
            formation_rigid_3d_debug=bool(args.formation_rigid_3d_debug),
            left_pose_frame_viz=not bool(args.no_left_pose_frame_viz),
            left_pose_frame_viz_every=int(args.left_pose_frame_viz_every),
            left_pose_debug=bool(args.left_pose_debug),
            left_pose_debug_every=int(args.left_pose_debug_every),
            webcam_rot_stride=int(args.webcam_rot_stride),
            show_webcam_preview=bool(args.show_webcam_preview),
            global_hotkeys=not bool(args.no_global_hotkeys),
            mode_vis_min=float(args.mode_vis_min),
            record_trajectory=args.record_trajectory,
        )
    else:
        stop_event = Event()
        run_online_crazyflow(
            target_provider=live_target.get,
            point_count=point_count,
            duration=float(args.duration),
            fps=int(args.fps),
            target_alpha=float(args.target_alpha),
            min_separation_m=float(args.min_separation_m),
            stop_event=stop_event,
            morph_mode=int(args.mode),
            drone_model=str(args.drone_model),
        )


if __name__ == "__main__":
    main()
