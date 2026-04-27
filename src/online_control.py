"""Online Crazyflow control using the same morph target logic as webcam_main.

The first test mode intentionally keeps the target fixed at mode=1, open=1.0.
This validates Crazyflow startup and the initial target mapping before wiring in
live MediaPipe hand updates.
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable

import cv2
import jax.numpy as jnp
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from runtime.hand_tracking_webcam_modes import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    LpShapePipelineState,
    MCP_IDS,
    MODE_DEBOUNCE_FRAMES,
    MORPH_PLANE_RADIUS_A,
    MORPH_PLANE_RADIUS_B,
    PLANE_SNAP_OFF,
    PLANE_SNAP_ON,
    PLOT_EVERY_N_FRAMES,
    RunningMode,
    SPHERE_SNAP_OFF,
    SPHERE_SNAP_ON,
    WRIST_ID,
    advance_lp_shape_p,
    analyze_hand_topology,
    classify_mode_from_fingers,
    draw_all_hands,
    extract_world_points_mm_result,
    find_left_right_indices,
    index_mcp_tip_segment_norm,
    mode_epsilon_pair,
    resolve_model_path,
    shared_update_mode_state,
    shared_update_open_state,
    update_3d_plot,
)
from shared.modes_runtime import ModeState, RightHandState
from shared.morph_renderers import init_fixed_surface_points, mapped_fixed_surface_points


@dataclass
class ScaleConfig:
    xy_radius: float
    z_center: float
    z_amplitude: float
    z_min: float
    z_max: float
    reference_xy_extent_mm: float
    reference_z_extent_mm: float


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


def closest_pair(points: np.ndarray) -> tuple[float, int, int]:
    """Return distance and indices of the closest pair in one target frame."""
    best_dist = float("inf")
    best_i = -1
    best_j = -1
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            dist = float(np.linalg.norm(points[i] - points[j]))
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_j = j
    return best_dist, best_i, best_j


def normalize_morph_points(points_mm: np.ndarray, scale: ScaleConfig) -> np.ndarray:
    """Map morph-renderer millimeter targets into the Crazyflow workspace."""
    pts = np.asarray(points_mm, dtype=float)
    xy_scale = max(float(scale.reference_xy_extent_mm), 1.0)
    z_scale = max(float(scale.reference_z_extent_mm), 1.0)

    out = np.zeros_like(pts)
    out[:, 0] = pts[:, 0] / xy_scale * float(scale.xy_radius)
    out[:, 1] = pts[:, 1] / xy_scale * float(scale.xy_radius)
    out[:, 2] = float(scale.z_center) + pts[:, 2] / z_scale * float(scale.z_amplitude)
    out[:, 2] = np.clip(out[:, 2], float(scale.z_min), float(scale.z_max))
    return out.astype(np.float32)


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
    print("raw_mm=" + " ".join(f"{k}:({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for k, p in enumerate(points_mm)))
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
    print("raw_mm=" + " ".join(f"{k}:({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for k, p in enumerate(points_mm)))
    print(f"Closest initial target spacing: pair=({i},{j}), dist={dist:.2f}m")
    state = LiveTargetState(target)
    state.mode = int(morph_mode)
    state.open_alpha = float(open_alpha)
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
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model),
        running_mode=RunningMode.VIDEO,
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
    plot_every_n = max(1, int(plot_every_n))

    plt.ion()
    fig = plt.figure("Online Control Webcam + 3D")
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")

    print(
        f"MediaPipe webcam started on camera {camera_index}. "
        "Left hand = MODE, right hand = OPEN. Press q in webcam window to stop."
    )
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
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
                    debounce_frames=MODE_DEBOUNCE_FRAMES,
                    mode_smooth=0.22,
                )
                active_mode = int(mode_state.morph_mode)
                advance_lp_shape_p(dist_norm, active_mode, lp_shape)

                pts_r = extract_world_points_mm_result(result, idx_r) if idx_r is not None else None
                hands_3d = []
                if pts_r is not None:
                    right_state.last_right_pts = list(pts_r)
                    hands_3d = [pts_r]
                elif right_state.last_right_pts is not None:
                    hands_3d = [right_state.last_right_pts]

                open_out = shared_update_open_state(
                    pts_r,
                    right_state=right_state,
                    analyze_topology_fn=analyze_hand_topology,
                    open_smooth=0.18,
                    plane_snap_on=PLANE_SNAP_ON,
                    plane_snap_off=PLANE_SNAP_OFF,
                    sphere_snap_on=SPHERE_SNAP_ON,
                    sphere_snap_off=SPHERE_SNAP_OFF,
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

                analyses = None
                if hands_3d and (frame_idx % plot_every_n) == 0:
                    analyses = update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_mode=mode_state.morph_mode,
                        morph_alpha_smoothed=open_out,
                        control_label="online open+p",
                        mode_shape_t=lp_shape.left_shape_t_ema,
                        epsilon_pair_display=lp_shape.epsilon_pair_display,
                        lp_show_refs=True,
                    )
                    try:
                        fig.canvas.flush_events()
                    except Exception:
                        pass
                    plt.pause(0.001)

                if analyses:
                    a0 = analyses[0]
                    open_v = float(open_out if open_out is not None else a0["morph_alpha"])
                    epsilon1, epsilon2 = mode_epsilon_pair(
                        int(mode_state.morph_mode),
                        lp_shape.left_shape_t_ema,
                    )
                    points_mm = mapped_fixed_surface_points(
                        radius=float(a0["radius"]),
                        open_alpha=open_v,
                        epsilon1=epsilon1,
                        epsilon2=epsilon2,
                        plane_radius_a=MORPH_PLANE_RADIUS_A,
                        plane_radius_b=MORPH_PLANE_RADIUS_B,
                        morph_mode=int(mode_state.morph_mode),
                    )
                    target = normalize_morph_points(points_mm, scale)
                    live_target.set(target, mode=int(mode_state.morph_mode), open_alpha=open_v)

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
                if key == ord("q"):
                    stop_event.set()
                    break
                frame_idx += 1
    finally:
        cap.release()
        plt.ioff()
        plt.close(fig)
        cv2.destroyWindow("Online Control Webcam")


def run_online_crazyflow(
    target_provider: Callable[[], np.ndarray],
    point_count: int,
    duration: float,
    fps: int,
    target_alpha: float,
    stop_event: Event | None = None,
) -> None:
    """Run Crazyflow and continuously track targets from target_provider."""
    n_worlds = 1
    n_drones = int(point_count)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.state)
    sim.reset()

    pos_buffer = deque(maxlen=16)
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

            smooth_target = target_alpha * raw_target + (1.0 - target_alpha) * smooth_target

            cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
            cmd[..., :3] = smooth_target
            cmd[..., 9] = 0.0

            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)

            if i % 8 == 0:
                pos_buffer.append(sim.data.states.pos[0].copy())

            if ((i * fps) % sim.control_freq) < fps:
                if len(pos_buffer) > 1:
                    lines = np.asarray(pos_buffer)
                    for d in range(n_drones):
                        draw_line(
                            sim,
                            lines[:, d, :],
                            rgba=colors[d].tolist(),
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
    camera_index: int,
    camera_buffer: int,
    model_path: str | None,
    plot_every_n: int,
    scale: ScaleConfig,
) -> None:
    """Run MediaPipe, Matplotlib, OpenCV, and Crazyflow in the main thread."""
    resolved_model = resolve_model_path(model_path, __file__)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {camera_index}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(camera_buffer))
    except Exception:
        pass

    n_worlds = 1
    n_drones = int(point_count)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.state)
    sim.reset()

    pos_buffer = deque(maxlen=16)
    first_target = live_target.get()
    first_target[:, 2] = np.maximum(first_target[:, 2], 1.10)
    smooth_target = first_target.copy()
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))

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
    plot_every_n = max(1, int(plot_every_n))
    control_steps_per_frame = max(1, int(round(sim.freq / max(float(fps), 1.0))))

    plt.ion()
    fig = plt.figure("Online Control Webcam + 3D")
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")

    print(
        f"MediaPipe webcam started on camera {camera_index}. "
        "Left hand = MODE, right hand = OPEN. Press q in webcam window to stop."
    )
    print("Holding default M1/open=1 target until s is pressed.")

    frame_idx = 0
    last_status_second = -1
    gesture_control_enabled = False
    start_time = time.monotonic()
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                elapsed = time.monotonic() - start_time
                if float(duration) > 0.0 and elapsed > float(duration):
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_ms = int(frame_idx * (1000 / max(float(fps), 1.0)))
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
                    debounce_frames=MODE_DEBOUNCE_FRAMES,
                    mode_smooth=0.22,
                )
                active_mode = int(mode_state.morph_mode)
                advance_lp_shape_p(dist_norm, active_mode, lp_shape)

                pts_r = extract_world_points_mm_result(result, idx_r) if idx_r is not None else None
                hands_3d = []
                if pts_r is not None:
                    right_state.last_right_pts = list(pts_r)
                    hands_3d = [pts_r]
                elif right_state.last_right_pts is not None:
                    hands_3d = [right_state.last_right_pts]

                open_out = shared_update_open_state(
                    pts_r,
                    right_state=right_state,
                    analyze_topology_fn=analyze_hand_topology,
                    open_smooth=0.18,
                    plane_snap_on=PLANE_SNAP_ON,
                    plane_snap_off=PLANE_SNAP_OFF,
                    sphere_snap_on=SPHERE_SNAP_ON,
                    sphere_snap_off=SPHERE_SNAP_OFF,
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

                analyses = None
                if hands_3d and (frame_idx % plot_every_n) == 0:
                    analyses = update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_mode=mode_state.morph_mode,
                        morph_alpha_smoothed=open_out,
                        control_label="online open+p",
                        mode_shape_t=lp_shape.left_shape_t_ema,
                        epsilon_pair_display=lp_shape.epsilon_pair_display,
                        lp_show_refs=True,
                    )
                    try:
                        fig.canvas.flush_events()
                    except Exception:
                        pass
                    plt.pause(0.001)

                if analyses and gesture_control_enabled:
                    a0 = analyses[0]
                    open_v = float(open_out if open_out is not None else a0["morph_alpha"])
                    epsilon1, epsilon2 = mode_epsilon_pair(
                        int(mode_state.morph_mode),
                        lp_shape.left_shape_t_ema,
                    )
                    points_mm = mapped_fixed_surface_points(
                        radius=float(a0["radius"]),
                        open_alpha=open_v,
                        epsilon1=epsilon1,
                        epsilon2=epsilon2,
                        plane_radius_a=MORPH_PLANE_RADIUS_A,
                        plane_radius_b=MORPH_PLANE_RADIUS_B,
                        morph_mode=int(mode_state.morph_mode),
                    )
                    target = normalize_morph_points(points_mm, scale)
                    live_target.set(target, mode=int(mode_state.morph_mode), open_alpha=open_v)

                status_second = int(elapsed)
                if status_second != last_status_second:
                    last_status_second = status_second
                    target_now = live_target.get()
                    min_dist, min_i, min_j = closest_pair(target_now)
                    open_txt = f"{float(open_out):.2f}" if open_out is not None else "-"
                    print(
                        f"online t={elapsed:.1f}s armed={'yes' if gesture_control_enabled else 'no'} "
                        f"mode={int(mode_state.morph_mode)} "
                        f"raw={mode_raw} open={open_txt} "
                        f"L={'yes' if idx_l is not None else 'no'} "
                        f"R={'yes' if idx_r is not None else 'no'} "
                        f"target_min=({min_i},{min_j}) {min_dist:.2f}m"
                    )

                raw_target = live_target.get()

                smooth_target = target_alpha * raw_target + (1.0 - target_alpha) * smooth_target
                cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
                cmd[..., :3] = smooth_target
                cmd[..., 9] = 0.0
                sim.state_control(cmd)
                sim.step(control_steps_per_frame)

                if frame_idx % 2 == 0:
                    pos_buffer.append(sim.data.states.pos[0].copy())
                if len(pos_buffer) > 1:
                    lines = np.asarray(pos_buffer)
                    for d in range(n_drones):
                        draw_line(
                            sim,
                            lines[:, d, :],
                            rgba=colors[d].tolist(),
                            start_size=0.5,
                            end_size=2.0,
                        )
                sim.render()

                cv2.putText(
                    frame,
                    f"ONLINE {'ARMED' if gesture_control_enabled else 'HOLD DEFAULT - press s'} "
                    f"M{mode_state.morph_mode} raw:{mode_raw} open:{open_out if open_out is not None else '-'} "
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
                if key == ord("q"):
                    break
                if key == ord("s"):
                    gesture_control_enabled = True
                    print("Gesture control armed: Crazyflow targets now follow mode/open recognition.")
                frame_idx += 1
    finally:
        cap.release()
        sim.close()
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point for online Crazyflow control."""
    parser = argparse.ArgumentParser(
        description="Run online Crazyflow control from webcam_main morph targets."
    )
    parser.add_argument("--point-count", type=int, default=24, help="Number of Crazyflow drones / morph points.")
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--open", type=float, default=1.0, dest="open_alpha")
    parser.add_argument("--shape-t", type=float, default=None)
    parser.add_argument("--radius-mm", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds; <=0 means run until q.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--target-alpha", type=float, default=0.06)
    parser.add_argument("--xy-radius", type=float, default=1.20)
    parser.add_argument("--z-center", type=float, default=1.40)
    parser.add_argument("--z-amplitude", type=float, default=0.35)
    parser.add_argument("--z-min", type=float, default=1.10)
    parser.add_argument("--z-max", type=float, default=2.10)
    parser.add_argument("--reference-xy-extent-mm", type=float, default=100.0)
    parser.add_argument("--reference-z-extent-mm", type=float, default=50.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-buffer", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--plot-every", type=int, default=PLOT_EVERY_N_FRAMES)
    parser.add_argument("--no-webcam", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    scale = ScaleConfig(
        xy_radius=float(args.xy_radius),
        z_center=float(args.z_center),
        z_amplitude=float(args.z_amplitude),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        reference_xy_extent_mm=float(args.reference_xy_extent_mm),
        reference_z_extent_mm=float(args.reference_z_extent_mm),
    )
    live_target = make_initial_live_target(
        point_count=int(args.point_count),
        radius_mm=float(args.radius_mm),
        morph_mode=int(args.mode),
        open_alpha=float(args.open_alpha),
        shape_t=args.shape_t,
        scale=scale,
    )
    if args.print_only:
        return

    if not args.no_webcam:
        run_integrated_online_control(
            live_target=live_target,
            point_count=int(args.point_count),
            duration=float(args.duration),
            fps=int(args.fps),
            target_alpha=float(args.target_alpha),
            camera_index=int(args.camera),
            camera_buffer=int(args.camera_buffer),
            model_path=args.model,
            plot_every_n=int(args.plot_every),
            scale=scale,
        )
    else:
        stop_event = Event()
        run_online_crazyflow(
            target_provider=live_target.get,
            point_count=int(args.point_count),
            duration=float(args.duration),
            fps=int(args.fps),
            target_alpha=float(args.target_alpha),
            stop_event=stop_event,
        )


if __name__ == "__main__":
    main()
