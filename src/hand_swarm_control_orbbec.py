import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from pyk4a import Config, FPS, PyK4A

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGERTIP_IDS = [4, 8, 12, 16, 20]
MCP_IDS = [5, 9, 13, 17]
WRIST_ID = 0

# Keep consistent with iso_swarm/src/hand_tracking_orbbec.py
OPEN_GAMMA = 1.8
PLANE_SNAP_ON = 0.88
PLANE_SNAP_OFF = 0.82
SPHERE_SNAP_ON = 0.12
SPHERE_SNAP_OFF = 0.18


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def rate_limit(prev: float, new: float, max_delta: float) -> float:
    if new > prev + max_delta:
        return prev + max_delta
    if new < prev - max_delta:
        return prev - max_delta
    return new


def world_landmarks_to_mm(world_lms) -> np.ndarray:
    points = []
    for lm in world_lms:
        points.append([lm.x * 1000.0, -lm.y * 1000.0, -lm.z * 1000.0])
    return np.asarray(points, dtype=np.float32)


def compute_open_from_world(points_mm: np.ndarray) -> tuple[float, float, float, float]:
    """
    Returns (open_free, planarity, isotropy, finger_spread).
    open_free in [0,1]: 0=sphere(fist), 1=plane(open).
    """
    fit_ids = [i for i in range(points_mm.shape[0]) if i != WRIST_ID]
    fit_pts = points_mm[fit_ids]
    centroid = fit_pts.mean(axis=0)
    centered = fit_pts - centroid

    cov = np.cov(centered.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]
    lamb_sum = float(np.sum(eigvals)) + 1e-8
    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-8))
    isotropy = float(eigvals[2] / lamb_sum)

    hand_scale = float(np.mean(np.linalg.norm(centered, axis=1)) + 1e-6)
    palm_center = points_mm[[WRIST_ID, *MCP_IDS]].mean(axis=0)
    tip_dist = np.linalg.norm(points_mm[FINGERTIP_IDS] - palm_center, axis=1)
    finger_spread = float(np.mean(tip_dist) / hand_scale)

    spread_score = clamp01((finger_spread - 1.00) / (1.65 - 1.00))
    planarity_score = clamp01((planarity - 0.12) / (0.55 - 0.12))
    isotropy_score = clamp01((isotropy - 0.06) / (0.22 - 0.06))  # 1 => more sphere

    open_free = clamp01(0.50 * spread_score + 0.35 * planarity_score + 0.15 * (1.0 - isotropy_score))
    open_free = clamp01(open_free ** OPEN_GAMMA)
    return open_free, planarity, isotropy, finger_spread


def draw_2d_hand(frame: np.ndarray, hand_landmarks, open_score: float, planarity: float, spread: float) -> None:
    h, w, _ = frame.shape
    points = []
    for lm in hand_landmarks:
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 0, 0), 1)

    cv2.putText(
        frame,
        f"open={open_score:.2f} planarity={planarity:.2f} spread={spread:.2f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def make_plane_sphere_morph_targets(
    n_drones: int,
    open_score: float,
    center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
) -> np.ndarray:
    """
    3D plane <-> full-sphere morph (no circular orbit over time).

    open_score=1 : points on a horizontal plane (flat blanket).
    open_score=0 : points distributed on a full sphere (closed fist).

    Drones keep fixed indices; only the surface geometry changes with the gesture.
    """
    open_score = clamp01(float(open_score))
    sphere_strength = 1.0 - open_score

    # Larger radius when open (blanket spread), smaller when closed (sphere compact).
    morph_radius = 0.22 + 0.18 * open_score
    # Keep the whole sphere safely above ground (raise formation).
    base_z = 1.12 + 0.06 * open_score
    z_floor = 0.85

    golden = np.pi * (3.0 - np.sqrt(5.0))
    targets = np.zeros((n_drones, 3), dtype=np.float32)

    for i in range(n_drones):
        # Plane sample: sunflower disk in XY plane
        r_norm = np.sqrt((i + 0.5) / n_drones)
        theta = i * golden
        plane = np.array(
            [
                morph_radius * r_norm * np.cos(theta),
                morph_radius * r_norm * np.sin(theta),
                0.0,
            ],
            dtype=np.float32,
        )

        # Sphere sample: Fibonacci full-sphere point with same index
        # IMPORTANT: map (x,y,z) correctly so the "height" goes to world Z.
        z = 1.0 - 2.0 * (i + 0.5) / n_drones
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = i * golden
        sphere = morph_radius * np.array([r * np.cos(phi), r * np.sin(phi), z], dtype=np.float32)

        local = (1.0 - sphere_strength) * plane + sphere_strength * sphere
        targets[i] = center + np.array([local[0], local[1], base_z + local[2]], dtype=np.float32)

    targets[:, 2] = np.maximum(targets[:, 2], z_floor)
    return targets


def grid_2d_np(n: int, spacing: float = 0.25, center_xy: np.ndarray | None = None) -> np.ndarray:
    """Numpy version of crazyflow.utils.grid_2d for initial staging."""
    if center_xy is None:
        center_xy = np.zeros(2, dtype=np.float32)
    N = int(np.ceil(np.sqrt(n)))
    points = np.linspace(-0.5 * spacing * (N - 1), 0.5 * spacing * (N - 1), N, dtype=np.float32)
    x, y = np.meshgrid(points, points)
    grid = np.stack([x.flatten(), y.flatten()], axis=-1) + center_xy[None, :]
    order = np.argsort(np.linalg.norm(grid, axis=-1))
    return grid[order[:n]]


def min_pairwise_dist(points: np.ndarray) -> float:
    n = points.shape[0]
    md = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d < md:
                md = d
    return float(md if np.isfinite(md) else 0.0)


def scale_about_center(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return center + (points - center) * float(scale)


def enforce_min_separation(
    targets: np.ndarray,
    center: np.ndarray,
    min_sep: float,
    mode: str,
    max_scale: float = 2.5,
) -> tuple[np.ndarray, float, float]:
    """
    Enforce min separation with correct metric:
    - mode='plane': use XY distances (projection)
    - mode='sphere': use full 3D distances (true surface separation)
    Returns (adjusted_targets, min_dist_before, applied_scale).
    """
    adjusted = targets.copy()
    if mode == "plane":
        md = min_pairwise_dist(adjusted[:, :2])
        if md < min_sep:
            s = min(max_scale, (min_sep / max(md, 1e-6)) * 1.02)
            adjusted[:, :2] = scale_about_center(adjusted[:, :2], center[:2], s)
            return adjusted, md, s
        return adjusted, md, 1.0
    if mode == "sphere":
        md = min_pairwise_dist(adjusted)
        if md < min_sep:
            s = min(max_scale, (min_sep / max(md, 1e-6)) * 1.02)
            adjusted = scale_about_center(adjusted, center, s)
            return adjusted, md, s
        return adjusted, md, 1.0
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    model_path = "hand_landmarker.task"
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    k4a = PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30,
        )
    )

    n_worlds = 1
    n_drones = 24
    # NOTE: drawing trajectory lines adds MuJoCo geoms every frame and can crash with:
    # "Ran out of geoms. maxgeom: 1000". Keep trails off by default.
    draw_trails = False
    # Match controller update rate to camera loop (~30Hz) to avoid control/pipeline mismatch.
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.state, state_freq=30)
    sim.reset()
    # === Align with test1_crazyflow.py style: start above ground ===
    # Sim.reset() arranges drones on a wide XY grid, but Z can be near ground.
    # If we command complex formations immediately, many drones crash.
    # Use a compact staging formation first (prevents long-distance "rush to center")
    takeoff_z = 1.25
    takeoff_sec = 3.0
    gather_sec = 4.0
    ramp_sec = 7.0

    # Center formation around current mean XY (reduces translation).
    reset_pos = np.array(sim.data.states.pos[0], dtype=np.float32)
    formation_center = reset_pos.mean(axis=0).astype(np.float32)
    formation_center[2] = 0.0

    stage_xy = grid_2d_np(n_drones, spacing=0.35, center_xy=formation_center[:2])
    stage_pos = np.zeros((n_drones, 3), dtype=np.float32)
    stage_pos[:, :2] = stage_xy
    stage_pos[:, 2] = takeoff_z

    open_score_ema = 0.5
    open_score_smooth = 0.2
    open_rate_limit_per_sec = 0.9  # limit how fast gesture can change
    snap_state = None  # None / "plane" / "sphere"
    last_hand_time = None
    last_open_out = 1.0
    target_smooth = None
    target_smooth_alpha = 0.08
    max_step_per_sec = 0.35  # meters/s limit on target changes per drone
    pos_buffer = deque(maxlen=20)

    with HandLandmarker.create_from_options(options) as landmarker:
        k4a.start()
        print("Running hand->swarm control. Press q to quit.")
        start_t = time.time()
        prev_time = start_t
        frame_idx = 0
        try:
            while True:
                now = time.time()
                t = now - start_t
                dt = max(1e-3, now - prev_time)
                prev_time = now

                capture = k4a.get_capture()
                if capture.color is None:
                    continue

                color = capture.color
                frame = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR) if (color.ndim == 3 and color.shape[2] == 4) else color
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int(t * 1000))

                planarity = 0.0
                spread = 0.0
                isotropy = 0.0
                hand_detected = bool(result.hand_landmarks and result.hand_world_landmarks)
                if hand_detected:
                    hand_lm_2d = result.hand_landmarks[0]
                    hand_lm_3d = result.hand_world_landmarks[0]
                    points_mm = world_landmarks_to_mm(hand_lm_3d)
                    open_free, planarity, isotropy, spread = compute_open_from_world(points_mm)
                    open_free = rate_limit(open_score_ema, float(open_free), max_delta=open_rate_limit_per_sec * dt)
                    open_score_ema = open_score_smooth * open_free + (1.0 - open_score_smooth) * open_score_ema

                    # snap based on free/smoothed open_score_ema (do not lock input)
                    if snap_state == "plane":
                        if open_score_ema < PLANE_SNAP_OFF:
                            snap_state = None
                    elif snap_state == "sphere":
                        if open_score_ema > SPHERE_SNAP_OFF:
                            snap_state = None
                    else:
                        if open_score_ema > PLANE_SNAP_ON:
                            snap_state = "plane"
                        elif open_score_ema < SPHERE_SNAP_ON:
                            snap_state = "sphere"

                    open_out = float(open_score_ema)
                    if snap_state == "plane":
                        open_out = 1.0
                    elif snap_state == "sphere":
                        open_out = 0.0
                    last_open_out = float(open_out)
                    last_hand_time = now

                    draw_2d_hand(frame, hand_lm_2d, open_out, planarity, spread)
                    cv2.putText(
                        frame,
                        f"iso={isotropy:.2f}" + (f" SNAP:{snap_state.upper()}" if snap_state else ""),
                        (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                else:
                    # If no hand, DO NOT change the formation. Keep last known output.
                    open_out = float(last_open_out)
                    cv2.putText(
                        frame,
                        "No hand detected, hold last command",
                        (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 180, 255),
                        2,
                    )

                center = formation_center.copy()
                raw_target = None
                sep_md = 0.0
                sep_scale = 1.0
                if hand_detected:
                    raw_target = make_plane_sphere_morph_targets(
                        n_drones=n_drones,
                        open_score=open_out,
                        center=center,
                    )
                if raw_target is not None:
                    # Correct min-distance metric:
                    # plane state -> XY spacing; sphere state -> 3D spacing (avoid huge scale from XY pole overlap)
                    mode = "sphere" if open_out < 0.55 else "plane"
                    min_sep = 0.20 if n_drones >= 20 else 0.18
                    raw_target, sep_md, sep_scale = enforce_min_separation(
                        raw_target, center=center, min_sep=min_sep, mode=mode
                    )
                    if target_smooth is None:
                        target_smooth = raw_target.copy()
                    else:
                        # Smooth + per-step limiter to avoid "crazy" jumps.
                        blended = target_smooth_alpha * raw_target + (1.0 - target_smooth_alpha) * target_smooth
                        max_step = max_step_per_sec * dt
                        delta = blended - target_smooth
                        # limit per-drone step by vector norm (not per-axis) to avoid zig-zag instability
                        dnorm = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
                        scale = np.minimum(1.0, max_step / dnorm)
                        target_smooth = target_smooth + delta * scale
                elif target_smooth is None:
                    target_smooth = stage_pos.copy()

                # === 3-stage plan ===
                # 1) Takeoff: lift in place (keep reset XY), to avoid collisions during ascent
                # 2) Gather: move into compact staging grid at constant altitude
                # 3) Morph: ramp into morph formation
                reset_xy = reset_pos[:, :2]
                takeoff_pos = stage_pos.copy()
                takeoff_pos[:, :2] = reset_xy

                if t < takeoff_sec:
                    target_cmd = (1.0 - min(1.0, t / takeoff_sec)) * reset_pos + min(1.0, t / takeoff_sec) * takeoff_pos
                elif t < (takeoff_sec + gather_sec):
                    gg = min(1.0, (t - takeoff_sec) / gather_sec)
                    target_cmd = (1.0 - gg) * takeoff_pos + gg * stage_pos
                else:
                    ramp = min(1.0, (t - takeoff_sec - gather_sec) / ramp_sec)
                    target_cmd = (1.0 - ramp) * stage_pos + ramp * target_smooth

                cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
                cmd[..., :3] = target_cmd
                cmd[..., 6:10] = [0.0, 0.0, 0.0, 1.0]
                sim.state_control(cmd)
                sim.step(sim.freq // sim.control_freq)

                if frame_idx % 5 == 0:
                    pos_buffer.append(sim.data.states.pos[0].copy())

                # Trails are optional; enabling them continuously may exceed MuJoCo maxgeom.
                if draw_trails and (frame_idx % 30 == 0) and len(pos_buffer) > 1:
                    lines = np.array(pos_buffer)
                    for d in range(n_drones):
                        draw_line(
                            sim,
                            lines[:, d, :],
                            rgba=[0.0, 1.0, 0.0, 1.0],
                            start_size=0.3,
                            end_size=0.9,
                        )
                sim.render()

                cv2.putText(
                    frame,
                    f"hand={int(hand_detected)} open={open_out:.2f} free={open_score_ema:.2f} sep_scale={sep_scale:.2f} md={sep_md:.2f}",
                    (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # Print debug once per second
                if (frame_idx % 30) == 0:
                    pos = np.array(sim.data.states.pos[0], dtype=np.float32)
                    zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
                    if target_smooth is not None:
                        tzmin, tzmax = float(target_smooth[:, 2].min()), float(target_smooth[:, 2].max())
                    else:
                        tzmin, tzmax = (0.0, 0.0)
                    print(
                        f"[t={t:6.2f}] hand={hand_detected} open_out={open_out:.2f} free={open_score_ema:.2f} "
                        f"pos_z=[{zmin:.2f},{zmax:.2f}] tgt_z=[{tzmin:.2f},{tzmax:.2f}] sep_scale={sep_scale:.2f} md={sep_md:.2f}"
                    )
                cv2.imshow("Orbbec Hand Input", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                frame_idx += 1
        finally:
            k4a.stop()
            sim.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
