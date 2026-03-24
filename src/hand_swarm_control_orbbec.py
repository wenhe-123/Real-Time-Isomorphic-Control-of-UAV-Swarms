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


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def world_landmarks_to_mm(world_lms) -> np.ndarray:
    points = []
    for lm in world_lms:
        points.append([lm.x * 1000.0, -lm.y * 1000.0, -lm.z * 1000.0])
    return np.asarray(points, dtype=np.float32)


def compute_morph_alpha(points_mm: np.ndarray) -> tuple[float, float, float]:
    centroid = points_mm.mean(axis=0)
    centered = points_mm - centroid
    cov = np.cov(centered.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]
    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-8))

    hand_scale = float(np.mean(np.linalg.norm(centered, axis=1)) + 1e-6)
    palm_center = points_mm[[0, *MCP_IDS]].mean(axis=0)
    tip_dist = np.linalg.norm(points_mm[FINGERTIP_IDS] - palm_center, axis=1)
    finger_spread = float(np.mean(tip_dist) / hand_scale)

    # open_score in [0,1]: 0=fist(closed), 1=open(flat)
    spread_score = clamp01((finger_spread - 1.00) / (1.75 - 1.00))
    planarity_score = clamp01((planarity - 0.10) / (0.60 - 0.10))
    open_score = clamp01(0.55 * spread_score + 0.45 * planarity_score)
    return open_score, planarity, finger_spread


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


def make_circle_targets(
    n_drones: int,
    open_score: float,
    t: float,
    center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
) -> np.ndarray:
    # Mapping idea:
    # fist(open~0) -> compact/high/fast rotation
    # open(open~1) -> wide/lower/slower rotation
    radius = 0.08 + 0.30 * open_score
    z = 0.65 - 0.20 * open_score
    rot_speed = 1.8 - 1.2 * open_score

    targets = np.zeros((n_drones, 3), dtype=np.float32)
    for i in range(n_drones):
        theta = 2.0 * np.pi * (i / n_drones) + rot_speed * t
        targets[i, 0] = center[0] + radius * np.cos(theta)
        targets[i, 1] = center[1] + radius * np.sin(theta)
        targets[i, 2] = center[2] + z
    return targets


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
    n_drones = 6
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.state)
    sim.reset()

    open_score_ema = 0.5
    open_score_smooth = 0.2
    target_smooth = None
    target_smooth_alpha = 0.12
    pos_buffer = deque(maxlen=60)

    with HandLandmarker.create_from_options(options) as landmarker:
        k4a.start()
        print("Running hand->swarm control. Press q to quit.")
        start_t = time.time()
        frame_idx = 0
        try:
            while True:
                t = time.time() - start_t

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
                if result.hand_landmarks and result.hand_world_landmarks:
                    hand_lm_2d = result.hand_landmarks[0]
                    hand_lm_3d = result.hand_world_landmarks[0]
                    points_mm = world_landmarks_to_mm(hand_lm_3d)
                    open_raw, planarity, spread = compute_morph_alpha(points_mm)
                    open_score_ema = open_score_smooth * open_raw + (1.0 - open_score_smooth) * open_score_ema
                    draw_2d_hand(frame, hand_lm_2d, open_score_ema, planarity, spread)
                else:
                    cv2.putText(
                        frame,
                        "No hand detected, hold last command",
                        (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 180, 255),
                        2,
                    )

                raw_target = make_circle_targets(n_drones=n_drones, open_score=open_score_ema, t=t)
                if target_smooth is None:
                    target_smooth = raw_target.copy()
                else:
                    target_smooth = target_smooth_alpha * raw_target + (1.0 - target_smooth_alpha) * target_smooth

                cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
                cmd[..., :3] = target_smooth
                cmd[..., 6:10] = [0.0, 0.0, 0.0, 1.0]
                sim.state_control(cmd)
                sim.step(sim.freq // sim.control_freq)

                if frame_idx % 5 == 0:
                    pos_buffer.append(sim.data.states.pos[0].copy())

                if len(pos_buffer) > 1:
                    lines = np.array(pos_buffer)
                    for d in range(n_drones):
                        draw_line(
                            sim,
                            lines[:, d, :],
                            rgba=[0.0, 1.0, 0.0, 1.0],
                            start_size=0.4,
                            end_size=1.8,
                        )
                sim.render()

                cv2.putText(
                    frame,
                    f"radius={0.08 + 0.30 * open_score_ema:.2f} z={0.65 - 0.20 * open_score_ema:.2f}",
                    (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
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
