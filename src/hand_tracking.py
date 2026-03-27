import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# ===== MediaPipe initialize =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ===== connecting（21 points）=====
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20) # pinky
]
FINGERTIP_IDS = [4, 8, 12, 16, 20]
MCP_IDS = [5, 9, 13, 17]

# ===== draw keypoints =====
def draw_hand(frame, result):
    if result.hand_landmarks:
        h, w, _ = frame.shape

        for idx, hand_landmarks in enumerate(result.hand_landmarks):

            points = []

            # draw 21 keypoints
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # connecting
            for connection in HAND_CONNECTIONS:
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # label right/left hand
            if result.handedness:
                label = result.handedness[idx][0].category_name
                cv2.putText(frame, label, points[0],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

    return frame


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


def world_landmarks_to_mm(world_lms):
    pts = []
    for lm in world_lms:
        pts.append([lm.x * 1000.0, -lm.y * 1000.0, -lm.z * 1000.0])
    return np.asarray(pts, dtype=np.float32)


def compute_open_score(points_mm):
    centroid = points_mm.mean(axis=0)
    centered = points_mm - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-8))
    hand_scale = float(np.mean(np.linalg.norm(centered, axis=1)) + 1e-6)
    palm_center = points_mm[[0, *MCP_IDS]].mean(axis=0)
    tip_dist = np.linalg.norm(points_mm[FINGERTIP_IDS] - palm_center, axis=1)
    spread = float(np.mean(tip_dist) / hand_scale)

    spread_score = clamp01((spread - 1.00) / (1.75 - 1.00))
    planarity_score = clamp01((planarity - 0.10) / (0.60 - 0.10))
    open_score_plane = clamp01(0.55 * spread_score + 0.45 * planarity_score)

    # More robust fist cue: fingertips collapse toward palm/wrist when fist closes.
    wrist = points_mm[0]
    palm_center = points_mm[[0, *MCP_IDS]].mean(axis=0)
    tip_to_palm = np.linalg.norm(points_mm[FINGERTIP_IDS] - palm_center, axis=1).mean()
    tip_to_wrist = np.linalg.norm(points_mm[FINGERTIP_IDS] - wrist, axis=1).mean()
    fist_compact = clamp01(1.0 - ((0.6 * tip_to_palm + 0.4 * tip_to_wrist) / (hand_scale * 2.2 + 1e-6)))

    # Final open score: plane evidence minus fist compact evidence.
    open_score = clamp01(0.70 * open_score_plane + 0.30 * (1.0 - fist_compact))
    fist_score = clamp01(1.0 - open_score)
    return open_score, fist_score, centroid, eigvecs


def draw_blanket_surface(ax, radius, open_score):
    # open_score=1 -> flat blanket; open_score=0 -> sphere-like blanket
    sphere_alpha = 1.0 - open_score
    centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # Fixed canonical axes for readability (avoid rotating with hand pose).
    u_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    w_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    rho = np.linspace(0.0, 1.0, 24)
    th, rr = np.meshgrid(theta, rho)

    x_local = radius * rr * np.cos(th)
    y_local = radius * rr * np.sin(th)
    z_cap = radius * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0))
    z_top = sphere_alpha * z_cap
    z_bottom = -sphere_alpha * z_cap

    for z_local in (z_top, z_bottom):
        px = centroid[0] + x_local * u_vec[0] + y_local * v_vec[0] + z_local * w_vec[0]
        py = centroid[1] + x_local * u_vec[1] + y_local * v_vec[1] + z_local * w_vec[1]
        pz = centroid[2] + x_local * u_vec[2] + y_local * v_vec[2] + z_local * w_vec[2]
        ax.plot_surface(px, py, pz, color="tab:cyan", alpha=0.45, linewidth=0)
        ax.plot_wireframe(px, py, pz, color="tab:blue", linewidth=0.4, alpha=0.5)


def update_3d_visualization(ax_pts, ax_blanket, points_mm, open_score, fist_score):
    ax_pts.clear()
    ax_blanket.clear()

    ax_pts.set_title("3D Hand Keypoints")
    ax_blanket.set_title("Blanket Morph (Plane <-> Sphere)")
    for ax in (ax_pts, ax_blanket):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=22, azim=-68)

    centroid = points_mm.mean(axis=0)
    radius = max(60.0, float(np.mean(np.linalg.norm(points_mm - centroid, axis=1))))

    ax_pts.scatter(points_mm[:, 0], points_mm[:, 1], points_mm[:, 2], c="r", s=24)
    for a, b in HAND_CONNECTIONS:
        ax_pts.plot(
            [points_mm[a, 0], points_mm[b, 0]],
            [points_mm[a, 1], points_mm[b, 1]],
            [points_mm[a, 2], points_mm[b, 2]],
            "b-",
            linewidth=1.2,
        )

    draw_blanket_surface(ax_blanket, radius=radius, open_score=open_score)
    ax_blanket.text(
        -radius,
        -radius,
        radius * 0.95,
        f"open={open_score:.2f} fist={fist_score:.2f}",
        color="purple",
    )

    lim = max(120.0, radius * 1.4)
    for ax in (ax_pts, ax_blanket):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)


def draw_score_bars(frame, open_score, fist_score):
    h, w, _ = frame.shape
    x0 = 12
    y0 = h - 70
    bar_w = 180
    bar_h = 16
    cv2.putText(frame, "OPEN", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (120, 120, 120), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * open_score), y0 + bar_h), (80, 220, 80), -1)

    y1 = y0 + 28
    cv2.putText(frame, "FIST", (x0, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1)
    cv2.rectangle(frame, (x0, y1), (x0 + bar_w, y1 + bar_h), (120, 120, 120), 1)
    cv2.rectangle(frame, (x0, y1), (x0 + int(bar_w * fist_score), y1 + bar_h), (220, 120, 80), -1)


def main():
    model_path = "hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        plt.ion()
        fig = plt.figure("Hand 3D + Blanket")
        ax_pts = fig.add_subplot(121, projection="3d")
        ax_blanket = fig.add_subplot(122, projection="3d")

        # use the camera of labtop 
        cap = cv2.VideoCapture(0)
        frame_idx = 0
        open_ema = 0.5
        fist_ema = 0.5
        open_smooth = 0.2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(mp_image, int(frame_idx * (1000 / 30)))

            frame = draw_hand(frame, result)
            if result.hand_world_landmarks:
                points_mm = world_landmarks_to_mm(result.hand_world_landmarks[0])
                open_raw, fist_raw, _, _ = compute_open_score(points_mm)
                open_ema = open_smooth * open_raw + (1.0 - open_smooth) * open_ema
                fist_ema = open_smooth * fist_raw + (1.0 - open_smooth) * fist_ema
                update_3d_visualization(ax_pts, ax_blanket, points_mm, open_ema, fist_ema)
                cv2.putText(
                    frame,
                    f"open={open_ema:.2f} fist={fist_ema:.2f} (plane<->sphere)",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                draw_score_bars(frame, open_ema, fist_ema)
            plt.pause(0.001)

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1

        cap.release()
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()