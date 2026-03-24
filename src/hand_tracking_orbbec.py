import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pyk4a import Config, FPS, PyK4A

# ===== MediaPipe initialize =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ===== connecting（21 points）=====
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

# fingertips in MediaPipe hand landmark index
FINGERTIP_IDS = [4, 8, 12, 16, 20]
WRIST_ID = 0
MCP_IDS = [5, 9, 13, 17]


# ===== draw keypoints =====
def draw_hand(frame, result, depth_map=None, print_depth=False):
    keypoints_3d = []
    if result.hand_landmarks:
        h, w, _ = frame.shape

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            points = []
            points_3d = []
            world_landmarks = None
            if hasattr(result, "hand_world_landmarks") and len(result.hand_world_landmarks) > idx:
                world_landmarks = result.hand_world_landmarks[idx]

            # draw 21 keypoints
            for kp_id, lm in enumerate(hand_landmarks):
                x = int(lm.x * w)
                y = int(lm.y * h)
                x = np.clip(x, 0, w - 1)
                y = np.clip(y, 0, h - 1)
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                depth_mm = None
                if depth_map is not None and y < depth_map.shape[0] and x < depth_map.shape[1]:
                    depth_mm = int(depth_map[y, x])

                # Use MediaPipe world landmarks for stable hand-shape 3D visualization.
                # Fallback to NaN if world landmarks are unavailable.
                if world_landmarks is not None and kp_id < len(world_landmarks):
                    wlm = world_landmarks[kp_id]
                    # Convert meters -> mm for easier reading.
                    x3d = float(wlm.x * 1000.0)
                    y3d = float(-wlm.y * 1000.0)
                    z3d = float(-wlm.z * 1000.0)
                    points_3d.append((x3d, y3d, z3d))
                else:
                    points_3d.append((np.nan, np.nan, np.nan))

                if depth_mm is not None and depth_mm > 0:
                    cv2.putText(
                        frame,
                        f"{depth_mm}",
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )
                    if print_depth:
                        print(
                            f"hand:{idx} kp:{kp_id:02d} x:{x:4d} y:{y:4d} depth_mm:{depth_mm:5d}"
                        )

            # connecting
            for connection in HAND_CONNECTIONS:
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # label right/left hand
            if result.handedness:
                label = result.handedness[idx][0].category_name
                cv2.putText(
                    frame,
                    label,
                    points[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            keypoints_3d.append(points_3d)

    return frame, keypoints_3d


def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def _orthonormal_basis_from_dir(direction):
    d = _safe_normalize(direction)
    if np.linalg.norm(d) < 1e-8:
        d = np.array([0.0, 0.0, 1.0], dtype=float)
    ref = np.array([1.0, 0.0, 0.0], dtype=float) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    u = _safe_normalize(np.cross(d, ref))
    v = _safe_normalize(np.cross(d, u))
    return d, u, v


def draw_bone_cylinder(ax, p0, p1, radius, color="tab:cyan", alpha=0.35):
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return
    d, u, v = _orthonormal_basis_from_dir(vec)
    t = np.linspace(0.0, length, 8)
    ang = np.linspace(0.0, 2.0 * np.pi, 16)
    tt, aa = np.meshgrid(t, ang)
    center = p0[None, None, :] + tt[..., None] * d[None, None, :]
    ring = np.cos(aa)[..., None] * u[None, None, :] + np.sin(aa)[..., None] * v[None, None, :]
    surf = center + radius * ring
    ax.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=color, alpha=alpha, linewidth=0)


def draw_joint_sphere(ax, c, radius, color="tab:orange", alpha=0.45):
    u = np.linspace(0.0, 2.0 * np.pi, 16)
    v = np.linspace(0.0, np.pi, 12)
    uu, vv = np.meshgrid(u, v)
    xs = c[0] + radius * np.cos(uu) * np.sin(vv)
    ys = c[1] + radius * np.sin(uu) * np.sin(vv)
    zs = c[2] + radius * np.cos(vv)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


def draw_morph_surface(ax, centroid, eigvecs, radius, alpha):
    """
    Draw continuous isomorphic morph surface:
    alpha=0 -> plane-like disk
    alpha=1 -> sphere
    """
    alpha = _clamp01(alpha)
    u_vec = _safe_normalize(eigvecs[:, 0])
    v_vec = _safe_normalize(eigvecs[:, 1])
    w_vec = _safe_normalize(eigvecs[:, 2])

    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    rho = np.linspace(0.0, 1.0, 24)
    th, rr = np.meshgrid(theta, rho)

    # radial profile: from flat disk to hemisphere-like bulge, then full sphere by mirroring.
    # thickness increases smoothly with alpha.
    x_local = radius * rr * np.cos(th)
    y_local = radius * rr * np.sin(th)
    z_cap = radius * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0))
    z_local_top = alpha * z_cap
    z_local_bottom = -alpha * z_cap

    px_t = centroid[0] + x_local * u_vec[0] + y_local * v_vec[0] + z_local_top * w_vec[0]
    py_t = centroid[1] + x_local * u_vec[1] + y_local * v_vec[1] + z_local_top * w_vec[1]
    pz_t = centroid[2] + x_local * u_vec[2] + y_local * v_vec[2] + z_local_top * w_vec[2]
    ax.plot_surface(px_t, py_t, pz_t, color="tab:cyan", alpha=0.40, linewidth=0)

    # For alpha close to 0, top and bottom collapse to same plane naturally.
    px_b = centroid[0] + x_local * u_vec[0] + y_local * v_vec[0] + z_local_bottom * w_vec[0]
    py_b = centroid[1] + x_local * u_vec[1] + y_local * v_vec[1] + z_local_bottom * w_vec[1]
    pz_b = centroid[2] + x_local * u_vec[2] + y_local * v_vec[2] + z_local_bottom * w_vec[2]
    ax.plot_surface(px_b, py_b, pz_b, color="tab:cyan", alpha=0.40, linewidth=0)


def analyze_hand_topology(hand_points):
    all_pts = np.array(hand_points, dtype=float)
    valid_all = ~np.isnan(all_pts[:, 2])
    if np.sum(valid_all) < 8:
        return None

    # Remove wrist from topology fitting to avoid fist "pointed tail" artifact.
    fit_ids = [i for i in range(len(hand_points)) if i != WRIST_ID]
    fit_pts = np.array(
        [hand_points[i] for i in fit_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if fit_pts.shape[0] < 7:
        fit_pts = all_pts[valid_all]

    centroid = fit_pts.mean(axis=0)
    centered = fit_pts - centroid

    # PCA for normal and shape anisotropy
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]  # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    normal = _safe_normalize(eigvecs[:, 2])  # smallest-variance direction
    lamb_sum = float(np.sum(eigvals)) + 1e-8
    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-8))
    isotropy = float(eigvals[2] / lamb_sum)  # larger -> more sphere-like

    # Finger spread factor: fingertip distances to palm center, normalized by hand size.
    # palm center: wrist + MCP joints mean
    palm_ids = [WRIST_ID] + MCP_IDS
    palm_pts = np.array(
        [hand_points[i] for i in palm_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    tip_pts = np.array(
        [hand_points[i] for i in FINGERTIP_IDS if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if palm_pts.shape[0] > 0:
        palm_center = palm_pts.mean(axis=0)
    else:
        palm_center = centroid

    hand_scale = float(np.mean(np.linalg.norm(centered, axis=1))) + 1e-6
    if tip_pts.shape[0] > 0:
        tip_dist = np.linalg.norm(tip_pts - palm_center, axis=1)
        finger_spread = float(np.mean(tip_dist) / hand_scale)
    else:
        finger_spread = 0.0

    # Continuous morph score for fist->open interpolation (isomorphic simulation)
    # alpha=0 => sphere-like fist, alpha=1 => plane-like open hand.
    spread_score = _clamp01((finger_spread - 1.00) / (1.65 - 1.00))
    planarity_score = _clamp01((planarity - 0.12) / (0.55 - 0.12))
    alpha = _clamp01(0.60 * spread_score + 0.40 * planarity_score)

    # Keep a coarse label for debugging/overlay only.
    if alpha > 0.67:
        topology = "plane"
    elif alpha < 0.33:
        topology = "sphere"
    else:
        topology = "intermediate"

    radius = float(np.mean(np.linalg.norm(fit_pts - centroid, axis=1)))
    return {
        "centroid": centroid,
        "normal": normal,
        "eigvecs": eigvecs,
        "planarity": planarity,
        "isotropy": isotropy,
        "finger_spread": finger_spread,
        "morph_alpha": alpha,
        "topology": topology,
        "radius": radius,
        "points": fit_pts,
    }


def update_3d_plot(ax_hand, ax_topo, hands_3d, morph_alpha_smoothed=None):
    ax_hand.clear()
    ax_topo.clear()

    ax_hand.set_title("Hand Keypoints 3D (MediaPipe World)")
    ax_hand.set_xlabel("X (mm)")
    ax_hand.set_ylabel("Y (mm)")
    ax_hand.set_zlabel("Z (mm)")

    ax_topo.set_title("Plane-to-Sphere Morph Output")
    ax_topo.set_xlabel("X (mm)")
    ax_topo.set_ylabel("Y (mm)")
    ax_topo.set_zlabel("Z (mm)")

    analyses = []
    for hand_points in hands_3d:
        arr = np.array(hand_points, dtype=float)
        if arr.size == 0:
            continue
        valid = ~np.isnan(arr[:, 2])
        if not np.any(valid):
            continue

        pts = arr[valid]
        ax_hand.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="r", s=20)

        # Draw finger connections only when both endpoints are valid.
        for a, b in HAND_CONNECTIONS:
            if (
                a < len(arr)
                and b < len(arr)
                and not np.isnan(arr[a, 2])
                and not np.isnan(arr[b, 2])
            ):
                ax_hand.plot(
                    [arr[a, 0], arr[b, 0]],
                    [arr[a, 1], arr[b, 1]],
                    [arr[a, 2], arr[b, 2]],
                    "b-",
                    linewidth=1.2,
                )

        analysis = analyze_hand_topology(hand_points)
        if analysis is None:
            continue
        analyses.append(analysis)

        c = analysis["centroid"]
        n = analysis["normal"]
        r = max(analysis["radius"], 1.0)
        topo = analysis["topology"]
        morph_alpha = analysis["morph_alpha"]
        if morph_alpha_smoothed is not None:
            morph_alpha = morph_alpha_smoothed

        # Shared points in surface panel
        ax_topo.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="gray", s=10, alpha=0.4)
        ax_topo.scatter([c[0]], [c[1]], [c[2]], c="k", s=40)

        # Draw normal vector from centroid
        ax_topo.quiver(c[0], c[1], c[2], n[0], n[1], n[2], length=1.2 * r, color="m")

        draw_morph_surface(
            ax_topo,
            centroid=c,
            eigvecs=analysis["eigvecs"],
            radius=r,
            alpha=morph_alpha,
        )

        ax_topo.text(
            c[0],
            c[1],
            c[2],
            f"{topo} a={morph_alpha:.2f}",
            color="tab:purple",
        )

    ax_hand.view_init(elev=20, azim=-70)
    ax_topo.view_init(elev=20, azim=-70)
    ax_hand.set_box_aspect((1.0, 1.0, 1.0))
    ax_topo.set_box_aspect((1.0, 1.0, 1.0))
    return analyses


def main():
    model_path = "hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )

    # same camera opening style as test_orbbec.py
    k4a = PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30,
        )
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        plt.ion()
        fig = plt.figure("Hand 3D and Topology")
        ax_hand = fig.add_subplot(121, projection="3d")
        ax_topo = fig.add_subplot(122, projection="3d")

        k4a.start()
        print("Running Orbbec hand tracking... press q to quit")

        try:
            frame_idx = 0
            morph_alpha_ema = None
            alpha_smooth = 0.18
            while True:
                try:
                    capture = k4a.get_capture()
                except Exception as exc:
                    print(f"[WARN] get_capture failed: {exc}")
                    continue
                if capture.color is None:
                    continue

                color = capture.color

                # pyk4a color is usually BGRA
                if color.ndim == 3 and color.shape[2] == 4:
                    frame = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                else:
                    frame = color

                # NOTE:
                # Some Orbbec K4A wrapper builds crash on transformed_depth conversion
                # (descriptor mismatch in transformation_depth_image_to_color_camera).
                # Use raw depth directly and resize to color size for robust runtime behavior.
                depth_map = capture.depth
                if depth_map is not None and (
                    depth_map.shape[0] != frame.shape[0] or depth_map.shape[1] != frame.shape[1]
                ):
                    depth_map = cv2.resize(
                        depth_map,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = landmarker.detect(mp_image)
                except Exception as exc:
                    print(f"[WARN] mediapipe detect failed: {exc}")
                    continue
                frame, hands_3d = draw_hand(
                    frame,
                    result,
                    depth_map=depth_map,
                    print_depth=(frame_idx % 30 == 0),
                )
                # smooth morph alpha for real-time control stability
                if hands_3d:
                    tmp = analyze_hand_topology(hands_3d[0])
                    if tmp is not None:
                        if morph_alpha_ema is None:
                            morph_alpha_ema = tmp["morph_alpha"]
                        else:
                            morph_alpha_ema = (
                                alpha_smooth * tmp["morph_alpha"] + (1.0 - alpha_smooth) * morph_alpha_ema
                            )

                analyses = update_3d_plot(
                    ax_hand,
                    ax_topo,
                    hands_3d,
                    morph_alpha_smoothed=morph_alpha_ema,
                )
                plt.pause(0.001)

                if analyses:
                    # show first hand summary on 2D window
                    a0 = analyses[0]
                    cv2.putText(
                        frame,
                        f"Topo:{a0['topology']} A:{(morph_alpha_ema if morph_alpha_ema is not None else a0['morph_alpha']):.2f} "
                        f"Spread:{a0['finger_spread']:.2f} Planarity:{a0['planarity']:.2f}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    if frame_idx % 30 == 0:
                        print(
                            "topology="
                            f"{a0['topology']} "
                            f"alpha={(morph_alpha_ema if morph_alpha_ema is not None else a0['morph_alpha']):.3f} "
                            f"spread={a0['finger_spread']:.3f} "
                            f"planarity={a0['planarity']:.3f} "
                            f"isotropy={a0['isotropy']:.3f}"
                        )
                cv2.imshow("Hand Tracking Orbbec", frame)

                # Press s to save current 3D plot image.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    out_name = f"hand_3d_frame_{frame_idx:06d}.png"
                    fig.savefig(out_name, dpi=150, bbox_inches="tight")
                    print(f"Saved 3D plot: {out_name}")

                if key == ord("q"):
                    break
                frame_idx += 1
        finally:
            k4a.stop()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
