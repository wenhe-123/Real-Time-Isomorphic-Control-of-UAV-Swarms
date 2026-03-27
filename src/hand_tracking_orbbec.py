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

# Nonlinear gain for morph sensitivity:
# open in [0,1] (0=sphere/fist, 1=plane/open). open**GAMMA makes fist side "more spherical".
OPEN_GAMMA = 1.8

# Keep morph axis scale fixed (only the morph surface changes size).
MORPH_AXIS_LIM_MM = 200.0

# Performance / real-time controls
PLOT_EVERY_N_FRAMES = 5  # update matplotlib every N frames
THETA_N = 28             # morph surface mesh resolution (lower = faster)
RHO_N = 14
ENABLE_3D_PLOT = True    # press 'p' to toggle at runtime

# Snap-to-canonical plane/sphere (helps reach exact endpoints)
# hysteresis threshold to avoid jittering
PLANE_SNAP_ON = 0.88
PLANE_SNAP_OFF = 0.82
SPHERE_SNAP_ON = 0.12
SPHERE_SNAP_OFF = 0.18

# 2D HUD anti-flicker
HUD_UPDATE_EVERY_N_FRAMES = 10
HUD_OPEN_STEP = 0.03
HUD_METRIC_STEP = 0.05

# SNAP display debounce (avoid visible blinking)
SNAP_SHOW_AFTER_FRAMES = 6
SNAP_HOLD_AFTER_RELEASE_FRAMES = 10


def draw_hud(frame, lines, origin=(16, 16), line_h=26, pad=8, alpha=0.55):
    """
    Draw readable HUD with a stable translucent background.
    """
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max([s[0] for s in sizes] + [1])
    h = line_h * len(lines)
    x0, y0 = x - pad, y - pad
    x1, y1 = x + w + pad, y + h + pad
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame.shape[1] - 1, x1)
    y1 = min(frame.shape[0] - 1, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, t in enumerate(lines):
        yy = y + i * line_h + 18
        cv2.putText(frame, t, (x, yy), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


# ===== draw keypoints =====
def draw_hand(frame, result, depth_map=None, print_depth=False):
    keypoints_3d = []
    if result.hand_landmarks:   #detect hand landmarks
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
                    depth_mm = int(depth_map[y, x])   #get depth value from depth map in pixel coordinates

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


def _blanket_param(radius, sphere_alpha, theta_n=THETA_N, rho_n=RHO_N):
    theta = np.linspace(0.0, 2.0 * np.pi, theta_n)
    rho = np.linspace(0.0, 1.0, rho_n)
    th, rr = np.meshgrid(theta, rho)
    x = radius * rr * np.cos(th)
    y = radius * rr * np.sin(th)
    z_cap = radius * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0))
    z = sphere_alpha * z_cap
    return x, y, z


def draw_blanket_morph_canonical(ax, radius, open_alpha, show_refs=True):
    """
    Canonical, easy-to-read morph surface (fixed axes).

    We render:
    - reference plane (open=1) as a gray wireframe disk
    - reference sphere (open=0) as a faint wireframe sphere (disk parameterization mirrored)
    - current morph surface as a solid + wireframe (cyan/blue)
    Plus a cross-section curve to make the interpolation obvious.
    """
    open_alpha = _clamp01(open_alpha)
    sphere_alpha = 1.0 - open_alpha

    # Make the plane state easier to read: enlarge radius when open.
    # This does not change the underlying control signal, only visualization.
    radius_vis = float(radius) * (0.90 + 0.95 * open_alpha)

    # references
    x_ref, y_ref, z_plane = _blanket_param(radius_vis, sphere_alpha=0.0)
    x_s, y_s, z_s = _blanket_param(radius_vis, sphere_alpha=1.0)

    if show_refs:
        # plane refs (both sides overlap at z=0)
        ax.plot_wireframe(x_ref, y_ref, z_plane, color="0.7", linewidth=0.35, alpha=0.55)
        # full sphere refs (top+bottom)
        ax.plot_wireframe(x_s, y_s, z_s, color="0.85", linewidth=0.25, alpha=0.35)
        ax.plot_wireframe(x_s, y_s, -z_s, color="0.85", linewidth=0.25, alpha=0.35)

    # current surface
    x, y, z = _blanket_param(radius_vis, sphere_alpha=sphere_alpha)
    ax.plot_surface(x, y, z, color="tab:cyan", alpha=0.35, linewidth=0)
    ax.plot_surface(x, y, -z, color="tab:cyan", alpha=0.35, linewidth=0)
    ax.plot_wireframe(x, y, z, color="tab:blue", linewidth=0.35, alpha=0.55)
    ax.plot_wireframe(x, y, -z, color="tab:blue", linewidth=0.35, alpha=0.55)

    # cross-section at theta=0 (y=0), show how z lifts as alpha changes
    rr = np.linspace(0.0, 1.0, 80)
    xs = radius_vis * rr
    ys = np.zeros_like(xs)
    zcap = radius_vis * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0))
    ax.plot(xs, ys, sphere_alpha * zcap, color="tab:purple", linewidth=2.0, alpha=0.9)
    ax.plot(xs, ys, -sphere_alpha * zcap, color="tab:purple", linewidth=2.0, alpha=0.9)

    # Concentric-ring samples: center + rings on plane; morph lifts each ring toward latitudes
    # (apple-peel: each ring becomes a latitude circle as the blanket wraps the sphere).
    R = float(radius_vis)
    num_rings = 4
    pts_per_ring = 6

    def build_ring_points():
        top_list = []
        bot_list = []
        ring_polys_top = []
        ring_polys_bot = []
        # North pole / disk center: maps to top of cap when sphere_alpha>0
        z0 = sphere_alpha * R
        top_list.append((0.0, 0.0, z0))
        # South reference: slight XY offset when near-plane so top/bottom don't collide at open=1;
        # offset -> 0 when fully spherical (sphere_alpha -> 1).
        off_scale = float((1.0 - sphere_alpha) ** 2)
        off_r = 0.06 * R * off_scale
        bot_list.append((off_r * np.cos(np.pi / 4.0), off_r * np.sin(np.pi / 4.0), -z0))
        # Complementary angles between hemispheres ("mosquito coil"): bottom ring is rotated
        # by half a slot + small per-ring twist so samples interleave instead of mirroring.
        half_slot = np.pi / float(pts_per_ring)
        twist_per_ring = np.pi / float(2 * num_rings * pts_per_ring)
        for ri in range(1, num_rings + 1):
            r = R * (ri / num_rings)
            z_ring = sphere_alpha * float(np.sqrt(max(0.0, R * R - r * r)))
            ring_top = []
            ring_bot = []
            ring_twist = ri * twist_per_ring
            for j in range(pts_per_ring):
                th_top = 2.0 * np.pi * j / pts_per_ring + ring_twist
                th_bot = th_top + half_slot + 0.5 * ring_twist
                px_t = r * np.cos(th_top)
                py_t = r * np.sin(th_top)
                px_b = r * np.cos(th_bot)
                py_b = r * np.sin(th_bot)
                ring_top.append((px_t, py_t, z_ring))
                ring_bot.append((px_b, py_b, -z_ring))
            top_list.extend(ring_top)
            bot_list.extend(ring_bot)
            arr_t = np.array(ring_top + [ring_top[0]], dtype=float)
            arr_b = np.array(ring_bot + [ring_bot[0]], dtype=float)
            ring_polys_top.append(arr_t)
            ring_polys_bot.append(arr_b)
        return (
            np.array(top_list, dtype=float),
            np.array(bot_list, dtype=float),
            ring_polys_top,
            ring_polys_bot,
        )

    pts_top, pts_bottom, ring_polys_top, ring_polys_bot = build_ring_points()

    # Draw ring polylines (peel / latitude circles)
    for arr in ring_polys_top:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="tab:green", linewidth=1.0, alpha=0.55)
    for arr in ring_polys_bot:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="darkgreen", linewidth=1.0, alpha=0.45)

    if sphere_alpha < 0.28:
        # Near plane: top/bottom surfaces collapse; separate points in BOTH XY and Z
        eps = max(3.5, 0.020 * radius_vis)
        top = pts_top.copy()
        bot = pts_bottom.copy()
        top[:, :2] *= 0.88
        bot[:, :2] *= 1.12
        ax.scatter(top[:, 0], top[:, 1], top[:, 2] + eps, c="k", s=22, alpha=0.65)
        ax.scatter(bot[:, 0], bot[:, 1], bot[:, 2] - eps, c="k", s=22, alpha=0.65)
    else:
        ax.scatter(pts_top[:, 0], pts_top[:, 1], pts_top[:, 2], c="k", s=22, alpha=0.65)
        ax.scatter(pts_bottom[:, 0], pts_bottom[:, 1], pts_bottom[:, 2], c="k", s=22, alpha=0.65)


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
    # morph_alpha=0 => sphere-like fist, morph_alpha=1 => plane-like open hand.
    spread_score = _clamp01((finger_spread - 1.00) / (1.65 - 1.00))
    planarity_score = _clamp01((planarity - 0.12) / (0.55 - 0.12))
    # isotropy_score: 0(plane-like) -> 1(sphere-like)
    isotropy_score = _clamp01((isotropy - 0.06) / (0.22 - 0.06))
    alpha = _clamp01(0.50 * spread_score + 0.35 * planarity_score + 0.15 * (1.0 - isotropy_score))
    # Boost curvature/control sensitivity near fist:
    # alpha is "open". When alpha<1, alpha**GAMMA becomes smaller -> more spherical.
    alpha = _clamp01(alpha ** OPEN_GAMMA)

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

    ax_topo.set_title("Blanket Morph: plane(open=1) ↔ sphere(open=0)")
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

        # Canonical morph with references + cross-section + sample points.
        draw_blanket_morph_canonical(ax_topo, radius=max(140.0, 2.2 * r), open_alpha=morph_alpha, show_refs=True)
        ax_topo.text(
            -max(140.0, 2.2 * r),
            -max(140.0, 2.2 * r),
            max(140.0, 2.2 * r) * 0.92,
            "Refs: gray=plane, light=full sphere\n"
            f"Current: blue/cyan  open={morph_alpha:.2f}\n"
            f"planarity={analysis['planarity']:.2f}  isotropy={analysis['isotropy']:.2f}",
            color="tab:purple",
        )

    ax_hand.view_init(elev=20, azim=-70)
    ax_topo.view_init(elev=22, azim=-58)
    ax_hand.set_box_aspect((1.0, 1.0, 1.0))
    ax_topo.set_box_aspect((1.0, 1.0, 1.0))

    # Fix axis scale for readability (no auto-zoom).
    lim = MORPH_AXIS_LIM_MM
    ax_topo.set_xlim(-lim, lim)
    ax_topo.set_ylim(-lim, lim)
    ax_topo.set_zlim(-lim, lim)
    return analyses


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
            open_free_ema = None
            alpha_smooth = 0.18
            snap_state = None  # None / "plane" / "sphere"
            hud_cache = {"open": None, "free": None, "plan": None, "iso": None, "spread": None, "text": None}
            snap_vis_state = None
            snap_stable_frames = 0
            snap_hold_frames = 0
            enable_3d = ENABLE_3D_PLOT
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

                t_ms = int(frame_idx * (1000 / 30))
                try:
                    result = landmarker.detect_for_video(mp_image, t_ms)
                except Exception as exc:
                    print(f"[WARN] mediapipe detect_for_video failed: {exc}")
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
                        if open_free_ema is None:
                            open_free_ema = float(tmp["morph_alpha"])
                        else:
                            open_free_ema = (
                                alpha_smooth * float(tmp["morph_alpha"]) + (1.0 - alpha_smooth) * open_free_ema
                            )

                        # Snap-to-canonical with hysteresis (easier to hit exact plane/sphere)
                        # IMPORTANT:
                        # Use the "free" continuous value to manage snap state,
                        # otherwise snap would never release (because we clamp to 0/1).
                        open_free = float(open_free_ema)
                        if snap_state == "plane":
                            if open_free < PLANE_SNAP_OFF:
                                snap_state = None
                        elif snap_state == "sphere":
                            if open_free > SPHERE_SNAP_OFF:
                                snap_state = None
                        else:
                            if open_free > PLANE_SNAP_ON:
                                snap_state = "plane"
                            elif open_free < SPHERE_SNAP_ON:
                                snap_state = "sphere"

                        open_out = open_free
                        if snap_state == "plane":
                            open_out = 1.0
                        elif snap_state == "sphere":
                            open_out = 0.0
                    else:
                        open_out = None
                else:
                    open_out = None

                # Debounce SNAP display to avoid flicker.
                if snap_state is None:
                    snap_stable_frames = 0
                    if snap_vis_state is not None:
                        snap_hold_frames += 1
                        if snap_hold_frames >= SNAP_HOLD_AFTER_RELEASE_FRAMES:
                            snap_vis_state = None
                            snap_hold_frames = 0
                else:
                    snap_hold_frames = 0
                    if snap_vis_state == snap_state:
                        snap_stable_frames = min(SNAP_SHOW_AFTER_FRAMES, snap_stable_frames + 1)
                    else:
                        snap_stable_frames += 1
                        if snap_stable_frames >= SNAP_SHOW_AFTER_FRAMES:
                            snap_vis_state = snap_state
                            snap_stable_frames = 0

                analyses = None
                if enable_3d and (frame_idx % PLOT_EVERY_N_FRAMES) == 0:
                    analyses = update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_alpha_smoothed=open_out,
                    )
                    plt.pause(0.0001)

                if analyses:
                    # show first hand summary on 2D window
                    a0 = analyses[0]
                    open_disp = open_out if open_out is not None else a0["morph_alpha"]
                    free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]

                    need_refresh = (frame_idx % HUD_UPDATE_EVERY_N_FRAMES) == 0 or hud_cache["open"] is None
                    if not need_refresh:
                        if abs(float(open_disp) - float(hud_cache["open"])) > HUD_OPEN_STEP:
                            need_refresh = True
                        if abs(float(free_disp) - float(hud_cache["free"])) > HUD_OPEN_STEP:
                            need_refresh = True
                        if abs(float(a0["planarity"]) - float(hud_cache["plan"])) > HUD_METRIC_STEP:
                            need_refresh = True
                        if abs(float(a0["isotropy"]) - float(hud_cache["iso"])) > HUD_METRIC_STEP:
                            need_refresh = True
                        if abs(float(a0["finger_spread"]) - float(hud_cache["spread"])) > HUD_METRIC_STEP:
                            need_refresh = True

                    if need_refresh:
                        hud_cache["open"] = float(open_disp)
                        hud_cache["free"] = float(free_disp)
                        hud_cache["plan"] = float(a0["planarity"])
                        hud_cache["iso"] = float(a0["isotropy"])
                        hud_cache["spread"] = float(a0["finger_spread"])
                        snap_txt = f"  SNAP:{snap_vis_state.upper()}" if snap_vis_state is not None else ""
                        hud_cache["text"] = [
                            f"Topo:{a0['topology']}{snap_txt}",
                            f"open:{open_disp:.2f}  free:{free_disp:.2f}",
                            f"spread:{a0['finger_spread']:.2f}  plan:{a0['planarity']:.2f}  iso:{a0['isotropy']:.2f}",
                        ]

                    if hud_cache["text"] is not None:
                        pass
                    if frame_idx % 30 == 0:
                        out_v = open_out if open_out is not None else a0["morph_alpha"]
                        free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                        print(
                            "topology="
                            f"{a0['topology']} "
                            f"open_out={out_v:.3f} free={free_v:.3f} "
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
                if key == ord("p"):
                    enable_3d = not enable_3d
                    print(f"3D plot enabled: {enable_3d}")

                if key == ord("q"):
                    break

                # Draw HUD EVERY frame using cached text (prevents flicker).
                if hud_cache["text"] is not None:
                    draw_hud(frame, hud_cache["text"], origin=(16, 16))
                frame_idx += 1
        finally:
            k4a.stop()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
