import argparse

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pyk4a import Config, FPS, PyK4A
from pyk4a.calibration import CalibrationType

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
# Index/middle/ring/pinky tips only — thumb often stays extended in a fist and inflates “open” spread.
FINGERTIP_IDS_FOUR = [8, 12, 16, 20]
WRIST_ID = 0
MCP_IDS = [5, 9, 13, 17]

# Nonlinear gain for morph sensitivity:
# open in [0,1] (0=sphere/fist, 1=plane/open). open**GAMMA makes fist side "more spherical".
OPEN_GAMMA = 1.8
# Labels / SNAP: same banding as the original MP-world script (plane > 0.67, sphere < 0.33 on instant alpha).
TOPO_ALPHA_PLANE = 0.67
TOPO_ALPHA_SPHERE = 0.33

# Reject depth reads that hit background / wrong layer (linear RGB→depth map often mis-anchors palm edge).
DEPTH_ABS_MAX_MM = 1800.0
# Same-hand metric depth rarely differs by >~235 mm from wrist at desk distance (pinky mis-map ~230 mm).
DEPTH_MAX_DELTA_FROM_WRIST_MM = 235.0
# Second pass vs robust palm depth: only wrist + index + middle MCP (5,9). Ring/pinky MCP (13,17)
# often mis-hit on linear RGB→depth and must NOT define the reference median.
DEPTH_REF_ANCHOR_IDS = (WRIST_ID, 5, 9)
# Second pass: reject joints far from that reference (open hand + bad tips still caught).
DEPTH_MEDIAN_MAX_DELTA_MM = 175.0

# Empirical raw morph_alpha range with shape_norm (fist ~0.22, open ~0.72): map to [0,1] for blanket + HUD.
OPEN_REMAP_LO = 0.22
OPEN_REMAP_HI = 0.72

# 3D skeleton + topology: MP-only keeps a coherent hand; fused mixes depth frame + MP world → often skewed.
HAND_3D_SOURCE_MP = "mp"
HAND_3D_SOURCE_FUSED = "fused"
# Coordinate frame for 3D output (after metric base_mm from MP or fused).
HAND_FRAME_SCALED = "scaled"  # wrist origin + isotropic palm scale (legacy)
HAND_FRAME_PALM_PLANE = "palm_plane"  # wrist origin; XY = plane(wrist, index MCP, middle MCP); then / palm scale
HAND_FRAME_METRIC_MM = "metric_mm"  # raw mm in camera/world frame from base_mm

# Keep morph axis scale fixed (only the morph surface changes size).
MORPH_AXIS_LIM_MM = 200.0
# Wrist-centered + palm-scale normalization: 3D plot limits (unit hand size).
NORM_AXIS_HALFLIM = 1.35

# Performance / real-time controls
PLOT_EVERY_N_FRAMES = 5  # update matplotlib every N frames
THETA_N = 28             # morph surface mesh resolution (lower = faster)
RHO_N = 14
ENABLE_3D_PLOT = True    # press 'p' to toggle at runtime

# Snap-to-canonical plane/sphere (EMA raw_free); hysteresis matches original standalone script.
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

# --- Depth camera metric 3D + fusion with MediaPipe ---
# Depth unproject → 3D in **depth camera** frame (mm). MediaPipe world → wrist-centric pseudo-metric (mm).
# IMPORTANT: Never do wrist_cam_depth + (p_mp - wrist_mp): those vectors live in different frames and
# will scramble the skeleton. Per-joint blend only: w * p_depth + (1-w) * p_mediapipe (see _fuse_cam_and_mp).
DEPTH_FUSION_WEIGHT = 0.55  # 1 = depth unproject only; 0 = MediaPipe world only
POINT_EMA_ALPHA = 0.28  # temporal smoothing on fused 3D (per keypoint, mm space)
# Median depth in a (2r+1)^2 patch on the depth image reduces single-pixel noise (critical for fusion≈1).
DEPTH_MEDIAN_PATCH_RADIUS = 2

# Femto Bolt is factory-calibrated; Orbbec SDK (and this K4A wrapper) can expose vision + IMU calibration
# programmatically. After PyK4A.open/start, `k4a.calibration` is the K4A-style blob used by
# `convert_2d_to_3d` for depth/color intrinsics and depth↔color extrinsics. IMU is separate from RGB-D
# unprojection in this script. If unprojection looks wrong, some wrappers still ship placeholder blobs —
# compare with OrbbecViewer / native SDK export or Orbbec docs for your firmware.


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


def _mp_world_to_mm(wlm):
    return (
        float(wlm.x * 1000.0),
        float(-wlm.y * 1000.0),
        float(-wlm.z * 1000.0),
    )


def _map_color_pixel_to_depth_pixel(x_c: int, y_c: int, w_c: int, h_c: int, w_d: int, h_d: int):
    """Linear map MediaPipe/OpenCV color (x,y) → depth image indices (approx. alignment)."""
    xd = int(np.clip(round(x_c * (w_d / max(w_c, 1))), 0, w_d - 1))
    yd = int(np.clip(round(y_c * (h_d / max(h_c, 1))), 0, h_d - 1))
    return xd, yd


def _unproject_depth_pixel_to_depth_camera_mm(calibration, x_d: float, y_d: float, depth_mm: float):
    """Depth-image pixel + depth (mm) → 3D in **depth camera** frame (mm). Safe for Orbbec (no depth→color warp)."""
    if calibration is None or depth_mm is None or depth_mm <= 0:
        return None
    try:
        p = calibration.convert_2d_to_3d(
            (float(x_d), float(y_d)),
            float(depth_mm),
            CalibrationType.DEPTH,
            CalibrationType.DEPTH,
        )
        return (float(p[0]), float(p[1]), float(p[2]))
    except (ValueError, Exception):
        return None


def _unproject_color_aligned_to_depth_camera_mm(calibration, x_c: float, y_c: float, depth_mm: float):
    """Color pixel + color-aligned depth (mm) → 3D in **depth camera** frame (mm). Needs valid transformed_depth."""
    if calibration is None or depth_mm is None or depth_mm <= 0:
        return None
    try:
        p = calibration.convert_2d_to_3d(
            (float(x_c), float(y_c)),
            float(depth_mm),
            CalibrationType.COLOR,
            CalibrationType.DEPTH,
        )
        return (float(p[0]), float(p[1]), float(p[2]))
    except (ValueError, Exception):
        return None


def _metric_hand_to_shape_normalized(points):
    """
    Wrist at origin; divide by a palm scale so overall size does not grow with camera distance.
    Z is depth relative to wrist (same as X,Y), in units of hand size — matches webcam_modes spirit.
    """
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] < 21:
        return [tuple(float(x) for x in row) for row in arr]
    wrist = arr[WRIST_ID]
    if not np.all(np.isfinite(wrist)):
        return [tuple(float(x) for x in row) for row in arr]
    rel = arr - wrist
    mcp_norms = []
    for i in MCP_IDS:
        if i < len(rel) and np.isfinite(rel[i]).all():
            mcp_norms.append(float(np.linalg.norm(rel[i])))
    if mcp_norms:
        s = float(np.mean(mcp_norms))
    elif np.isfinite(rel[9]).all():
        s = float(np.linalg.norm(rel[9]))
    else:
        ft = []
        for i in FINGERTIP_IDS:
            if i < len(rel) and np.isfinite(rel[i]).all():
                ft.append(float(np.linalg.norm(rel[i])))
        s = float(np.mean(ft)) if ft else 1.0
    s = max(s, 1e-3)
    out = rel / s
    return [tuple(float(x) for x in out[i]) for i in range(21)]


def _palm_plane_basis_from_world(points):
    """
    Wrist at origin; XY plane contains wrist (0), index MCP (5), middle MCP (9).
    Returns (wrist, R) with R[:,0]=e_x, R[:,1]=e_y, R[:,2]=e_z in world frame.
    Local coords: p_local = R.T @ (p_world - wrist). e_x along wrist→middle MCP; e_z ⊥ palm (right-handed).
    """
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] < 21:
        return None
    w = arr[WRIST_ID]
    pi = arr[5]
    pm = arr[9]
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(pi)) or not np.all(np.isfinite(pm)):
        return None
    u = pm - w
    v = pi - w
    n = np.cross(u, v)
    ln = float(np.linalg.norm(n))
    if ln < 1e-6:
        return None
    e_z = n / ln
    lu = float(np.linalg.norm(u))
    if lu < 1e-6:
        return None
    e_x = u / lu
    e_y = np.cross(e_z, e_x)
    ly = float(np.linalg.norm(e_y))
    if ly < 1e-6:
        return None
    e_y = e_y / ly
    e_z = np.cross(e_x, e_y)
    lz = float(np.linalg.norm(e_z))
    if lz < 1e-6:
        return None
    e_z = e_z / lz
    R = np.stack([e_x, e_y, e_z], axis=1)
    return w, R


def _metric_hand_to_palm_plane_normalized(points):
    """
    Wrist at origin; XY = plane through wrist + index MCP + middle MCP; divide by mean MCP distance (unit hand).
    """
    basis = _palm_plane_basis_from_world(points)
    if basis is None:
        return _metric_hand_to_shape_normalized(points)
    w, R = basis
    arr = np.asarray(points, dtype=float)
    n = arr.shape[0]
    rel = np.zeros((n, 3), dtype=float)
    for k in range(n):
        p = arr[k]
        if not np.all(np.isfinite(p)):
            rel[k] = np.nan
        else:
            rel[k] = R.T @ (p - w)
    mcp_norms = []
    for mid in MCP_IDS:
        if mid < n and np.isfinite(arr[mid]).all() and np.isfinite(w).all():
            mcp_norms.append(float(np.linalg.norm(arr[mid] - w)))
    if mcp_norms:
        s = float(np.mean(mcp_norms))
    elif n > 9 and np.isfinite(arr[9]).all():
        s = float(np.linalg.norm(arr[9] - w))
    else:
        s = 1.0
    s = max(s, 1e-3)
    out = rel / s
    return [tuple(float(x) for x in out[i]) for i in range(n)]


def _palm_plane_curl_metrics(points_21):
    """
    Fingertip contraction in palm-plane normalized coords: r_xy = sqrt(x^2+y^2), z = normal to palm.
    Lower mean r_xy (four fingers) ⇒ more curled toward palm in-plane; |z| = fold toward/away from palm plane.
    """
    arr = np.asarray(points_21, dtype=float)
    if arr.shape[0] < 21:
        return None
    four = []
    zs = []
    for i in FINGERTIP_IDS_FOUR:
        if i < len(arr) and np.isfinite(arr[i]).all():
            x, y, z = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])
            four.append(float(np.hypot(x, y)))
            zs.append(abs(z))
    thumb_r = None
    thumb_z = None
    if 4 < len(arr) and np.isfinite(arr[4]).all():
        x, y, z = float(arr[4, 0]), float(arr[4, 1]), float(arr[4, 2])
        thumb_r = float(np.hypot(x, y))
        thumb_z = abs(z)
    return {
        "mean_r_xy_four": float(np.mean(four)) if four else None,
        "mean_abs_z_four": float(np.mean(zs)) if zs else None,
        "thumb_r_xy": thumb_r,
        "thumb_abs_z": thumb_z,
    }


def load_depth_unproject_rigid_npy(path: str | None) -> np.ndarray | None:
    """Load optional 4×4 (float64) rigid transform in mm (homogeneous) for depth-unprojected points."""
    if not path:
        return None
    try:
        T = np.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"--depth-unproject-rigid-npy: file not found: {path!r}. "
            "Use a real 4×4 float64 .npy from calibration, or omit this flag (documentation used /path/to/ as placeholder)."
        ) from e
    if getattr(T, "shape", None) != (4, 4):
        raise ValueError(f"--depth-unproject-rigid-npy must be 4×4, got {getattr(T, 'shape', None)}")
    return np.asarray(T, dtype=np.float64)


def _transform_point_rigid_4x4_mm(p_xyz: tuple | None, T: np.ndarray | None) -> tuple | None:
    """Apply T @ [x,y,z,1] in mm. Used to move depth-camera points toward the frame used for blending."""
    if T is None or p_xyz is None:
        return p_xyz
    if np.any(np.isnan(np.asarray(p_xyz, dtype=float))):
        return p_xyz
    v = np.array(
        [float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2]), 1.0],
        dtype=np.float64,
    )
    o = T @ v
    return (float(o[0]), float(o[1]), float(o[2]))


def _fuse_cam_and_mp(p_cam, p_mp, fusion_weight: float):
    """
    Per joint only: w * p_depth + (1-w) * p_mediapipe. Same index i for both; no cross-frame wrist+delta.
    """
    w = float(np.clip(fusion_weight, 0.0, 1.0))
    p_mp = np.array(p_mp, dtype=float)
    if p_cam is None or np.any(np.isnan(p_cam)):
        return (float(p_mp[0]), float(p_mp[1]), float(p_mp[2]))
    p_cam = np.array(p_cam, dtype=float)
    out = w * p_cam + (1.0 - w) * p_mp
    return (float(out[0]), float(out[1]), float(out[2]))


def _ema_point_triplet(prev, cur, alpha: float):
    if prev is None:
        return cur
    a = np.array(prev, dtype=float)
    b = np.array(cur, dtype=float)
    if np.any(np.isnan(b)):
        return tuple(a.tolist())
    return tuple((float(v) for v in ((1.0 - alpha) * a + alpha * b)))


def _median_valid_depth_mm(depth_img: np.ndarray, u: int, v: int, patch_r: int):
    """
    Median of valid (>0) depths in [v±patch_r, u±patch_r] on depth image (indices u,v in that image).
    Reduces speckle; patch_r=0 falls back to single pixel.
    """
    if depth_img is None or patch_r < 0:
        return None
    h, w = int(depth_img.shape[0]), int(depth_img.shape[1])
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    if patch_r == 0:
        d = int(depth_img[v, u])
        return d if d > 0 else None
    u0, u1 = max(0, u - patch_r), min(w, u + patch_r + 1)
    v0, v1 = max(0, v - patch_r), min(h, v + patch_r + 1)
    patch = depth_img[v0:v1, u0:u1].astype(np.float64).ravel()
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return int(np.median(patch))


def _read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, patch_r: int = DEPTH_MEDIAN_PATCH_RADIUS):
    """Depth in mm for HUD + unproject. Prefer color-aligned; else raw depth via linear map + median patch."""
    if depth_aligned is not None and depth_aligned.shape[0] == h and depth_aligned.shape[1] == w:
        return _median_valid_depth_mm(depth_aligned, x, y, patch_r)
    if depth_raw is not None:
        xd, yd = _map_color_pixel_to_depth_pixel(x, y, w, h, depth_raw.shape[1], depth_raw.shape[0])
        return _median_valid_depth_mm(depth_raw, xd, yd, patch_r)
    return None


def _unproject_to_depth_cam_mm(
    calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw
):
    if calibration is None or depth_mm is None or depth_mm <= 0:
        return None
    if depth_aligned is not None and depth_aligned.shape[0] == h and depth_aligned.shape[1] == w:
        return _unproject_color_aligned_to_depth_camera_mm(calibration, float(x), float(y), float(depth_mm))
    if depth_raw is not None:
        xd, yd = _map_color_pixel_to_depth_pixel(x, y, w, h, depth_raw.shape[1], depth_raw.shape[0])
        return _unproject_depth_pixel_to_depth_camera_mm(calibration, float(xd), float(yd), float(depth_mm))
    return None


# ===== draw keypoints =====
def draw_hand(
    frame,
    result,
    depth_raw=None,
    depth_aligned=None,
    print_depth=False,
    *,
    calibration=None,
    fusion_weight: float = DEPTH_FUSION_WEIGHT,
    ema_alpha: float = POINT_EMA_ALPHA,
    ema_points=None,
    depth_patch_radius: int = DEPTH_MEDIAN_PATCH_RADIUS,
    hand_frame: str = HAND_FRAME_SCALED,
    filter_depth_outliers: bool = True,
    depth_max_delta_mm: float = DEPTH_MAX_DELTA_FROM_WRIST_MM,
    depth_median_max_delta_mm: float | None = DEPTH_MEDIAN_MAX_DELTA_MM,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
    depth_unproject_rigid_T: np.ndarray | None = None,
    skip_wrist_labels: bool = False,
):
    """
    depth_raw: native depth image (H_d,W_d), uint16 mm — used on Orbbec/Femto (no transformed_depth).
    depth_aligned: optional depth warped to color (same H,W as frame); only if SDK supports it.
    calibration: pyk4a Calibration for metric unproject.

    Output 3D is fused (depth camera mm + MediaPipe) then EMA-smoothed per keypoint.
    depth_patch_radius: median depth over (2r+1)^2 in depth image (reduces noise when fusion≈1).
    hand_frame: scaled | palm_plane | metric_mm — see --hand-frame.
    filter_depth_outliers: drop per-joint depths far from wrist / absurd mm → fusion uses MP for those joints.
    hand_3d_source: "mp" = 3D from MediaPipe world only (recommended); "fused" = depth+MP per joint (fragile).
    depth_unproject_rigid_T: optional 4×4 (mm) applied to each depth-unprojected point before fusion (see --depth-unproject-rigid-npy).
    skip_wrist_labels: if True, do not draw Left/Right at the wrist (use overlay_wrist_labels after computing role strings).
    """
    keypoints_3d = []
    all_ema_out = []

    if not result.hand_landmarks:
        return frame, keypoints_3d, ema_points

    h, w, _ = frame.shape

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        points = []
        points_3d = []
        world_landmarks = None
        if hasattr(result, "hand_world_landmarks") and len(result.hand_world_landmarks) > idx:
            world_landmarks = result.hand_world_landmarks[idx]

        mp_mm = []
        if world_landmarks is not None:
            for kp_id in range(21):
                if kp_id < len(world_landmarks):
                    mp_mm.append(_mp_world_to_mm(world_landmarks[kp_id]))
                else:
                    mp_mm.append((np.nan, np.nan, np.nan))
        else:
            mp_mm = [(np.nan, np.nan, np.nan)] * 21

        depth_vals = []
        for kp_id, lm in enumerate(hand_landmarks):
            x = int(np.clip(int(lm.x * w), 0, w - 1))
            y = int(np.clip(int(lm.y * h), 0, h - 1))
            points.append((x, y))
            depth_vals.append(
                _read_depth_mm_at_landmark(x, y, h, w, depth_aligned, depth_raw, depth_patch_radius)
            )

        if filter_depth_outliers:
            depth_vals = _reject_depth_outliers(
                depth_vals,
                max_delta_mm=depth_max_delta_mm,
                median_max_delta_mm=depth_median_max_delta_mm,
            )

        hand_ema_in = ema_points[idx] if ema_points is not None and idx < len(ema_points) else None

        fused_raw = []
        for kp_id in range(21):
            x, y = points[kp_id]
            depth_mm = depth_vals[kp_id]

            p_cam = None
            if depth_mm is not None and calibration is not None:
                p_cam = _unproject_to_depth_cam_mm(
                    calibration, x, y, depth_mm, h, w, depth_aligned, depth_raw
                )
                p_cam = _transform_point_rigid_4x4_mm(p_cam, depth_unproject_rigid_T)

            p_mp = mp_mm[kp_id]
            fused = _fuse_cam_and_mp(p_cam, p_mp, fusion_weight)
            fused_raw.append(fused)

        if hand_3d_source == HAND_3D_SOURCE_FUSED:
            base_mm = fused_raw
        else:
            if world_landmarks is not None:
                base_mm = [tuple(mp_mm[i]) for i in range(21)]
                if not np.all(np.isfinite(np.array(base_mm, dtype=float))):
                    base_mm = fused_raw
            else:
                base_mm = fused_raw

        if hand_frame == HAND_FRAME_PALM_PLANE:
            viz_pts = _metric_hand_to_palm_plane_normalized(base_mm)
        elif hand_frame == HAND_FRAME_SCALED:
            viz_pts = _metric_hand_to_shape_normalized(base_mm)
        else:
            viz_pts = list(base_mm)

        norm_depth_label = hand_frame in (HAND_FRAME_SCALED, HAND_FRAME_PALM_PLANE)

        hand_ema_out = []
        for kp_id in range(21):
            prev_k = None
            if hand_ema_in is not None and kp_id < len(hand_ema_in):
                prev_k = hand_ema_in[kp_id]
            sm = _ema_point_triplet(prev_k, viz_pts[kp_id], ema_alpha)
            hand_ema_out.append(sm)
            points_3d.append(sm)

        all_ema_out.append(hand_ema_out)

        for kp_id, lm in enumerate(hand_landmarks):
            x, y = points[kp_id]
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            depth_mm = depth_vals[kp_id]
            if depth_mm is not None and depth_mm > 0:
                dw = depth_vals[0]
                if norm_depth_label and dw is not None and dw > 0:
                    label = f"{depth_mm - dw:+d}"
                else:
                    label = f"{depth_mm}"
                cv2.putText(
                    frame,
                    label,
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

        for connection in HAND_CONNECTIONS:
            p1 = points[connection[0]]
            p2 = points[connection[1]]
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        if result.handedness and not skip_wrist_labels:
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

    return frame, keypoints_3d, all_ema_out


def overlay_wrist_labels(frame, result, labels_by_idx: dict, *, font_scale: float = 0.72):
    """
    Draw short strings near wrist (landmark 0). ``labels_by_idx`` maps hand index
    (same order as ``result.hand_landmarks``) to text, e.g. ``M2`` / ``open 0.73``.
    """
    if not result.hand_landmarks or not labels_by_idx:
        return frame
    h, w, _ = frame.shape
    for idx, hand_lms in enumerate(result.hand_landmarks):
        if idx not in labels_by_idx:
            continue
        lm0 = hand_lms[0]
        px = int(np.clip(int(lm0.x * w), 0, w - 1))
        py = int(np.clip(int(lm0.y * h), 0, h - 1))
        cv2.putText(
            frame,
            labels_by_idx[idx],
            (px, max(24, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def _topology_label_from_alpha(alpha: float) -> str:
    """Coarse label from continuous morph_alpha (use EMA for HUD so it matches raw_free / SNAP)."""
    a = float(alpha)
    if a > TOPO_ALPHA_PLANE:
        return "plane"
    if a < TOPO_ALPHA_SPHERE:
        return "sphere"
    return "intermediate"


def _remap_open_display(alpha: float, lo: float, hi: float) -> float:
    """Linear stretch of raw morph_alpha to [0, 1] for visualization (empirical fist/open range)."""
    return _clamp01((float(alpha) - lo) / max(float(hi - lo), 1e-6))


def _reject_depth_outliers(
    depth_vals,
    *,
    max_delta_mm: float = DEPTH_MAX_DELTA_FROM_WRIST_MM,
    median_max_delta_mm: float | None = DEPTH_MEDIAN_MAX_DELTA_MM,
):
    """
    Invalidate per-joint depths that are implausible vs wrist and vs palm-row median.
    Fusion then falls back to MediaPipe for those joints only.
    """
    if not depth_vals or len(depth_vals) < 21:
        return depth_vals
    dw = depth_vals[0]
    if dw is None or dw <= 0:
        return depth_vals
    dw = float(dw)
    out = list(depth_vals)
    for i in range(len(out)):
        d = out[i]
        if d is None or d <= 0:
            continue
        df = float(d)
        if df > DEPTH_ABS_MAX_MM or abs(df - dw) > max_delta_mm:
            out[i] = None

    if median_max_delta_mm is None or median_max_delta_mm <= 0:
        return out

    ref_vals = []
    for i in DEPTH_REF_ANCHOR_IDS:
        if i < len(out) and out[i] is not None and out[i] > 0:
            ref_vals.append(float(out[i]))
    if len(ref_vals) < 2:
        return out
    mref = float(np.median(ref_vals))
    for i in range(len(out)):
        if i == WRIST_ID:
            continue
        d = out[i]
        if d is None or d <= 0:
            continue
        if abs(float(d) - mref) > float(median_max_delta_mm):
            out[i] = None
    return out


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
        lw_ref = 0.65 if sphere_alpha < 0.12 else 0.35
        ax.plot_wireframe(x_ref, y_ref, z_plane, color="0.7", linewidth=lw_ref, alpha=0.6)
        # full sphere refs (top+bottom)
        ax.plot_wireframe(x_s, y_s, z_s, color="0.85", linewidth=0.25, alpha=0.35)
        ax.plot_wireframe(x_s, y_s, -z_s, color="0.85", linewidth=0.25, alpha=0.35)

    # current surface
    x, y, z = _blanket_param(radius_vis, sphere_alpha=sphere_alpha)
    # plot_surface on a perfectly flat z=0 mesh often omits faces; use a tiny ±z lift so the disk fills visibly.
    near_flat = sphere_alpha < 0.08
    if near_flat:
        eps = max(0.6, 0.006 * radius_vis)
        z_up = np.full_like(z, eps, dtype=float)
        z_dn = np.full_like(z, -eps, dtype=float)
        ax.plot_surface(
            x,
            y,
            z_up,
            color="tab:cyan",
            alpha=0.5,
            linewidth=0.22,
            edgecolor="tab:blue",
            antialiased=True,
            shade=True,
        )
        ax.plot_surface(
            x,
            y,
            z_dn,
            color="tab:cyan",
            alpha=0.4,
            linewidth=0.18,
            edgecolor="tab:cyan",
            antialiased=True,
            shade=True,
        )
    else:
        ax.plot_surface(x, y, z, color="tab:cyan", alpha=0.35, linewidth=0)
        ax.plot_surface(x, y, -z, color="tab:cyan", alpha=0.35, linewidth=0)
        ax.plot_wireframe(x, y, z, color="tab:blue", linewidth=0.35, alpha=0.55)
        ax.plot_wireframe(x, y, -z, color="tab:blue", linewidth=0.35, alpha=0.55)

    # cross-section at theta=0 (y=0), show how z lifts as alpha changes
    rr = np.linspace(0.0, 1.0, 80)
    xs = radius_vis * rr
    ys = np.zeros_like(xs)
    zcap = radius_vis * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0))
    if sphere_alpha < 1e-5:
        # Full plane: meridian collapses to z=0; draw the disk radius in the xy plane.
        ax.plot(xs, ys, np.zeros_like(xs), color="tab:purple", linewidth=2.8, alpha=0.95)
        ax.scatter([radius_vis], [0.0], [0.0], c="tab:purple", s=36, zorder=5)
    else:
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
    span_ratio = float(eigvals[0] / (eigvals[2] + 1e-8))

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

    # Continuous morph score: same blend as the original MediaPipe-world standalone script (all hand frames).
    # morph_alpha=0 => sphere-like fist, morph_alpha=1 => plane-like open hand. OPEN_GAMMA biases fist→more sphere.
    spread_score = _clamp01((finger_spread - 1.00) / (1.65 - 1.00))
    planarity_score = _clamp01((planarity - 0.12) / (0.55 - 0.12))
    isotropy_score = _clamp01((isotropy - 0.06) / (0.22 - 0.06))
    alpha = _clamp01(0.50 * spread_score + 0.35 * planarity_score + 0.15 * (1.0 - isotropy_score))
    alpha = _clamp01(alpha ** OPEN_GAMMA)

    # Instant-frame label (HUD should use _topology_label_from_alpha on EMA for consistency).
    topology = _topology_label_from_alpha(alpha)

    radius = float(np.mean(np.linalg.norm(fit_pts - centroid, axis=1)))
    return {
        "centroid": centroid,
        "normal": normal,
        "eigvecs": eigvecs,
        "planarity": planarity,
        "isotropy": isotropy,
        "span_ratio": span_ratio,
        "finger_spread": finger_spread,
        "morph_alpha": alpha,
        "topology": topology,
        "radius": radius,
        "points": fit_pts,
    }


def update_3d_plot(
    ax_hand,
    ax_topo,
    hands_3d,
    morph_alpha_smoothed=None,
    *,
    shape_normalized: bool = False,
    hand_frame: str = HAND_FRAME_SCALED,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
):
    ax_hand.clear()
    ax_topo.clear()

    _src = "MediaPipe" if hand_3d_source == HAND_3D_SOURCE_MP else "depth+MP fused"
    ax_hand.set_title(f"Hand 3D ({_src}) — joints 0..20, HAND_CONNECTIONS = MediaPipe order")
    if shape_normalized:
        if hand_frame == HAND_FRAME_PALM_PLANE:
            ax_hand.set_xlabel("X (norm, along mid MCP)")
            ax_hand.set_ylabel("Y (norm, in palm)")
            ax_hand.set_zlabel("Z (norm, ⊥ palm)")
        else:
            ax_hand.set_xlabel("X (norm)")
            ax_hand.set_ylabel("Y (norm)")
            ax_hand.set_zlabel("Z (norm, rel. depth)")
    else:
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

        # Tight, centered limits (hand plot previously had no limits → misleading 3D view).
        sub = arr[valid]
        ctr_hand = sub.mean(axis=0)
        span = float(np.max(np.ptp(sub, axis=0)))
        if shape_normalized:
            half = min(float(NORM_AXIS_HALFLIM), max(0.35, 0.55 * span + 0.18))
        else:
            half = min(float(MORPH_AXIS_LIM_MM), max(120.0, 0.55 * span + 90.0))
        ax_hand.set_xlim(ctr_hand[0] - half, ctr_hand[0] + half)
        ax_hand.set_ylim(ctr_hand[1] - half, ctr_hand[1] + half)
        ax_hand.set_zlim(ctr_hand[2] - half, ctr_hand[2] + half)

        analysis = analyze_hand_topology(hand_points)
        if analysis is None:
            # Earlier versions always drew the morph; sanity/PCA reject used to leave ax_topo empty after clear().
            morph_fb = 0.55
            if morph_alpha_smoothed is not None:
                morph_fb = float(morph_alpha_smoothed)
            morph_fb = _clamp01(morph_fb)
            r_draw = 200.0
            draw_blanket_morph_canonical(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
            ax_topo.text(
                -r_draw,
                -r_draw,
                r_draw * 0.92,
                "Refs: gray=plane, light=full sphere\n"
                f"Current (fallback): open={morph_fb:.2f}\n"
                "Topology unavailable this frame (see console); left = hand pose.",
                color="tab:orange",
            )
            continue
        analyses.append(analysis)

        c = analysis["centroid"]
        n = analysis["normal"]
        r = max(analysis["radius"], 1.0)
        topo = analysis["topology"]
        morph_alpha = analysis["morph_alpha"]
        if morph_alpha_smoothed is not None:
            morph_alpha = morph_alpha_smoothed
        morph_alpha = _clamp01(float(morph_alpha))

        draw_blanket_morph_canonical(ax_topo, radius=max(140.0, 2.2 * r), open_alpha=morph_alpha, show_refs=True)
        ax_topo.text(
            -max(140.0, 2.2 * r),
            -max(140.0, 2.2 * r),
            max(140.0, 2.2 * r) * 0.92,
            "Refs: gray=plane, light=full sphere\n"
            f"Current: blue/cyan  open={morph_alpha:.2f}\n"
            f"planarity={analysis['planarity']:.2f}  isotropy={analysis['isotropy']:.2f}"
            + (
                f"  λ0/λ2={analysis['span_ratio']:.1f}"
                if shape_normalized
                else ""
            ),
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
    ap = argparse.ArgumentParser(
        description="Orbbec/K4A hand tracking with depth-fused 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Femto Bolt (e.g. F00364-152): for metric RGB–depth alignment, obtain from Orbbec SDK / Viewer:\n"
            "  • Color intrinsics (fx,fy,cx,cy) + distortion; depth intrinsics; depth scale (mm).\n"
            "  • Extrinsic T_depth_to_color (or color_to_depth).\n"
            "This script uses pyk4a’s K4A-style calibration blob; if the wrapper reports placeholder calib,\n"
            "use --depth-fusion 0 for stable MediaPipe 3D until calibration is verified.\n"
            "When --depth-fusion is near 1, try --depth-patch 3 and lower --ema-alpha (e.g. 0.15) "
            "if the skeleton is jittery.\n"
            "Default: --hand-frame scaled (wrist-centered + palm-scale). "
            "Use --hand-frame palm_plane for wrist origin + palm XY plane + fingertip curl metrics.\n"
            "Default --hand-3d mp: 3D skeleton from MediaPipe (stable); use --hand-3d fused for depth blend.\n"
        ),
    )
    ap.add_argument("--model", type=str, default="hand_landmarker.task", help="hand_landmarker.task path")
    ap.add_argument(
        "--depth-fusion",
        type=float,
        default=DEPTH_FUSION_WEIGHT,
        help="0=MediaPipe world only, 1=depth unproject only (per-joint linear blend in between)",
    )
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=POINT_EMA_ALPHA,
        help="EMA smoothing 0..1 on fused keypoints (higher = faster tracking)",
    )
    ap.add_argument(
        "--hand-frame",
        choices=(HAND_FRAME_SCALED, HAND_FRAME_PALM_PLANE, HAND_FRAME_METRIC_MM),
        default=HAND_FRAME_SCALED,
        help=(
            "3D skeleton + topology coords: scaled=legacy wrist+palm-scale; "
            "palm_plane=wrist origin, XY=plane(wrist,index MCP,middle MCP), /palm-scale (fingertip curl in HUD); "
            "metric_mm=no normalization (raw mm)."
        ),
    )
    ap.add_argument(
        "--no-shape-normalize",
        action="store_true",
        help="Same as --hand-frame metric_mm (ignored if --hand-frame is set to palm_plane or metric_mm).",
    )
    ap.add_argument(
        "--no-depth-outlier-filter",
        action="store_true",
        help="Keep all per-joint depth samples (default: drop joints with depth far from wrist / >1.8m).",
    )
    ap.add_argument(
        "--no-open-remap",
        action="store_true",
        help="Use raw morph_alpha for blanket/HUD (default with shape_norm: map ~0.22–0.72 → 0–1).",
    )
    ap.add_argument(
        "--open-remap-lo",
        type=float,
        default=None,
        metavar="X",
        help=f"Lower bound for open linear remap (default {OPEN_REMAP_LO}).",
    )
    ap.add_argument(
        "--open-remap-hi",
        type=float,
        default=None,
        metavar="X",
        help=f"Upper bound for open linear remap (default {OPEN_REMAP_HI}).",
    )
    ap.add_argument(
        "--depth-patch",
        type=int,
        default=DEPTH_MEDIAN_PATCH_RADIUS,
        metavar="R",
        help=(
            "Median depth over (2R+1)^2 pixels on depth image (0=single pixel). "
            "Larger R reduces speckle when --depth-fusion is high; try 2–4."
        ),
    )
    ap.add_argument(
        "--depth-max-delta-mm",
        type=float,
        default=None,
        metavar="D",
        help=f"Max |depth−wrist| per joint in mm (default {DEPTH_MAX_DELTA_FROM_WRIST_MM}).",
    )
    ap.add_argument(
        "--depth-median-max-delta-mm",
        type=float,
        default=None,
        metavar="D",
        help=(
            "Second pass: max |depth−median(wrist+MCPs)|; 0 disables (default "
            f"{DEPTH_MEDIAN_MAX_DELTA_MM})."
        ),
    )
    ap.add_argument(
        "--use-transformed-depth",
        action="store_true",
        help=(
            "Use SDK depth→color alignment (transformed_depth). Real Azure Kinect only; "
            "Orbbec Femto / K4A-wrapper often crashes on this path — leave off (default)."
        ),
    )
    ap.add_argument(
        "--hand-3d",
        choices=(HAND_3D_SOURCE_MP, HAND_3D_SOURCE_FUSED),
        default=HAND_3D_SOURCE_MP,
        help=(
            "3D plot + topology: mp=MediaPipe world (stable hand shape); "
            "fused=depth+MP per joint (only if RGB–D alignment is good)."
        ),
    )
    ap.add_argument(
        "--depth-unproject-rigid-npy",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional 4×4 float64 .npy (mm, homogeneous): T@[x,y,z,1] applied to each depth-unprojected point "
            "before fusion. Use offline calibration (depth camera → frame comparable to MediaPipe). "
            "Ignored for pure 3D when --hand-3d mp and depth-fusion=0."
        ),
    )
    args = ap.parse_args()
    model_path = args.model
    fusion_w = float(np.clip(args.depth_fusion, 0.0, 1.0))
    ema_a = float(np.clip(args.ema_alpha, 0.0, 1.0))
    depth_patch_r = int(np.clip(args.depth_patch, 0, 15))
    if args.hand_frame != HAND_FRAME_SCALED:
        hand_frame = args.hand_frame
    elif args.no_shape_normalize:
        hand_frame = HAND_FRAME_METRIC_MM
    else:
        hand_frame = HAND_FRAME_SCALED
    shape_norm = hand_frame in (HAND_FRAME_SCALED, HAND_FRAME_PALM_PLANE)
    depth_outlier_filter = not args.no_depth_outlier_filter
    if shape_norm and not args.no_open_remap:
        lo = OPEN_REMAP_LO if args.open_remap_lo is None else args.open_remap_lo
        hi = OPEN_REMAP_HI if args.open_remap_hi is None else args.open_remap_hi
        open_remap = (lo, hi) if hi > lo + 1e-6 else None
    else:
        open_remap = None

    depth_max_delta_mm = (
        DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    )
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)

    depth_rigid_T = load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy)

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
        calib = k4a.calibration
        print(
            "Depth-fused 3D. "
            f"fusion={fusion_w:.2f} ema={ema_a:.2f} depth_patch={depth_patch_r}  "
            f"hand_frame={hand_frame}  shape_norm={shape_norm}  hand_3d={args.hand_3d}  "
            f"depth_outlier_filter={depth_outlier_filter}  "
            f"dΔ={depth_max_delta_mm:.0f}mm medΔ={depth_median_max_delta_mm}  "
            f"open_remap={open_remap}  aligned_depth={args.use_transformed_depth}  "
            f"depth_rigid_T={'on' if depth_rigid_T is not None else 'off'}  "
            "q=quit  p=3D  s=save"
        )

        try:
            frame_idx = 0
            warned_fusion_linear_map = False
            ema_3d = None
            open_free_ema = None
            alpha_smooth = 0.18
            snap_state = None  # None / "plane" / "sphere"
            hud_cache = {
                "open": None,
                "free": None,
                "plan": None,
                "iso": None,
                "spread": None,
                "curl": None,
                "text": None,
            }
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

                # Default: raw depth + color→depth linear map + DEPTH unproject (Orbbec-safe).
                # Optional: transformed_depth (can abort on Orbbec K4A-wrapper — use real Kinect only).
                depth_raw = capture.depth
                depth_aligned = None
                if args.use_transformed_depth:
                    try:
                        td = capture.transformed_depth
                        if (
                            td is not None
                            and td.size > 0
                            and td.shape[0] == frame.shape[0]
                            and td.shape[1] == frame.shape[1]
                        ):
                            depth_aligned = td
                        elif td is not None:
                            print(
                                f"[WARN] transformed_depth shape {td.shape} != color {frame.shape}; "
                                "ignoring aligned depth"
                            )
                    except Exception as exc:
                        print(f"[WARN] transformed_depth failed: {exc}")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                t_ms = int(frame_idx * (1000 / 30))
                try:
                    result = landmarker.detect_for_video(mp_image, t_ms)
                except Exception as exc:
                    print(f"[WARN] mediapipe detect_for_video failed: {exc}")
                    continue
                if (
                    fusion_w >= 0.99
                    and not args.use_transformed_depth
                    and args.hand_3d == HAND_3D_SOURCE_FUSED
                    and not warned_fusion_linear_map
                ):
                    print(
                        "[INFO] depth-fusion≈1 + --hand-3d fused: linear RGB→depth map; 3D can be wrong "
                        "without good calibration. Try --hand-3d mp for stable skeleton, or --depth-patch 3–4."
                    )
                    warned_fusion_linear_map = True
                frame, hands_3d, ema_3d = draw_hand(
                    frame,
                    result,
                    depth_raw=depth_raw,
                    depth_aligned=depth_aligned,
                    print_depth=(frame_idx % 30 == 0),
                    calibration=calib,
                    fusion_weight=fusion_w,
                    ema_alpha=ema_a,
                    ema_points=ema_3d,
                    depth_patch_radius=depth_patch_r,
                    hand_frame=hand_frame,
                    filter_depth_outliers=depth_outlier_filter,
                    depth_max_delta_mm=depth_max_delta_mm,
                    depth_median_max_delta_mm=depth_median_max_delta_mm,
                    hand_3d_source=args.hand_3d,
                    depth_unproject_rigid_T=depth_rigid_T,
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
                        shape_normalized=shape_norm,
                        hand_frame=hand_frame,
                        hand_3d_source=args.hand_3d,
                    )
                    plt.pause(0.0001)

                if analyses:
                    # show first hand summary on 2D window
                    a0 = analyses[0]
                    topo_lbl = _topology_label_from_alpha(
                        float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"])
                    )
                    open_disp = open_out if open_out is not None else a0["morph_alpha"]
                    free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                    if open_remap is not None:
                        lo_r, hi_r = open_remap
                        open_disp = _remap_open_display(open_disp, lo_r, hi_r)
                        free_disp = _remap_open_display(free_disp, lo_r, hi_r)

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
                        if hand_frame == HAND_FRAME_PALM_PLANE and hands_3d:
                            cm = _palm_plane_curl_metrics(hands_3d[0])
                            curl_s = None
                            if cm and cm.get("mean_r_xy_four") is not None:
                                tr = cm.get("thumb_r_xy")
                                thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
                                curl_s = (
                                    f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}"
                                    f"{thumb_s}"
                                )
                            if curl_s != hud_cache.get("curl"):
                                need_refresh = True

                    if need_refresh:
                        hud_cache["open"] = float(open_disp)
                        hud_cache["free"] = float(free_disp)
                        hud_cache["plan"] = float(a0["planarity"])
                        hud_cache["iso"] = float(a0["isotropy"])
                        hud_cache["spread"] = float(a0["finger_spread"])
                        snap_txt = f"  SNAP:{snap_vis_state.upper()}" if snap_vis_state is not None else ""
                        lines = [
                            f"Topo:{topo_lbl}{snap_txt}",
                            f"open:{open_disp:.2f}  free:{free_disp:.2f}",
                            f"spread:{a0['finger_spread']:.2f}  plan:{a0['planarity']:.2f}  iso:{a0['isotropy']:.2f}",
                        ]
                        if hand_frame == HAND_FRAME_PALM_PLANE and hands_3d:
                            cm = _palm_plane_curl_metrics(hands_3d[0])
                            if cm and cm.get("mean_r_xy_four") is not None:
                                tr = cm.get("thumb_r_xy")
                                thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
                                curl_s = (
                                    f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}"
                                    f"{thumb_s}"
                                )
                                hud_cache["curl"] = curl_s
                                lines.append(curl_s)
                            else:
                                hud_cache["curl"] = None
                        else:
                            hud_cache["curl"] = None
                        hud_cache["text"] = lines

                    if hud_cache["text"] is not None:
                        pass
                    if frame_idx % 30 == 0:
                        out_v = open_out if open_out is not None else a0["morph_alpha"]
                        free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                        topo_print = _topology_label_from_alpha(float(free_v))
                        if open_remap is not None:
                            lo_r, hi_r = open_remap
                            out_show = _remap_open_display(out_v, lo_r, hi_r)
                            free_show = _remap_open_display(free_v, lo_r, hi_r)
                            open_part = f"open={out_show:.3f} raw={out_v:.3f} free={free_show:.3f} raw_free={free_v:.3f}"
                        else:
                            open_part = f"open_out={out_v:.3f} free={free_v:.3f}"
                        curl_part = ""
                        if hand_frame == HAND_FRAME_PALM_PLANE and hands_3d:
                            cm = _palm_plane_curl_metrics(hands_3d[0])
                            if cm and cm.get("mean_r_xy_four") is not None:
                                tr = cm.get("thumb_r_xy")
                                thumb_p = f" thumb_r={tr:.3f}" if tr is not None else ""
                                curl_part = (
                                    f" curl_rxy4={cm['mean_r_xy_four']:.3f} curl_z4={cm['mean_abs_z_four']:.3f}"
                                    f"{thumb_p}"
                                )
                        print(
                            "topology="
                            f"{topo_print} inst={a0['morph_alpha']:.3f} "
                            f"{open_part} "
                            f"spread={a0['finger_spread']:.3f} "
                            f"planarity={a0['planarity']:.3f} "
                            f"isotropy={a0['isotropy']:.3f}"
                            f"{curl_part}"
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
