"""
Laptop webcam hand tracking with three morph modes (test on webcam first).

- **Left hand**: only selects **mode** 1/2/3 (index/middle/ring “how many up”; see
  classify_mode_from_fingers). Does not drive open/close.
- **Right hand**: only drives **open ↔ closed** shape (same topology / morph_alpha as
  hand_tracking_orbbec). Left never affects planarity, spread, or open.
- Both hands drawn; thicker skeleton on Left = mode, on Right = open/morph.

Modes:
  1 — Plane ↔ sphere (same blanket morph as hand_tracking_orbbec.py)
  2 — Plane ↔ square pyramid: four **colored** lateral faces (like box walls); scatter on sides only
  3 — Plane ↔ box-fold: small open base; taller walls; folded R boosted; corner+face-center dots

Controls: q quit, p toggle 3D, s save matplotlib figure.

If the left hand is missing, the last mode is kept. If the right hand is missing,
open/morph freezes at the last value until the right hand returns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# MediaPipe exposes RunningMode (not VisionRunningMode) on tasks.vision
RunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]
# Index / middle / ring — used for 1–2–3 mode gesture
MODE_COUNT_TIP_IDS = [8, 12, 16]
WRIST_ID = 0
MCP_IDS = [5, 9, 13, 17]

OPEN_GAMMA = 1.8
MORPH_AXIS_LIM_MM = 200.0
# Wrist-centered normalized plots (match hand_tracking_orbbec.py)
NORM_AXIS_HALFLIM = 1.35
HAND_3D_SOURCE_MP = "mp"
HAND_3D_SOURCE_FUSED = "fused"
HAND_FRAME_SCALED = "scaled"
HAND_FRAME_PALM_PLANE = "palm_plane"
HAND_FRAME_METRIC_MM = "metric_mm"

PLOT_EVERY_N_FRAMES = 5
THETA_N = 28
RHO_N = 14
ENABLE_3D_PLOT = True

PLANE_SNAP_ON = 0.88
PLANE_SNAP_OFF = 0.82
SPHERE_SNAP_ON = 0.12
SPHERE_SNAP_OFF = 0.18

HUD_UPDATE_EVERY_N_FRAMES = 10
HUD_OPEN_STEP = 0.03
HUD_METRIC_STEP = 0.05

SNAP_SHOW_AFTER_FRAMES = 6
SNAP_HOLD_AFTER_RELEASE_FRAMES = 10

# Mode classification: normalized tip distances vs. hand scale (index/middle/ring)
MODE_EXTEND_MIN = 0.62  # below this max(dn): fist / no clear gesture → mode 1
# Tips within this gap of max(dn) count as "same tier extended" (fixes slightly bent finger)
MODE_TIER_GAP = 0.26
# Require raw mode this many consecutive frames before accepting (reduces flicker)
MODE_DEBOUNCE_FRAMES = 7

# 3D morph sample markers (match across mode1 blanket / mode2 pyramid / mode3 box)
MORPH_SAMPLE_SCATTER_S = 22
MORPH_SAMPLE_ALPHA = 0.65

# Shared horizontal scale for all modes (open hand → large footprint; half-width R of [-R,R]²)
MORPH_PLANE_RADIUS_A = 0.90
MORPH_PLANE_RADIUS_B = 0.95

# Plane-state scatter: same n×n grid on [-R,R]² in all modes (spacing ∝ 2R/(n-1))
MORPH_PLANE_GRID_N = 5
# Blanket: below this sphere_alpha, treat as “flat” and use the shared plane grid for samples
MORPH_PLANE_GRID_SPHERE_ALPHA = 0.22

# Mode3 box: ``k_net`` only for ``R_geom`` area formula; visual wall height uses ``MORPH_BOX_WALL_HEIGHT_RATIO``
MORPH_BOX_NET_K = 0.95
MORPH_BOX_WALL_HEIGHT_RATIO = 1.86
# Open/plane: base half-width = R_ref * this
MORPH_BOX_OPEN_BASE_SCALE = 0.45
# Fully folded: half-width *= this vs ``R_geom`` (slightly larger closed box)
MORPH_BOX_FOLD_SIZE_SCALE = 1.20


def _morph_box_half_width_for_matching_net_area(
    R_ref: float, k: float = MORPH_BOX_NET_K
) -> float:
    """
    Half-edge R of the box such that the **unfolded** net (one square base + four rectangles 2R×kR)
    has the same area as the canonical square of half-width ``R_ref``: ``(2·R_ref)²``.

    Area net = ``4·R² + 8·k·R² = 4·R²·(1+2k)``  ⇒  ``R = R_ref / sqrt(1+2k)`` — base corners
    sit closer to the origin than ``R_ref``, while total in-plane coverage matches modes 1/2.
    """
    denom = 1.0 + 2.0 * float(k)
    return float(R_ref) / float(np.sqrt(denom))


def _morph_plane_extent_radius(radius: float, open_alpha: float) -> float:
    """Horizontal extent R for morph geometry and plane samples (mm)."""
    return float(radius) * (MORPH_PLANE_RADIUS_A + MORPH_PLANE_RADIUS_B * _clamp01(open_alpha))


def _morph_plane_square_half_width_to_disk_radius(R_half: float) -> float:
    """Disk radius whose area equals the square [-R_half, R_half]² (mode1 blanket vs mode2/3 square)."""
    return 2.0 * float(R_half) / float(np.sqrt(np.pi))


def _morph_plane_grid_points_z0(R: float, n: int):
    """n×n points on [-R,R]² at z=0; shape (n*n, 3)."""
    if n < 2:
        n = 2
    xs = np.linspace(-R, R, n)
    ys = np.linspace(-R, R, n)
    pts = []
    for x in xs:
        for y in ys:
            pts.append((float(x), float(y), 0.0))
    return np.array(pts, dtype=float)


def _allocate_integer_by_weights(n_total: int, weights: Sequence[float]) -> List[int]:
    """Largest-remainder split of n_total across positive weights."""
    w = np.array([max(0.0, float(x)) for x in weights], dtype=float)
    s = float(np.sum(w))
    if n_total <= 0 or s <= 0.0:
        return [0] * len(weights)
    raw = n_total * (w / s)
    floors = np.floor(raw).astype(int)
    rem = int(n_total) - int(np.sum(floors))
    frac = raw - floors
    order = np.argsort(-frac)
    for k in range(rem):
        floors[order[k]] += 1
    return [int(x) for x in floors]


def _triangle_bary_samples(n_pts: int, A: np.ndarray, B: np.ndarray, P: np.ndarray) -> List[Tuple[float, float, float]]:
    """~n_pts deterministic samples inside triangle A,B,P (3-vectors)."""
    if n_pts <= 0:
        return []
    A = np.asarray(A, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    P = np.asarray(P, dtype=float).reshape(3)
    m = max(3, int(np.ceil(np.sqrt(float(n_pts)))))
    out: List[Tuple[float, float, float]] = []
    for u in np.linspace(0.0, 1.0, m):
        vmax = 1.0 - u
        if vmax < 1e-12 and u < 1e-12:
            continue
        vs = np.linspace(0.0, vmax, max(1, int(np.ceil(m * max(vmax, 0.2)))))
        for v in vs:
            w = 1.0 - u - v
            if w < -1e-9:
                continue
            p = w * P + u * A + v * B
            out.append((float(p[0]), float(p[1]), float(p[2])))
            if len(out) >= n_pts:
                return out[:n_pts]
    return out[:n_pts]


def _unique_xyz_rows(arr: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Drop duplicate 3D points (e.g. shared base/wall corners) after rounding."""
    arr = np.asarray(arr, dtype=float).reshape(-1, 3)
    if arr.size == 0:
        return arr.reshape(0, 3)
    r = np.round(arr, decimals=decimals)
    _, idx = np.unique(r, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _boxfold_anchor_points(R: float, H: float, theta: float) -> np.ndarray:
    """
    **Base** (z=0): four corners + face center. **Each wall**: four corners + face center only
    (no edge midpoints). Shared vertices deduped.
    """
    R = float(R)
    H = float(H)
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    rows: List[Tuple[float, float, float]] = []

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            rows.append((sx * R, sy * R, 0.0))
    rows.append((0.0, 0.0, 0.0))

    rows.extend(
        [
            (-R, R, 0.0),
            (R, R, 0.0),
            (R, R + H * ct, H * st),
            (-R, R + H * ct, H * st),
            (0.0, R + 0.5 * H * ct, 0.5 * H * st),
        ]
    )
    rows.extend(
        [
            (-R, -R, 0.0),
            (R, -R, 0.0),
            (R, -R - H * ct, H * st),
            (-R, -R - H * ct, H * st),
            (0.0, -R - 0.5 * H * ct, 0.5 * H * st),
        ]
    )
    rows.extend(
        [
            (R, -R, 0.0),
            (R, R, 0.0),
            (R + H * ct, R, H * st),
            (R + H * ct, -R, H * st),
            (R + 0.5 * H * ct, 0.0, 0.5 * H * st),
        ]
    )
    rows.extend(
        [
            (-R, -R, 0.0),
            (-R, R, 0.0),
            (-R - H * ct, R, H * st),
            (-R - H * ct, -R, H * st),
            (-R - 0.5 * H * ct, 0.0, 0.5 * H * st),
        ]
    )

    return _unique_xyz_rows(np.array(rows, dtype=float))


def _resolve_model_path(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    here = Path(__file__).resolve().parent
    for p in (here / "hand_landmarker.task", Path("hand_landmarker.task")):
        if p.is_file():
            return str(p)
    return "hand_landmarker.task"


def draw_hud(frame, lines, origin=(16, 16), line_h=26, pad=8, alpha=0.55):
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


def _hand_label(result, hand_idx: int) -> str:
    if result.handedness and hand_idx < len(result.handedness):
        return result.handedness[hand_idx][0].category_name
    return "?"


def find_hand_index_by_side(result, side: str) -> Optional[int]:
    """side: 'right' or 'left' (case-insensitive)."""
    if not result.hand_landmarks or not result.handedness:
        return None
    side = side.lower()
    for idx in range(len(result.hand_landmarks)):
        if idx < len(result.handedness):
            name = result.handedness[idx][0].category_name.lower()
            if name == side:
                return idx
    return None


def extract_world_points_mm_result(result, hand_idx: int):
    """21×(x,y,z) mm from MediaPipe world landmarks."""
    if not result.hand_world_landmarks or hand_idx >= len(result.hand_world_landmarks):
        return None
    wlm = result.hand_world_landmarks[hand_idx]
    pts = []
    for w in wlm:
        pts.append(
            (
                float(w.x * 1000.0),
                float(-w.y * 1000.0),
                float(-w.z * 1000.0),
            )
        )
    return pts


def draw_single_hand(
    frame,
    result,
    hand_idx: int,
    *,
    point_bgr=(0, 255, 0),
    line_bgr=(255, 0, 0),
    label_bgr=(255, 255, 255),
    line_thickness=2,
    label_suffix: str = "",
    label_text: Optional[str] = None,
    depth_map=None,
    print_depth=False,
):
    """Draw one hand; return list of one 21-point world list (mm) or empty."""
    keypoints_3d: List = []
    if not result.hand_landmarks or hand_idx >= len(result.hand_landmarks):
        return frame, keypoints_3d

    h, w, _ = frame.shape
    hand_landmarks = result.hand_landmarks[hand_idx]
    world_landmarks = None
    if hasattr(result, "hand_world_landmarks") and len(result.hand_world_landmarks) > hand_idx:
        world_landmarks = result.hand_world_landmarks[hand_idx]

    points = []
    points_3d = []
    for kp_id, lm in enumerate(hand_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, point_bgr, -1)

        depth_mm = None
        if depth_map is not None and y < depth_map.shape[0] and x < depth_map.shape[1]:
            depth_mm = int(depth_map[y, x])

        if world_landmarks is not None and kp_id < len(world_landmarks):
            wlm = world_landmarks[kp_id]
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
                print(f"kp:{kp_id:02d} x:{x:4d} y:{y:4d} depth_mm:{depth_mm:5d}")

    for connection in HAND_CONNECTIONS:
        p1 = points[connection[0]]
        p2 = points[connection[1]]
        cv2.line(frame, p1, p2, line_bgr, line_thickness)

    if label_text is not None:
        label = label_text
    else:
        label = _hand_label(result, hand_idx) + label_suffix
    cv2.putText(
        frame,
        label,
        (points[0][0], max(24, points[0][1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        label_bgr,
        2,
        cv2.LINE_AA,
    )

    keypoints_3d.append(points_3d)
    return frame, keypoints_3d


def draw_all_hands(
    frame,
    result,
    *,
    mode_hand_idx: Optional[int] = None,
    morph_hand_idx: Optional[int] = None,
    morph_mode: int = 1,
    open_value: Optional[float] = None,
    depth_map=None,
    print_depth=False,
):
    """
    Draw every detected hand with Left=orange tint, Right=green/blue lines.
    Thicker skeleton: left = mode, right = open/morph.
    Wrist labels: left hand shows ``M{mode}``; right hand shows ``open {value}``.
    Returns (frame, dict idx -> 21 world points).
    """
    out: dict = {}
    if not result.hand_landmarks:
        return frame, out

    left_pt = (0, 200, 255)
    left_ln = (255, 180, 60)
    right_pt = (0, 255, 80)
    right_ln = (60, 60, 255)

    for idx in range(len(result.hand_landmarks)):
        name = _hand_label(result, idx).lower()
        is_left = name == "left"
        pt = left_pt if is_left else right_pt
        ln = left_ln if is_left else right_ln
        lb = (255, 255, 200) if is_left else (200, 255, 200)
        th = 2
        if mode_hand_idx is not None and idx == mode_hand_idx:
            th = 4
        if morph_hand_idx is not None and idx == morph_hand_idx:
            th = max(th, 4)
        label_override: Optional[str] = None
        if mode_hand_idx is not None and idx == mode_hand_idx:
            label_override = f"M{int(morph_mode)}"
        elif morph_hand_idx is not None and idx == morph_hand_idx:
            if open_value is not None:
                label_override = f"open {float(open_value):.2f}"
            else:
                label_override = "open —"
        frame, kp = draw_single_hand(
            frame,
            result,
            idx,
            point_bgr=pt,
            line_bgr=ln,
            label_bgr=lb,
            line_thickness=th,
            label_suffix="",
            label_text=label_override,
            depth_map=depth_map,
            print_depth=print_depth,
        )
        if kp and len(kp) > 0:
            out[idx] = kp[0]
    return frame, out


def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def palm_center_and_scale(hand_points: Sequence[Tuple[float, float, float]]):
    palm_ids = [WRIST_ID] + MCP_IDS
    palm_pts = np.array(
        [hand_points[i] for i in palm_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if palm_pts.shape[0] == 0:
        return None, 1.0
    palm_center = palm_pts.mean(axis=0)
    wrist = np.array(hand_points[WRIST_ID], dtype=float)
    scale = float(np.mean(np.linalg.norm(palm_pts - wrist, axis=0))) + 1e-6
    return palm_center, scale


def classify_mode_from_fingers(hand_points: Sequence[Tuple[float, float, float]]):
    """
    Count index/middle/ring tips whose distance to palm is near the current max
    (within MODE_TIER_GAP in normalized units). That count maps to mode 1/2/3.
    Fist / low signal → tier 0 → mode 1 default.
    """
    pc, scale = palm_center_and_scale(hand_points)
    if pc is None:
        return 1, 0, {"d_norm": [], "reason": "no_palm"}

    dists = []
    for tid in MODE_COUNT_TIP_IDS:
        if tid >= len(hand_points) or np.isnan(hand_points[tid][2]):
            dists.append(0.0)
        else:
            p = np.array(hand_points[tid], dtype=float)
            dists.append(float(np.linalg.norm(p - pc)))
    dn = np.array(dists, dtype=float) / scale

    if dn.size == 0 or float(np.max(dn)) < MODE_EXTEND_MIN:
        return 1, 0, {"d_norm": dn.tolist(), "reason": "fist_or_low"}

    mx = float(np.max(dn))
    gap = max(MODE_TIER_GAP, 0.08 * mx)
    tier = int(np.sum(dn >= mx - gap))
    tier = max(1, min(3, tier))
    return tier, tier, {"d_norm": dn.tolist(), "max": mx, "gap": gap, "reason": "ok"}


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
    open_alpha = _clamp01(open_alpha)
    sphere_alpha = 1.0 - open_alpha
    R_sq = _morph_plane_extent_radius(radius, open_alpha)
    # Polar blanket on a disk with the **same area** as the canonical square [-R_sq,R_sq]² used in modes 2/3
    R_disk = _morph_plane_square_half_width_to_disk_radius(R_sq)

    x_ref, y_ref, z_plane = _blanket_param(R_disk, sphere_alpha=0.0)
    x_s, y_s, z_s = _blanket_param(R_disk, sphere_alpha=1.0)

    if show_refs:
        ax.plot_wireframe(x_ref, y_ref, z_plane, color="0.7", linewidth=0.35, alpha=0.55)
        ax.plot_wireframe(x_s, y_s, z_s, color="0.85", linewidth=0.25, alpha=0.35)
        ax.plot_wireframe(x_s, y_s, -z_s, color="0.85", linewidth=0.25, alpha=0.35)

    x, y, z = _blanket_param(R_disk, sphere_alpha=sphere_alpha)
    ax.plot_surface(x, y, z, color="tab:cyan", alpha=0.35, linewidth=0)
    ax.plot_surface(x, y, -z, color="tab:cyan", alpha=0.35, linewidth=0)
    ax.plot_wireframe(x, y, z, color="tab:blue", linewidth=0.35, alpha=0.55)
    ax.plot_wireframe(x, y, -z, color="tab:blue", linewidth=0.35, alpha=0.55)

    rr = np.linspace(0.0, 1.0, 80)
    xs = R_disk * rr
    ys = np.zeros_like(xs)
    zcap = R_disk * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0))
    ax.plot(xs, ys, sphere_alpha * zcap, color="tab:purple", linewidth=2.0, alpha=0.9)
    ax.plot(xs, ys, -sphere_alpha * zcap, color="tab:purple", linewidth=2.0, alpha=0.9)

    R = float(R_disk)
    num_rings = 4
    pts_per_ring = 6

    def build_ring_points():
        top_list = []
        bot_list = []
        ring_polys_top = []
        ring_polys_bot = []
        z0 = sphere_alpha * R
        top_list.append((0.0, 0.0, z0))
        off_scale = float((1.0 - sphere_alpha) ** 2)
        off_r = 0.06 * float(R_disk) * off_scale
        bot_list.append((off_r * np.cos(np.pi / 4.0), off_r * np.sin(np.pi / 4.0), -z0))
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

    for arr in ring_polys_top:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="tab:green", linewidth=1.0, alpha=0.55)
    for arr in ring_polys_bot:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="darkgreen", linewidth=1.0, alpha=0.45)

    # Plane-like state: same n×n grid on [-R,R]² as mode2/mode3 (coverage + spacing match)
    if sphere_alpha <= MORPH_PLANE_GRID_SPHERE_ALPHA:
        grid = _morph_plane_grid_points_z0(R_sq, MORPH_PLANE_GRID_N)
        ax.scatter(
            grid[:, 0],
            grid[:, 1],
            grid[:, 2],
            c="k",
            s=MORPH_SAMPLE_SCATTER_S,
            alpha=MORPH_SAMPLE_ALPHA,
        )
    elif sphere_alpha < 0.28:
        eps = max(3.5, 0.020 * R_sq)
        top = pts_top.copy()
        bot = pts_bottom.copy()
        top[:, :2] *= 0.88
        bot[:, :2] *= 1.12
        ax.scatter(
            top[:, 0], top[:, 1], top[:, 2] + eps, c="k", s=MORPH_SAMPLE_SCATTER_S, alpha=MORPH_SAMPLE_ALPHA
        )
        ax.scatter(
            bot[:, 0], bot[:, 1], bot[:, 2] - eps, c="k", s=MORPH_SAMPLE_SCATTER_S, alpha=MORPH_SAMPLE_ALPHA
        )
    else:
        ax.scatter(pts_top[:, 0], pts_top[:, 1], pts_top[:, 2], c="k", s=MORPH_SAMPLE_SCATTER_S, alpha=MORPH_SAMPLE_ALPHA)
        ax.scatter(
            pts_bottom[:, 0], pts_bottom[:, 1], pts_bottom[:, 2], c="k", s=MORPH_SAMPLE_SCATTER_S, alpha=MORPH_SAMPLE_ALPHA
        )


def draw_pyramid_morph_canonical(ax, radius, open_alpha, show_refs=True):
    """
    四棱锥：四个 **三角形侧面** 用与 box 类似的半透明色面 + wireframe；底面仅浅线框。

    顶点为 ``(0,0,z_apex)``，``z_apex = close·H``；平面态（close→0）侧面塌成底面内以原点为公共顶点的扇形。
    黑色采样点 **只**铺在四个侧面上（按面积均分 ``MORPH_PLANE_GRID_N²`` 个点）。
    """
    open_alpha = _clamp01(open_alpha)
    close = 1.0 - open_alpha
    R0 = float(radius)
    R = _morph_plane_extent_radius(R0, open_alpha)
    H_peak = R0 * (0.38 + 0.92 * close)
    Pz = close * H_peak
    P = np.array([0.0, 0.0, Pz])

    faces = (
        (np.array([R, -R, 0.0]), np.array([R, R, 0.0])),
        (np.array([R, R, 0.0]), np.array([-R, R, 0.0])),
        (np.array([-R, R, 0.0]), np.array([-R, -R, 0.0])),
        (np.array([-R, -R, 0.0]), np.array([R, -R, 0.0])),
    )
    n_mesh = 16
    lin = np.linspace(0.0, 1.0, n_mesh)
    U, V = np.meshgrid(lin, lin)
    mask = U + V <= 1.0 + 1e-9
    colors = ("tab:cyan", "olive", "tab:cyan", "olive")

    if show_refs:
        P_ref = np.array([0.0, 0.0, H_peak])
        for A, B in faces:
            w = 1.0 - U - V
            Xr = w * P_ref[0] + U * A[0] + V * B[0]
            Yr = w * P_ref[1] + U * A[1] + V * B[1]
            Zr = w * P_ref[2] + U * A[2] + V * B[2]
            ax.plot_wireframe(
                np.where(mask, Xr, np.nan),
                np.where(mask, Yr, np.nan),
                np.where(mask, Zr, np.nan),
                color="0.82",
                linewidth=0.2,
                alpha=0.28,
            )

    for (A, B), c in zip(faces, colors):
        w = 1.0 - U - V
        X = w * P[0] + U * A[0] + V * B[0]
        Y = w * P[1] + U * A[1] + V * B[1]
        Z = w * P[2] + U * A[2] + V * B[2]
        Xm = np.where(mask, X, np.nan)
        Ym = np.where(mask, Y, np.nan)
        Zm = np.where(mask, Z, np.nan)
        ax.plot_surface(Xm, Ym, Zm, color=c, alpha=0.34, linewidth=0)
        ax.plot_wireframe(Xm, Ym, Zm, color="tab:blue", linewidth=0.28, alpha=0.45)

    for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        xc = sx * R
        yc = sy * R
        ax.plot([xc, P[0]], [yc, P[1]], [0.0, P[2]], color="darkred", linewidth=1.4, alpha=0.82)

    base_xy = np.array([[-R, R, R, -R, -R], [-R, -R, R, R, -R]], dtype=float)
    ax.plot(base_xy[0], base_xy[1], np.zeros(5), color="0.35", linewidth=0.9, alpha=0.55)

    n_tot = MORPH_PLANE_GRID_N * MORPH_PLANE_GRID_N
    n0, n1, n2, n3 = _allocate_integer_by_weights(n_tot, [1.0, 1.0, 1.0, 1.0])
    pts: List[Tuple[float, float, float]] = []
    for (A, B), nc in zip(faces, (n0, n1, n2, n3)):
        pts.extend(_triangle_bary_samples(nc, A, B, P))
    sp = np.array(pts, dtype=float)
    if sp.size:
        ax.scatter(
            sp[:, 0],
            sp[:, 1],
            sp[:, 2],
            c="k",
            s=MORPH_SAMPLE_SCATTER_S,
            alpha=MORPH_SAMPLE_ALPHA,
        )


def draw_boxfold_morph_canonical(ax, radius, open_alpha, show_refs=True):
    """
    Axis-aligned hinges (y=±R, x=±R): four walls fold from flat (open hand) toward vertical (fist).

    ``R_open = MORPH_BOX_OPEN_BASE_SCALE·R_ref`` (small base). ``R_fold = MORPH_BOX_FOLD_SIZE_SCALE·R_geom``
    with ``R_geom`` from the net-area formula. ``R = (1-close)·R_open + close·R_fold``. Wall height
    ``H = R·MORPH_BOX_WALL_HEIGHT_RATIO`` (taller sides than ``k_net`` alone).

    Black scatter: corners + face centers only (no edge mids).
    open_alpha=1 → θ=0 (flat). open_alpha=0 → θ=90° (walls vertical).
    """
    open_alpha = _clamp01(open_alpha)
    close = 1.0 - open_alpha
    R_ref = _morph_plane_extent_radius(radius, open_alpha)
    R_geom = _morph_box_half_width_for_matching_net_area(R_ref, MORPH_BOX_NET_K)
    R_fold = R_geom * float(MORPH_BOX_FOLD_SIZE_SCALE)
    R_open = R_ref * float(MORPH_BOX_OPEN_BASE_SCALE)
    R = (1.0 - close) * R_open + close * R_fold
    theta = close * (0.5 * np.pi)
    H = R * MORPH_BOX_WALL_HEIGHT_RATIO
    n_u = 10
    n_v = 8

    # --- Bottom (reference): square [-R,R]^2 at z=0 (R morphs from small plane footprint to full box)
    if show_refs:
        sb = 18
        xb = np.linspace(-R, R, sb)
        yb = np.linspace(-R, R, sb)
        XB, YB = np.meshgrid(xb, yb)
        ZB = np.zeros_like(XB)
        ax.plot_surface(XB, YB, ZB, color="0.85", alpha=0.22, linewidth=0)
        ax.plot_wireframe(XB, YB, ZB, color="0.55", linewidth=0.35, alpha=0.5)

    us = np.linspace(-R, R, n_u)
    vs = np.linspace(0.0, H, n_v)
    U, V = np.meshgrid(us, vs)
    Ye, Ve = np.meshgrid(us, vs)

    x_n = U
    y_n = R + V * np.cos(theta)
    z_n = V * np.sin(theta)
    y_s = -R - V * np.cos(theta)
    z_s = V * np.sin(theta)
    x_s = U
    x_e = R + Ve * np.cos(theta)
    y_e = Ye
    z_e = Ve * np.sin(theta)
    x_w = -R - Ve * np.cos(theta)
    y_w = Ye
    z_w = Ve * np.sin(theta)

    for xx, yy, zz, c in (
        (x_n, y_n, z_n, "tab:cyan"),
        (x_s, y_s, z_s, "olive"),
        (x_e, y_e, z_e, "tab:cyan"),
        (x_w, y_w, z_w, "olive"),
    ):
        ax.plot_surface(xx, yy, zz, color=c, alpha=0.34, linewidth=0)
        ax.plot_wireframe(xx, yy, zz, color="tab:blue", linewidth=0.3, alpha=0.45)

    sp = _boxfold_anchor_points(R, H, theta)
    if sp.size:
        ax.scatter(
            sp[:, 0],
            sp[:, 1],
            sp[:, 2],
            c="k",
            s=MORPH_SAMPLE_SCATTER_S,
            alpha=MORPH_SAMPLE_ALPHA,
        )


def analyze_hand_topology(hand_points):
    all_pts = np.array(hand_points, dtype=float)
    valid_all = ~np.isnan(all_pts[:, 2])
    if np.sum(valid_all) < 8:
        return None

    fit_ids = [i for i in range(len(hand_points)) if i != WRIST_ID]
    fit_pts = np.array(
        [hand_points[i] for i in fit_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if fit_pts.shape[0] < 7:
        fit_pts = all_pts[valid_all]

    centroid = fit_pts.mean(axis=0)
    centered = fit_pts - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    normal = _safe_normalize(eigvecs[:, 2])
    lamb_sum = float(np.sum(eigvals)) + 1e-8
    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-8))
    isotropy = float(eigvals[2] / lamb_sum)
    span_ratio = float(eigvals[0] / (eigvals[2] + 1e-8))

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

    spread_score = _clamp01((finger_spread - 1.00) / (1.65 - 1.00))
    planarity_score = _clamp01((planarity - 0.12) / (0.55 - 0.12))
    isotropy_score = _clamp01((isotropy - 0.06) / (0.22 - 0.06))
    alpha = _clamp01(0.50 * spread_score + 0.35 * planarity_score + 0.15 * (1.0 - isotropy_score))
    alpha = _clamp01(alpha ** OPEN_GAMMA)

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
    morph_mode: int,
    morph_alpha_smoothed=None,
    control_label: str = "",
    *,
    shape_normalized: bool = False,
    hand_frame: str = HAND_FRAME_SCALED,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
):
    ax_hand.clear()
    ax_topo.clear()

    _src = "MediaPipe" if hand_3d_source == HAND_3D_SOURCE_MP else "depth+MP fused"
    t = f"Hand Keypoints 3D ({_src}) — joints 0..20"
    if control_label:
        t += f" — {control_label}"
    ax_hand.set_title(t)
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

    mode_titles = {
        1: "Mode1: plane↔sphere",
        2: "Mode2: plane↔pyramid",
        3: "Mode3: plane↔box walls",
    }
    ax_topo.set_title(mode_titles.get(morph_mode, "Morph"))
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
            morph_fb = 0.55
            if morph_alpha_smoothed is not None:
                morph_fb = float(morph_alpha_smoothed)
            morph_fb = _clamp01(morph_fb)
            r_draw = 200.0
            if morph_mode == 1:
                draw_blanket_morph_canonical(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
                extra = "Refs: gray=plane, light=sphere caps (fallback)"
            elif morph_mode == 2:
                draw_pyramid_morph_canonical(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
                extra = "Square pyramid (fallback)"
            else:
                draw_boxfold_morph_canonical(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
                extra = "Box walls (fallback)"
            ax_topo.text(
                -r_draw,
                -r_draw,
                r_draw * 0.92,
                f"{extra}\nopen={morph_fb:.2f}\nTopology unavailable this frame.",
                color="tab:orange",
            )
            continue
        analyses.append(analysis)

        r = max(analysis["radius"], 1.0)
        morph_alpha = analysis["morph_alpha"]
        if morph_alpha_smoothed is not None:
            morph_alpha = morph_alpha_smoothed

        Rvis = max(140.0, 2.2 * r)
        if morph_mode == 1:
            draw_blanket_morph_canonical(ax_topo, radius=Rvis, open_alpha=morph_alpha, show_refs=True)
            extra = "Refs: gray=plane, light=sphere caps"
        elif morph_mode == 2:
            draw_pyramid_morph_canonical(ax_topo, radius=Rvis, open_alpha=morph_alpha, show_refs=True)
            extra = "Square pyramid: layers=max(|x|,|y|); red ridges"
        else:
            draw_boxfold_morph_canonical(ax_topo, radius=Rvis, open_alpha=morph_alpha, show_refs=True)
            extra = "Bottom z=0; 4 walls rotate on hinges θ=(1-open)·90°"

        topo_note = f"  λ0/λ2={analysis['span_ratio']:.1f}" if shape_normalized else ""
        ax_topo.text(
            -Rvis,
            -Rvis,
            Rvis * 0.92,
            f"{extra}\nopen={morph_alpha:.2f}  "
            f"plan={analysis['planarity']:.2f}  iso={analysis['isotropy']:.2f}{topo_note}",
            color="tab:purple",
        )

    ax_hand.view_init(elev=20, azim=-70)
    ax_topo.view_init(elev=22, azim=-58)
    ax_hand.set_box_aspect((1.0, 1.0, 1.0))
    ax_topo.set_box_aspect((1.0, 1.0, 1.0))

    lim = MORPH_AXIS_LIM_MM
    ax_topo.set_xlim(-lim, lim)
    ax_topo.set_ylim(-lim, lim)
    ax_topo.set_zlim(-lim, lim)
    return analyses


def main():
    ap = argparse.ArgumentParser(
        description="Left hand → mode 1/2/3; Right hand → open/morph only",
    )
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    args = ap.parse_args()
    model_path = _resolve_model_path(args.model)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    with HandLandmarker.create_from_options(options) as landmarker:
        plt.ion()
        fig = plt.figure("Hand 3D — Webcam modes")
        ax_hand = fig.add_subplot(121, projection="3d")
        ax_topo = fig.add_subplot(122, projection="3d")

        print(
            "Left hand = MODE (1/2/3). Right hand = OPEN / shape morph.  q=quit p=3D s=save"
        )

        try:
            frame_idx = 0
            open_free_ema = None
            alpha_smooth = 0.18
            snap_state = None
            hud_cache = {"open": None, "free": None, "plan": None, "iso": None, "spread": None, "text": None}
            snap_vis_state = None
            snap_stable_frames = 0
            snap_hold_frames = 0
            enable_3d = ENABLE_3D_PLOT
            mode_raw = 1
            mode_ema = 1.0
            mode_smooth = 0.22
            morph_mode = 1
            mode_raw_prev: Optional[int] = None
            mode_stable_frames = 0
            last_right_pts: Optional[list] = None
            last_open_out: Optional[float] = None
            last_mode_raw = 1

            while True:
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

                idx_L = find_hand_index_by_side(result, "left")
                idx_R = find_hand_index_by_side(result, "right")

                pts_L = (
                    extract_world_points_mm_result(result, idx_L) if idx_L is not None else None
                )

                # --- Mode: LEFT hand only (right hand ignored for mode) ---
                tier_count = -1
                if pts_L is not None:
                    mode_raw, tier_count, _dbg = classify_mode_from_fingers(pts_L)
                    last_mode_raw = mode_raw
                    if mode_raw_prev is None:
                        mode_raw_prev = mode_raw
                        mode_stable_frames = MODE_DEBOUNCE_FRAMES
                    elif mode_raw == mode_raw_prev:
                        mode_stable_frames += 1
                    else:
                        mode_raw_prev = mode_raw
                        mode_stable_frames = 0

                    if mode_stable_frames >= MODE_DEBOUNCE_FRAMES:
                        mode_ema = mode_smooth * float(mode_raw) + (1.0 - mode_smooth) * float(mode_ema)
                        morph_mode = int(round(mode_ema))
                        morph_mode = max(1, min(3, morph_mode))
                else:
                    mode_raw = last_mode_raw

                # --- Open / morph: RIGHT hand only (left hand ignored for open) ---
                hands_3d: List = []
                open_out: Optional[float] = None
                pts_R = None
                if idx_R is not None:
                    pts_R = extract_world_points_mm_result(result, idx_R)
                    if pts_R is not None:
                        last_right_pts = list(pts_R)
                        hands_3d = [pts_R]

                        tmp = analyze_hand_topology(pts_R)
                        if tmp is not None:
                            if open_free_ema is None:
                                open_free_ema = float(tmp["morph_alpha"])
                            else:
                                open_free_ema = (
                                    alpha_smooth * float(tmp["morph_alpha"])
                                    + (1.0 - alpha_smooth) * open_free_ema
                                )

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
                            last_open_out = float(open_out)
                else:
                    if last_right_pts is not None:
                        hands_3d = [last_right_pts]
                    open_out = last_open_out

                frame, _kp_map = draw_all_hands(
                    frame,
                    result,
                    mode_hand_idx=idx_L,
                    morph_hand_idx=idx_R,
                    morph_mode=morph_mode,
                    open_value=open_out,
                    depth_map=None,
                    print_depth=False,
                )

                hint_parts = []
                if idx_L is None:
                    hint_parts.append("no LEFT (mode)")
                if idx_R is None:
                    hint_parts.append("no RIGHT (open frozen)")
                hint = "  |  ".join(hint_parts) if hint_parts else "L=mode  R=open"
                cv2.putText(
                    frame,
                    f"L→M{morph_mode} raw:{mode_raw}  R→open:{open_out if open_out is not None else '—'}  "
                    f"tier:{tier_count if tier_count >= 0 else '-'}  {hint}"[:95],
                    (16, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

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
                if enable_3d and (frame_idx % PLOT_EVERY_N_FRAMES) == 0 and hands_3d:
                    analyses = update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_mode=morph_mode,
                        morph_alpha_smoothed=open_out,
                        control_label="Right→open",
                    )
                    plt.pause(0.0001)

                if analyses:
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
                            f"L→mode M{morph_mode}{snap_txt}  (1=sph 2=pyr 3=box)  |  R hand metrics",
                            f"open:{open_disp:.2f}  free:{free_disp:.2f}",
                            f"spread:{a0['finger_spread']:.2f}  plan:{a0['planarity']:.2f}  iso:{a0['isotropy']:.2f}",
                        ]

                    if frame_idx % 30 == 0:
                        out_v = open_out if open_out is not None else a0["morph_alpha"]
                        free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                        print(
                            f"M{morph_mode} raw={mode_raw} open_out={out_v:.3f} free={free_v:.3f} "
                            f"spread={a0['finger_spread']:.3f} plan={a0['planarity']:.3f} iso={a0['isotropy']:.3f}"
                        )

                cv2.imshow("Hand Tracking Webcam Modes", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    out_name = f"hand_webcam_mode_frame_{frame_idx:06d}.png"
                    fig.savefig(out_name, dpi=150, bbox_inches="tight")
                    print(f"Saved: {out_name}")
                if key == ord("p"):
                    enable_3d = not enable_3d
                    print(f"3D plot: {enable_3d}")
                if key == ord("q"):
                    break

                if hud_cache["text"] is not None:
                    draw_hud(frame, hud_cache["text"], origin=(16, 16))
                frame_idx += 1
        finally:
            cap.release()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
