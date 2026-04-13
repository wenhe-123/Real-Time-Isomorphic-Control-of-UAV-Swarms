"""
Dual-view hand tracking: Orbbec RGB-D + laptop webcam.

Orbbec: ``hand_tracking_orbbec.draw_hand`` (depth + calibration). Webcam: MediaPipe world.

**Per-view confidence (what we draw as vis_μ / vis_min):**
  Each 2D landmark carries **visibility** (preferred) or **presence** from MediaPipe Hands.
  These are network-estimated scores in ``[0,1]`` for “how reliably this joint is localized
  in the image”.  **Occlusion / motion blur / self-occlusion** typically **lower** visibility on
  the affected joints, which pulls down **vis_min** (worst joint) and often **vis_mean**.
  They are **not** depth-based; depth fusion is separate (Orbbec pane).

**w_geom:** geometry weight from PCA shape (planarity / isotropy) on that view’s 21×3 world
points — used as a **per-view** multiplier together with per-joint visibility in fusion.

**Fusion (MP world mm):** ``c_{v,k} = visibility_{v,k} * w_geom_v``; weighted mean and
``--fuse-vis-low/high`` gates as before.  **open** from fused hand via ``analyze_hand_topology``.

**Two landmarkers:** ``VIDEO`` mode keeps per-stream temporal state. One shared landmarker
alternating two cameras confuses tracking — use **one HandLandmarker per camera** and separate
timestamps, then fuse the two world-space results.

3D plot: ``hand_tracking_orbbec.update_3d_plot``.  Controls: q / p / s.
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyk4a import Config, FPS, PyK4A

import hand_tracking_orbbec as ob

# ===== MediaPipe initialize =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# ===== connecting（21 points）=====
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]
WRIST_ID = 0
MCP_IDS = [5, 9, 13, 17]

PLOT_EVERY_N_FRAMES = 5
ENABLE_3D_PLOT = True


def _open_webcam_capture(preferred_index: int, width: int, height: int, max_probe_index: int = 8):
    """
    Open webcam robustly with backend fallback and read-frame validation.
    Returns (cap, selected_index, backend_name).
    """
    backend_candidates = [("ANY", cv2.CAP_ANY)]
    if hasattr(cv2, "CAP_V4L2"):
        backend_candidates.insert(0, ("V4L2", cv2.CAP_V4L2))

    if preferred_index >= 0:
        indices = [preferred_index]
    else:
        indices = list(range(max_probe_index + 1))

    failures = []
    for idx in indices:
        for backend_name, backend in backend_candidates:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                failures.append(f"index={idx} backend={backend_name}: open failed")
                continue

            if width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

            ok = False
            for _ in range(8):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok = True
                    break
            if ok:
                return cap, idx, backend_name

            cap.release()
            failures.append(f"index={idx} backend={backend_name}: read failed")

    raise RuntimeError(
        "Cannot open a usable webcam. "
        f"preferred_index={preferred_index}, tried indices={indices}. "
        f"Details: {'; '.join(failures[:12])}"
    )


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


def draw_hand_webcam(frame, result, depth_map=None, print_depth=False):
    keypoints_3d = []
    if result.hand_landmarks:
        h, w, _ = frame.shape

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            points = []
            points_3d = []
            world_landmarks = None
            if hasattr(result, "hand_world_landmarks") and len(result.hand_world_landmarks) > idx:
                world_landmarks = result.hand_world_landmarks[idx]

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
                        print(
                            f"hand:{idx} kp:{kp_id:02d} x:{x:4d} y:{y:4d} depth_mm:{depth_mm:5d}"
                        )

            for connection in HAND_CONNECTIONS:
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

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


def extract_world_points_mm(result, hand_idx=0):
    """Return list of 21 (x,y,z) mm or None if missing."""
    if (
        not result.hand_world_landmarks
        or hand_idx >= len(result.hand_world_landmarks)
    ):
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


def extract_landmark_visibilities(result, hand_idx: int = 0) -> Optional[np.ndarray]:
    """Per-joint confidence in [0,1]: MediaPipe ``visibility``, else ``presence``, else 1."""
    if not result.hand_landmarks or hand_idx >= len(result.hand_landmarks):
        return None
    hlm = result.hand_landmarks[hand_idx]
    out = np.ones(21, dtype=np.float64)
    for i, lm in enumerate(hlm):
        v = getattr(lm, "visibility", None)
        if v is None:
            v = getattr(lm, "presence", None)
        if v is not None:
            out[i] = float(np.clip(float(v), 0.0, 1.0))
    return out


def summarize_mp_visibility(vis: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """Mean / min over 21 joints — quick per-view “how sure is MP about this hand in this image”."""
    if vis is None or vis.size < 1:
        return None
    return {
        "mean": float(np.mean(vis)),
        "min": float(np.min(vis)),
    }


def _conf_color(mean_vis: float) -> Tuple[int, int, int]:
    """BGR: green = confident, yellow = mid, blue-ish = low (readable on dark)."""
    if mean_vis >= 0.72:
        return (60, 220, 80)
    if mean_vis >= 0.45:
        return (80, 200, 255)
    return (100, 120, 255)


def _pca_eigenvalues_descending_mm(pts_21: List[Tuple[float, float, float]]) -> Optional[np.ndarray]:
    """
    PCA on all joints with finite coordinates (≥4 points). Used when analyze_hand_topology
    rejects the cloud (<8 valid etc.) but we still want λ₁,λ₂,λ₃ for debug / w_geom fallback.
    """
    arr = np.asarray(pts_21, dtype=float)
    if arr.size != 63:
        return None
    mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    sub = arr[mask]
    if sub.shape[0] < 4:
        return None
    centered = sub - sub.mean(axis=0)
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return None
    ev = np.linalg.eigh(cov)[0]
    return np.sort(ev.astype(float))[::-1]


def _w_geom_from_planarity_isotropy(planarity: float, isotropy: float) -> float:
    ps = float(np.clip((planarity - 0.12) / (0.55 - 0.12), 0.0, 1.0))
    isos = float(np.clip((isotropy - 0.06) / (0.22 - 0.06), 0.0, 1.0))
    w = float(0.5 * ps + 0.5 * (1.0 - isos))
    return float(np.clip(w, 0.05, 1.0))


def _w_geom_from_eigenvalues(ev: np.ndarray) -> float:
    """Same mapping as analyze_hand_topology when only λ₀,λ₁,λ₂ are known."""
    l0, l1, l2 = float(ev[0]), float(ev[1]), float(ev[2])
    ls = l0 + l1 + l2 + 1e-8
    planarity = (l1 - l2) / (l0 + 1e-8)
    isotropy = l2 / ls
    return _w_geom_from_planarity_isotropy(planarity, isotropy)


def _geom_weight_from_eigen_analysis(pts_21: List[Tuple[float, float, float]]) -> Tuple[float, np.ndarray]:
    """
    Scalar geometry confidence in (0,1] from PCA eigenvalues; aligned with ``analyze_hand_topology``.
    Falls back to direct PCA on ≥4 finite joints if the full topology check fails.
    Returns (w_geom, eigvals_descending); degenerate → (0.15, nan×3).
    """
    ev_fb = _pca_eigenvalues_descending_mm(pts_21)
    an = ob.analyze_hand_topology(pts_21)
    if an is not None:
        p = float(an["planarity"])
        iso = float(an["isotropy"])
        w_geom = _w_geom_from_planarity_isotropy(p, iso)
        ev = np.linalg.eigh(np.cov(np.asarray(an["points"], dtype=float).T))[0]
        ev = np.sort(ev)[::-1]
        return w_geom, ev
    if ev_fb is not None:
        w_geom = _w_geom_from_eigenvalues(ev_fb)
        return w_geom, ev_fb
    return 0.15, np.full(3, np.nan)


def fuse_dual_views_weighted(
    pts_o: Optional[List[Tuple[float, float, float]]],
    pts_l: Optional[List[Tuple[float, float, float]]],
    vis_o: Optional[np.ndarray],
    vis_l: Optional[np.ndarray],
    w_geom_o: float,
    w_geom_l: float,
    *,
    conf_low: float,
    conf_high: float,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """
    Per joint: c_{v,k} = vis_{v,k} * w_geom_v. Skip if c < conf_low; if any c >= conf_high,
    use the view with max c only; else x = sum(c x)/sum(c).
    """
    fused: List[Tuple[float, float, float]] = []
    w_sum_o = 0.0
    w_sum_l = 0.0
    n_joint = 0
    n_high_pick = 0

    for k in range(21):
        cand: List[Tuple[str, np.ndarray, float]] = []
        for name, pts, vis, wg in (
            ("O", pts_o, vis_o, w_geom_o),
            ("L", pts_l, vis_l, w_geom_l),
        ):
            if pts is None or vis is None or not np.isfinite(float(wg)):
                continue
            p = pts[k]
            if p is None or len(p) < 3 or not np.isfinite(p[2]):
                continue
            c = float(vis[k]) * float(wg)
            if c < conf_low:
                continue
            cand.append((name, np.array(p, dtype=float), c))

        if not cand:
            fused.append((np.nan, np.nan, np.nan))
            continue

        hi = [t for t in cand if t[2] >= conf_high]
        if hi:
            best = max(hi, key=lambda t: t[2])
            fused.append((float(best[1][0]), float(best[1][1]), float(best[1][2])))
            n_high_pick += 1
            if best[0] == "O":
                w_sum_o += 1.0
            else:
                w_sum_l += 1.0
            n_joint += 1
            continue

        sw = sum(t[2] for t in cand)
        acc = sum(t[2] * t[1] for t in cand)
        p_f = acc / max(sw, 1e-12)
        fused.append((float(p_f[0]), float(p_f[1]), float(p_f[2])))
        for name, vec, c in cand:
            frac = c / sw
            if name == "O":
                w_sum_o += frac
            else:
                w_sum_l += frac
        n_joint += 1

    if n_joint <= 0:
        wmo, wml = 0.0, 0.0
    else:
        wmo = float(w_sum_o / n_joint)
        wml = float(w_sum_l / n_joint)
    dbg = {
        "w_mean_o": wmo,
        "w_mean_l": wml,
        "n_joints_fused": n_joint,
        "n_high_exclusive": n_high_pick,
    }
    return fused, dbg


def overlay_inset(
    dst,
    src,
    margin=12,
    max_w_ratio=0.28,
    *,
    footer_lines: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None,
):
    """
    Resize ``src`` BGR into top-right corner of ``dst``.
    ``footer_lines``: optional lines ``(text, bgr)`` drawn **inside** the inset at the bottom
    (webcam confidence, etc.).
    """
    if src is None or src.size == 0:
        return dst
    dh, dw = dst.shape[:2]
    max_w = int(dw * max_w_ratio)
    sh, sw = src.shape[:2]
    scale = min(max_w / float(sw), (dh * 0.35) / float(sh), 1.0)
    nw = max(1, int(sw * scale))
    nh = max(1, int(sh * scale))
    small = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    if footer_lines:
        overlay = small.copy()
        h_s = small.shape[0]
        y0 = h_s - 4
        for i, (txt, col) in enumerate(reversed(footer_lines[:3])):
            yy = y0 - i * 14
            if yy < 12:
                break
            cv2.putText(
                overlay,
                txt,
                (3, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                txt,
                (3, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                col,
                1,
                cv2.LINE_AA,
            )
        small = overlay
    y1 = margin
    y2 = margin + nh
    x2 = dw - margin
    x1 = x2 - nw
    if x1 < 0 or y2 > dh:
        return dst
    roi = dst[y1:y2, x1:x2]
    if roi.shape[:2] != small.shape[:2]:
        return dst
    dst[y1:y2, x1:x2] = small
    cv2.rectangle(dst, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    cv2.putText(
        dst,
        "webcam",
        (x1 + 4, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return dst


def main():
    ap = argparse.ArgumentParser(
        description="Dual Orbbec (depth-fused overlay) + webcam; fused MP world for topology / 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Orbbec pane: hand_tracking_orbbec.draw_hand. Dual fusion: per-joint "
            "c = visibility * w_geom(view), weighted mean with --fuse-vis-low/high; "
            "see module docstring. Depth CLI matches hand_tracking_orbbec.py."
        ),
    )
    ap.add_argument("--model", type=str, default=None, help="Path to hand_landmarker.task")
    ap.add_argument(
        "--webcam-index",
        type=int,
        default=-1,
        help="OpenCV webcam index (-1=auto scan, default -1)",
    )
    ap.add_argument("--webcam-width", type=int, default=0, help="Optional capture width (0=default)")
    ap.add_argument("--webcam-height", type=int, default=0, help="Optional capture height (0=default)")
    ap.add_argument(
        "--webcam-max-index",
        type=int,
        default=8,
        help="When --webcam-index=-1, probe [0..N] (default 8)",
    )
    ap.add_argument(
        "--depth-fusion",
        type=float,
        default=ob.DEPTH_FUSION_WEIGHT,
        help="Orbbec pane: 0=MP world only, 1=depth unproject only",
    )
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=ob.POINT_EMA_ALPHA,
        help="EMA on Orbbec 3D keypoints (0..1)",
    )
    ap.add_argument(
        "--hand-frame",
        choices=(ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE, ob.HAND_FRAME_METRIC_MM),
        default=ob.HAND_FRAME_SCALED,
        help="Orbbec draw_hand coordinate normalization (3D plot axes follow this for Orbbec pane)",
    )
    ap.add_argument(
        "--no-shape-normalize",
        action="store_true",
        help="Same as --hand-frame metric_mm unless palm_plane/metric_mm set",
    )
    ap.add_argument(
        "--no-depth-outlier-filter",
        action="store_true",
        help="Orbbec: keep all depth samples",
    )
    ap.add_argument(
        "--no-open-remap",
        action="store_true",
        help="HUD: raw morph_alpha (default: remap when shape_norm)",
    )
    ap.add_argument("--open-remap-lo", type=float, default=None, metavar="X")
    ap.add_argument("--open-remap-hi", type=float, default=None, metavar="X")
    ap.add_argument(
        "--depth-patch",
        type=int,
        default=ob.DEPTH_MEDIAN_PATCH_RADIUS,
        metavar="R",
        help="Median depth patch radius on Orbbec depth image",
    )
    ap.add_argument("--depth-max-delta-mm", type=float, default=None, metavar="D")
    ap.add_argument("--depth-median-max-delta-mm", type=float, default=None, metavar="D")
    ap.add_argument(
        "--use-transformed-depth",
        action="store_true",
        help="Use SDK depth registered to color (often unsafe on Orbbec pyk4a)",
    )
    ap.add_argument(
        "--hand-3d",
        choices=(ob.HAND_3D_SOURCE_MP, ob.HAND_3D_SOURCE_FUSED),
        default=ob.HAND_3D_SOURCE_MP,
        help="Orbbec pane 3D source for draw_hand",
    )
    ap.add_argument(
        "--depth-unproject-rigid-npy",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional 4×4 .npy rigid after depth unproject",
    )
    ap.add_argument(
        "--fuse-vis-low",
        type=float,
        default=0.12,
        help="Drop joint contribution if c=vis*w_geom < this (default 0.12)",
    )
    ap.add_argument(
        "--fuse-vis-high",
        type=float,
        default=0.72,
        help="If c >= this for a joint, use only the strongest view (100%%) for that joint",
    )
    ap.add_argument(
        "--fuse-debug-every",
        type=int,
        default=0,
        metavar="FRAMES",
        help=(
            "Print fusion debug (eigenvalues, w_geom, mean weights) every FRAMES frames. "
            "Must be a positive integer, e.g. 30 (not the letter N). 0=off."
        ),
    )
    ap.add_argument(
        "--fuse-debug",
        action="store_true",
        help="Shortcut for --fuse-debug-every 30",
    )
    args = ap.parse_args()
    model_path = _resolve_model_path(args.model)
    fusion_w = float(np.clip(args.depth_fusion, 0.0, 1.0))
    ema_a = float(np.clip(args.ema_alpha, 0.0, 1.0))
    depth_patch_r = int(np.clip(args.depth_patch, 0, 15))
    if args.hand_frame != ob.HAND_FRAME_SCALED:
        hand_frame = args.hand_frame
    elif args.no_shape_normalize:
        hand_frame = ob.HAND_FRAME_METRIC_MM
    else:
        hand_frame = ob.HAND_FRAME_SCALED
    shape_norm = hand_frame in (ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE)
    depth_outlier_filter = not args.no_depth_outlier_filter
    if shape_norm and not args.no_open_remap:
        lo = ob.OPEN_REMAP_LO if args.open_remap_lo is None else args.open_remap_lo
        hi = ob.OPEN_REMAP_HI if args.open_remap_hi is None else args.open_remap_hi
        open_remap = (lo, hi) if hi > lo + 1e-6 else None
    else:
        open_remap = None
    depth_max_delta_mm = (
        ob.DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    )
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = ob.DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)
    depth_rigid_T = ob.load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy)
    fuse_conf_low = float(np.clip(args.fuse_vis_low, 0.0, 0.99))
    fuse_conf_high = float(np.clip(args.fuse_vis_high, fuse_conf_low + 1e-4, 1.0))
    fuse_debug_every = max(0, int(args.fuse_debug_every))
    if args.fuse_debug and fuse_debug_every == 0:
        fuse_debug_every = 30

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
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

    cap, webcam_index_used, webcam_backend = _open_webcam_capture(
        preferred_index=args.webcam_index,
        width=args.webcam_width,
        height=args.webcam_height,
        max_probe_index=args.webcam_max_index,
    )
    print(f"[INFO] Webcam opened: index={webcam_index_used} backend={webcam_backend}")

    landmarker_o = HandLandmarker.create_from_options(options)
    landmarker_l = HandLandmarker.create_from_options(options)
    plt.ion()
    fig = plt.figure("Hand 3D and Topology (dual)")
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")

    k4a.start()
    calib = k4a.calibration
    print(
        "Dual Orbbec + webcam. Orbbec draw: "
        f"fusion={fusion_w:.2f} ema={ema_a:.2f} depth_patch={depth_patch_r} "
        f"hand_frame={hand_frame} hand_3d={args.hand_3d}  "
        f"q=quit  p=toggle 3D  s=save plot"
    )

    try:
        frame_idx = 0
        mp_ts_o_ms = 0
        mp_ts_l_ms = 0
        warned_fusion_linear_map = False
        ema_3d = None
        open_free_ema = None
        alpha_smooth = 0.18
        snap_state = None
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
            ok_cam, frame_web = cap.read()
            if not ok_cam or frame_web is None:
                print("[WARN] webcam read failed")
                continue

            if capture.color is None:
                continue

            color = capture.color
            if color.ndim == 3 and color.shape[2] == 4:
                frame = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            else:
                frame = color

            depth_raw = capture.depth
            depth_map = depth_raw
            if depth_raw is not None and (
                depth_raw.shape[0] != frame.shape[0] or depth_raw.shape[1] != frame.shape[1]
            ):
                depth_map = cv2.resize(
                    depth_raw,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

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
                except Exception as exc:
                    print(f"[WARN] transformed_depth failed: {exc}")

            rgb_o = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_o = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_o)

            if frame_web.ndim == 2:
                frame_web = cv2.cvtColor(frame_web, cv2.COLOR_GRAY2BGR)
            elif frame_web.shape[2] == 4:
                frame_web = cv2.cvtColor(frame_web, cv2.COLOR_BGRA2BGR)

            rgb_w = cv2.cvtColor(frame_web, cv2.COLOR_BGR2RGB)
            mp_w = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_w)

            try:
                # VIDEO mode: each landmarker keeps its own stream state — do not share
                # one instance across Orbbec + webcam or tracking alternates unpredictably.
                mp_ts_o_ms += 1
                result_o = landmarker_o.detect_for_video(mp_o, mp_ts_o_ms)
                mp_ts_l_ms += 1
                result_l = landmarker_l.detect_for_video(mp_w, mp_ts_l_ms)
            except Exception as exc:
                print(f"[WARN] mediapipe detect_for_video failed: {exc}")
                continue

            if (
                fusion_w >= 0.99
                and not args.use_transformed_depth
                and args.hand_3d == ob.HAND_3D_SOURCE_FUSED
                and not warned_fusion_linear_map
            ):
                print(
                    "[INFO] depth-fusion≈1 + --hand-3d fused: linear RGB→depth map; "
                    "try --hand-3d mp or --depth-patch 3–4."
                )
                warned_fusion_linear_map = True

            has_o = bool(result_o.hand_landmarks)
            has_l = bool(result_l.hand_landmarks)

            pts_o = extract_world_points_mm(result_o, 0) if has_o else None
            pts_l = extract_world_points_mm(result_l, 0) if has_l else None

            vis_o = extract_landmark_visibilities(result_o, 0) if has_o else None
            vis_l = extract_landmark_visibilities(result_l, 0) if has_l else None
            summ_o = summarize_mp_visibility(vis_o)
            summ_l = summarize_mp_visibility(vis_l)

            w_geom_o, ev_o = (
                _geom_weight_from_eigen_analysis(pts_o)
                if pts_o is not None
                else (float("nan"), np.full(3, np.nan))
            )
            w_geom_l, ev_l = (
                _geom_weight_from_eigen_analysis(pts_l)
                if pts_l is not None
                else (float("nan"), np.full(3, np.nan))
            )

            mode = "none"
            w_mean_o = 0.0
            w_mean_l = 0.0
            fusion_dbg: Dict[str, Any] = {}
            fused_pts = None

            if (
                has_o
                and has_l
                and pts_o is not None
                and pts_l is not None
                and vis_o is not None
                and vis_l is not None
            ):
                mode = "dual"
                fused_pts, fusion_dbg = fuse_dual_views_weighted(
                    pts_o,
                    pts_l,
                    vis_o,
                    vis_l,
                    w_geom_o,
                    w_geom_l,
                    conf_low=fuse_conf_low,
                    conf_high=fuse_conf_high,
                )
                w_mean_o = float(fusion_dbg["w_mean_o"])
                w_mean_l = float(fusion_dbg["w_mean_l"])
            elif has_o and pts_o is not None:
                mode = "orbbec"
                fused_pts = list(pts_o)
                w_mean_o, w_mean_l = 1.0, 0.0
            elif has_l and pts_l is not None:
                mode = "webcam"
                fused_pts = list(pts_l)
                w_mean_o, w_mean_l = 0.0, 1.0

            if fuse_debug_every > 0 and frame_idx % fuse_debug_every == 0:

                def _fuse_dbg_side(prefix: str, pts, wg: float, ev: np.ndarray) -> str:
                    if pts is None:
                        return f"{prefix}:no_hand"
                    evs = np.round(ev, 2) if np.any(np.isfinite(ev)) else None
                    lam = f"lam={evs}" if evs is not None else "lam=—"
                    ws = f"w={float(wg):.3f}" if np.isfinite(float(wg)) else "w=—"
                    return f"{prefix} {ws} {lam}"

                print(
                    f"[fuse] {_fuse_dbg_side('O', pts_o, w_geom_o, ev_o)} | "
                    f"{_fuse_dbg_side('L', pts_l, w_geom_l, ev_l)} | "
                    f"meanW_O={w_mean_o:.3f} meanW_L={w_mean_l:.3f} "
                    f"hi={fusion_dbg.get('n_high_exclusive', 0)} "
                    f"low<{fuse_conf_low:.2f} high>={fuse_conf_high:.2f}"
                )

            hands_3d = [fused_pts] if fused_pts is not None else []

            frame_disp = frame.copy()
            frame_disp, _kp_o, ema_3d = ob.draw_hand(
                frame_disp,
                result_o,
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
                skip_wrist_labels=True,
            )
            frame_web_vis = frame_web.copy()
            frame_web_vis, _ = draw_hand_webcam(frame_web_vis, result_l, depth_map=None, print_depth=False)

            footer_l: List[Tuple[str, Tuple[int, int, int]]] = []
            if summ_l is not None and has_l:
                col_lm = _conf_color(summ_l["mean"])
                footer_l.append(
                    (
                        f"Webcam MP: mean={summ_l['mean']:.2f} min={summ_l['min']:.2f}",
                        col_lm,
                    )
                )
                wgl = f"{w_geom_l:.2f}" if np.isfinite(float(w_geom_l)) else "--"
                footer_l.append((f"w_geom={wgl}", (200, 200, 200)))
            elif has_l:
                footer_l.append(("Webcam: hand (no vis)", (180, 180, 180)))
            else:
                footer_l.append(("Webcam: no hand", (100, 100, 255)))

            overlay_inset(frame_disp, frame_web_vis, footer_lines=footer_l)

            y_hud = 22
            if has_o and has_l:
                cv2.putText(
                    frame_disp,
                    "Both views: hand detected",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (60, 255, 100),
                    2,
                    cv2.LINE_AA,
                )
            elif has_o or has_l:
                cv2.putText(
                    frame_disp,
                    "Single view (fusion prefers both)",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (80, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_disp,
                    "No hand",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (100, 100, 255),
                    2,
                    cv2.LINE_AA,
                )
            y_hud += 26
            if summ_o is not None and has_o:
                col_o = _conf_color(summ_o["mean"])
                cv2.putText(
                    frame_disp,
                    f"Orbbec MP: mean={summ_o['mean']:.2f} min={summ_o['min']:.2f}",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    col_o,
                    2,
                    cv2.LINE_AA,
                )
                y_hud += 22
                wgo = f"{w_geom_o:.2f}" if np.isfinite(float(w_geom_o)) else "--"
                cv2.putText(
                    frame_disp,
                    f"w_geom={wgo} (shape PCA)",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (210, 210, 210),
                    2,
                    cv2.LINE_AA,
                )
            elif has_o:
                cv2.putText(
                    frame_disp,
                    "Orbbec: hand (no vis)",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_disp,
                    "Orbbec: no hand",
                    (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 100, 255),
                    2,
                    cv2.LINE_AA,
                )

            open_out = None
            if hands_3d and hands_3d[0] is not None:
                tmp = ob.analyze_hand_topology(hands_3d[0])
                if tmp is not None:
                    if open_free_ema is None:
                        open_free_ema = float(tmp["morph_alpha"])
                    else:
                        open_free_ema = (
                            alpha_smooth * float(tmp["morph_alpha"]) + (1.0 - alpha_smooth) * open_free_ema
                        )

                    open_free = float(open_free_ema)
                    if snap_state == "plane":
                        if open_free < ob.PLANE_SNAP_OFF:
                            snap_state = None
                    elif snap_state == "sphere":
                        if open_free > ob.SPHERE_SNAP_OFF:
                            snap_state = None
                    else:
                        if open_free > ob.PLANE_SNAP_ON:
                            snap_state = "plane"
                        elif open_free < ob.SPHERE_SNAP_ON:
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

            if has_o and result_o.hand_landmarks:
                wl = {}
                if open_out is not None:
                    wl[0] = f"open {float(open_out):.2f}"
                else:
                    wl[0] = "open —"
                ob.overlay_wrist_labels(frame_disp, result_o, wl)

            _wgo = f"{w_geom_o:.2f}" if np.isfinite(float(w_geom_o)) else "--"
            _wgl = f"{w_geom_l:.2f}" if np.isfinite(float(w_geom_l)) else "--"
            cv2.putText(
                frame_disp,
                f"fuse:{mode} mO={w_mean_o:.2f} mL={w_mean_l:.2f} wgO={_wgo} wgL={_wgl}",
                (16, frame_disp.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if snap_state is None:
                snap_stable_frames = 0
                if snap_vis_state is not None:
                    snap_hold_frames += 1
                    if snap_hold_frames >= ob.SNAP_HOLD_AFTER_RELEASE_FRAMES:
                        snap_vis_state = None
                        snap_hold_frames = 0
            else:
                snap_hold_frames = 0
                if snap_vis_state == snap_state:
                    snap_stable_frames = min(ob.SNAP_SHOW_AFTER_FRAMES, snap_stable_frames + 1)
                else:
                    snap_stable_frames += 1
                    if snap_stable_frames >= ob.SNAP_SHOW_AFTER_FRAMES:
                        snap_vis_state = snap_state
                        snap_stable_frames = 0

            analyses = None
            if enable_3d and (frame_idx % PLOT_EVERY_N_FRAMES) == 0 and hands_3d:
                analyses = ob.update_3d_plot(
                    ax_hand,
                    ax_topo,
                    hands_3d,
                    morph_alpha_smoothed=open_out,
                    shape_normalized=False,
                    hand_frame=ob.HAND_FRAME_METRIC_MM,
                    hand_3d_source=ob.HAND_3D_SOURCE_MP,
                )
                plt.pause(0.0001)

            if analyses:
                a0 = analyses[0]
                topo_lbl = ob._topology_label_from_alpha(
                    float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"])
                )
                open_disp = open_out if open_out is not None else a0["morph_alpha"]
                free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                if open_remap is not None:
                    lo_r, hi_r = open_remap
                    open_disp = ob._remap_open_display(open_disp, lo_r, hi_r)
                    free_disp = ob._remap_open_display(free_disp, lo_r, hi_r)

                need_refresh = (frame_idx % ob.HUD_UPDATE_EVERY_N_FRAMES) == 0 or hud_cache["open"] is None
                if not need_refresh:
                    if abs(float(open_disp) - float(hud_cache["open"])) > ob.HUD_OPEN_STEP:
                        need_refresh = True
                    if abs(float(free_disp) - float(hud_cache["free"])) > ob.HUD_OPEN_STEP:
                        need_refresh = True
                    if abs(float(a0["planarity"]) - float(hud_cache["plan"])) > ob.HUD_METRIC_STEP:
                        need_refresh = True
                    if abs(float(a0["isotropy"]) - float(hud_cache["iso"])) > ob.HUD_METRIC_STEP:
                        need_refresh = True
                    if abs(float(a0["finger_spread"]) - float(hud_cache["spread"])) > ob.HUD_METRIC_STEP:
                        need_refresh = True

                if need_refresh:
                    hud_cache["open"] = float(open_disp)
                    hud_cache["free"] = float(free_disp)
                    hud_cache["plan"] = float(a0["planarity"])
                    hud_cache["iso"] = float(a0["isotropy"])
                    hud_cache["spread"] = float(a0["finger_spread"])
                    snap_txt = f"  SNAP:{snap_vis_state.upper()}" if snap_vis_state is not None else ""
                    hud_cache["text"] = [
                        f"Topo:{topo_lbl}{snap_txt}  [{mode}]",
                        f"open:{open_disp:.2f}  free:{free_disp:.2f}  mO={w_mean_o:.2f} mL={w_mean_l:.2f}",
                        f"spread:{a0['finger_spread']:.2f}  plan:{a0['planarity']:.2f}  iso:{a0['isotropy']:.2f}",
                    ]

                if frame_idx % 30 == 0:
                    out_v = open_out if open_out is not None else a0["morph_alpha"]
                    free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                    print(
                        f"mode={mode} meanW_O={w_mean_o:.3f} meanW_L={w_mean_l:.3f} "
                        f"topology={a0['topology']} open_out={out_v:.3f} free={free_v:.3f} "
                        f"spread={a0['finger_spread']:.3f} "
                        f"planarity={a0['planarity']:.3f} isotropy={a0['isotropy']:.3f}"
                    )

            cv2.imshow("Hand Tracking Dual (Orbbec + Webcam)", frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                out_name = f"hand_3d_dual_frame_{frame_idx:06d}.png"
                fig.savefig(out_name, dpi=150, bbox_inches="tight")
                print(f"Saved 3D plot: {out_name}")
            if key == ord("p"):
                enable_3d = not enable_3d
                print(f"3D plot enabled: {enable_3d}")
            if key == ord("q"):
                break

            if hud_cache["text"] is not None:
                draw_hud(frame_disp, hud_cache["text"], origin=(16, 16))

            frame_idx += 1
    finally:
        landmarker_o.close()
        landmarker_l.close()
        k4a.stop()
        cap.release()
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
