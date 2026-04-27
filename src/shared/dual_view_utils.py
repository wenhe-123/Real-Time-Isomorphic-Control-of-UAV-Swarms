"""Dual-view helpers: robust webcam open, per-view skeleton drawing, picture-in-picture (fusion-agnostic UI)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from shared.hand_constants import HAND_CONNECTIONS


def open_webcam_capture(preferred_index: int, width: int, height: int, max_probe_index: int = 8):
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


def overlay_inset(
    dst,
    src,
    margin=12,
    max_w_ratio=0.28,
    *,
    footer_lines: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None,
):
    """Resize src BGR into top-right corner of dst and draw optional footer text."""
    if src is None or src.size == 0:
        return dst
    dh, dw = dst.shape[:2]
    max_w = int(dw * max_w_ratio)
    sh, sw = src.shape[:2]
    scale = min(max_w / float(sw), (dh * 0.35) / float(sh), 1.0)
    nw = max(1, int(sw * scale))
    nh = max(1, int(sh * scale))
    inset = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)

    x1 = dw - margin
    y0 = margin
    x0 = max(0, x1 - nw)
    y1 = min(dh, y0 + nh)
    inset = inset[: y1 - y0, : x1 - x0]
    dst[y0:y1, x0:x1] = inset

    cv2.rectangle(dst, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (40, 200, 255), 1)

    if footer_lines:
        y = y1 - 8
        for text, bgr in reversed(footer_lines):
            cv2.putText(
                dst,
                text,
                (x0 + 6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                bgr,
                1,
                cv2.LINE_AA,
            )
            y -= 16
            if y < y0 + 10:
                break
    return dst

