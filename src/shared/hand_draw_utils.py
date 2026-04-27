"""2D hand visualization: draw single/dual-hand skeletons, keypoints, and optional labels on BGR frames."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from shared.hand_constants import HAND_CONNECTIONS
from shared.mp_hand_utils import hand_label


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
    keypoints_3d = []
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
        label = hand_label(result, hand_idx) + label_suffix
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
    Wrist labels: left hand shows M{mode}; right hand shows open {value}.
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
        name = hand_label(result, idx).lower()
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
                label_override = "open -"
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

