"""Debug 2D overlays for Orbbec hand landmarks (skeleton, depth labels, wrist tags)."""

from __future__ import annotations

import cv2


def draw_hand_2d_overlay(
    frame,
    *,
    idx: int,
    hand_landmarks,
    points,
    depth_vals,
    norm_depth_label: bool,
    print_depth: bool,
    draw_wrist_label: bool,
    handed_label: str | None,
    hand_connections,
) -> None:
    """Draw 2D skeleton, depth labels, and optional wrist handedness on a BGR frame.

    Args:
        frame: BGR image modified in place.
        idx: Hand index (for debug print prefixes).
        hand_landmarks: MediaPipe normalized landmarks for this hand.
        points: Color pixel ``(x, y)`` list per keypoint.
        depth_vals: Depth in mm per keypoint (parallel to ``points``).
        norm_depth_label: When ``True``, label depths relative to the wrist.
        print_depth: When ``True``, print depth values to stdout.
        draw_wrist_label: When ``True``, draw handedness text at the wrist.
        handed_label: Handedness string (e.g. ``"Left"``), or ``None``.
        hand_connections: Iterable of ``(a, b)`` skeleton edge index pairs.

    Returns:
        None.
    """
    for kp_id, _lm in enumerate(hand_landmarks):
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
                print(f"hand:{idx} kp:{kp_id:02d} x:{x:4d} y:{y:4d} depth_mm:{depth_mm:5d}")

    for a, b in hand_connections:
        p1 = points[a]
        p2 = points[b]
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

    if draw_wrist_label and handed_label:
        cv2.putText(
            frame,
            handed_label,
            points[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
