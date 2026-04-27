"""Common utilities: resolve hand-landmarker paths, draw semi-transparent HUD text on frames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


def resolve_model_path(explicit: Optional[str], current_file: str) -> str:
    """Resolve hand_landmarker model path with local-then-cwd fallback."""
    if explicit:
        return explicit
    here = Path(current_file).resolve().parent
    for p in (here / "hand_landmarker.task", Path("hand_landmarker.task")):
        if p.is_file():
            return str(p)
    return "hand_landmarker.task"


def draw_hud(frame, lines, origin=(16, 16), line_h=26, pad=8, alpha=0.55):
    """Draw readable HUD with a stable translucent background."""
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

