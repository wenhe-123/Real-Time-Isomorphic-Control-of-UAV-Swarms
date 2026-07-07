"""Common utilities: resolve hand-landmarker paths."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_model_path(explicit: Optional[str], current_file: str) -> str:
    """Resolve the MediaPipe hand landmarker model path with local-then-cwd fallback.

    Args:
        explicit: User-provided model path; returned unchanged when set.
        current_file: ``__file__`` of the calling module for sibling lookup.

    Returns:
        Absolute or relative path to ``hand_landmarker.task``.
    """
    if explicit:
        return explicit
    here = Path(current_file).resolve().parent
    for p in (here / "hand_landmarker.task", Path("hand_landmarker.task")):
        if p.is_file():
            return str(p)
    return "hand_landmarker.task"
