"""Pipeline entry for webcam modes (no backup dependency)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backup.runtime.hand_tracking_webcam_modes import main


if __name__ == "__main__":
    main()

