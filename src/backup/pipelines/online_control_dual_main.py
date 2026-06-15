"""Pipeline entry: Orbbec online control with dual webcam rotation fallback."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from online_control_dual import main


if __name__ == "__main__":
    main()
