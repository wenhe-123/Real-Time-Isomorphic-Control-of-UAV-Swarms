"""Standalone Orbbec hand-tracking demo (backup). Active swarm entry: ``online_control_dual.py``."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from hand_tracking_orbbec_standalone import main

if __name__ == "__main__":
    main()
