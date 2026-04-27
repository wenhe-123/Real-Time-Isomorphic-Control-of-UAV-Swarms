"""
Pipeline entry/bridge for webcam modes.

Exports symbols from the current stable implementation so other modularized
files can depend on `pipelines.*` imports instead of importing long modules
directly. This is an intermediate step before full function-level extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from runtime.hand_tracking_webcam_modes import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

