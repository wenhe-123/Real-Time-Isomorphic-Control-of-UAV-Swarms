"""
Pipeline entry for dual Orbbec + webcam tracking/fusion.

Current step: wraps the existing stable implementation while apps depend on
`pipelines/*` only. Next step is to split this into cli/io/fusion modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from runtime.hand_tracking_dual_orbbec_webcam import main


if __name__ == "__main__":
    main()

