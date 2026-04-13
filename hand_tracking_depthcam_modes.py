"""
Launcher: run from `iso_swarm/` as `python hand_tracking_depthcam_modes.py`.
Implementation: `src/hand_tracking_depthcam_modes.py`.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_script = _src / "hand_tracking_depthcam_modes.py"
runpy.run_path(str(_script), run_name="__main__")
