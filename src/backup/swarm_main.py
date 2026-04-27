"""
Pipeline entry for Orbbec-driven swarm control.

Current step: wraps the existing stable implementation while apps depend on
`pipelines/*` only. Next step is to split control mapping / io / runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from runtime.hand_swarm_control_orbbec import main


if __name__ == "__main__":
    main()

