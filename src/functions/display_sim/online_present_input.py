"""Shared per-frame inputs for sim vs real online presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from functions.mode_switch.online_frame_gesture import GestureFrameResult
from functions.runtime.online_boot import OnlineBoot
from functions.runtime.online_runtime_config import OnlineRuntimeConfig
from functions.swarm_motion.online_frame_filter import TargetFilterResult


@dataclass
class PresentFrameInput:
    frame: np.ndarray
    boot: OnlineBoot
    cfg: OnlineRuntimeConfig
    gest: GestureFrameResult
    filt: TargetFilterResult
    raw_target: np.ndarray
    left_pose_dbg: str
    elapsed: float
    just_gesture_armed: bool = False
    just_prearm_phase: bool = False
    section: Callable[[str], None] | None = None
