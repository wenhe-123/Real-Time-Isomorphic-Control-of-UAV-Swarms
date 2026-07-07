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
    """Per-frame bundle passed to sim and real online presentation paths.

    Attributes:
        frame: BGR color frame for HUD / preview.
        boot: Shared online boot state (executors, pose, mode boxes).
        cfg: Runtime configuration (plot cadence, flags).
        gest: Gesture classification result for this frame.
        filt: Filtered morph target and mode from ``TargetFilterResult``.
        raw_target: Unfiltered morph target in sim meters.
        left_pose_dbg: Debug string for left-hand pose overlay.
        elapsed: Monotonic seconds since session start.
        just_gesture_armed: ``True`` on the frame gesture arming toggled on.
        just_prearm_phase: ``True`` on prearm phase transition frames.
        section: Optional profiler callback ``section(name)``.
    """

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
