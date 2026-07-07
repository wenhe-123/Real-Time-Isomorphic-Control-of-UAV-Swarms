"""Production pipeline tuning (depth fusion, debounce, plot cadence)."""

from __future__ import annotations

from dataclasses import dataclass

# --- pipeline defaults ---
@dataclass(frozen=True)
class PipelineTuning:
    """Depth fusion, mode debounce, and 3D plot cadence for the online pipeline."""

    depth_fusion_enabled: bool
    depth_fusion_weight: float
    mode_debounce_frames: int
    plot_every_n: int


ONLINE_DEPTH_FUSION_ENABLED = False
ONLINE_MODE_DEBOUNCE_FRAMES = 1
ONLINE_PLOT_EVERY_N = 0


def online_pipeline_defaults() -> PipelineTuning:
    """Return production pipeline tuning constants for online control.

    Returns:
        Frozen tuning bundle with depth fusion off, minimal mode debounce, and
        plot cadence from module-level ``ONLINE_*`` constants.
    """
    return PipelineTuning(
        depth_fusion_enabled=ONLINE_DEPTH_FUSION_ENABLED,
        depth_fusion_weight=0.0,
        mode_debounce_frames=ONLINE_MODE_DEBOUNCE_FRAMES,
        plot_every_n=ONLINE_PLOT_EVERY_N,
    )
