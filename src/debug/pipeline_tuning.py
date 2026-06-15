"""Debug pipeline tuning: webcam/Orbbec demo parity and 3D plot cadence.

Production defaults live in ``functions.runtime.pipeline_tuning``. Enable via
``--debug-webcam-pipeline`` / ``--debug-3d-plot`` on ``online_control``.
"""

from __future__ import annotations

from functions.runtime.pipeline_tuning import PipelineTuning, online_pipeline_defaults

# When --debug-3d-plot is set without --debug-webcam-pipeline.
ONLINE_DEBUG_PLOT_EVERY_N = 4


def webcam_pipeline_defaults() -> PipelineTuning:
    from functions.display_sim.orbbec_hand import DEPTH_FUSION_WEIGHT
    from functions.mode_switch.webcam_mode_defaults import MODE_DEBOUNCE_FRAMES, PLOT_EVERY_N_FRAMES

    return PipelineTuning(
        depth_fusion_enabled=True,
        depth_fusion_weight=float(DEPTH_FUSION_WEIGHT),
        mode_debounce_frames=int(MODE_DEBOUNCE_FRAMES),
        plot_every_n=int(PLOT_EVERY_N_FRAMES),
    )


def resolve_pipeline_tuning(
    *,
    debug_webcam_pipeline: bool,
    plot_every_cli: int,
    debug_3d_plot: bool,
) -> PipelineTuning:
    """Resolve effective tuning for one online_control run (CLI / debug flags)."""
    base = webcam_pipeline_defaults() if debug_webcam_pipeline else online_pipeline_defaults()
    plot_every = int(plot_every_cli)
    if plot_every <= 0 and debug_3d_plot:
        plot_every = base.plot_every_n if debug_webcam_pipeline else ONLINE_DEBUG_PLOT_EVERY_N
    return PipelineTuning(
        depth_fusion_enabled=base.depth_fusion_enabled,
        depth_fusion_weight=base.depth_fusion_weight,
        mode_debounce_frames=base.mode_debounce_frames,
        plot_every_n=max(0, plot_every),
    )
