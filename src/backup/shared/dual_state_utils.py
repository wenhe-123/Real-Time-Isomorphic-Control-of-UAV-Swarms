"""Dual-camera (Orbbec + webcam) runtime: initial mutable state (frame index, EMA, HUD cache, 3D flag)."""

from __future__ import annotations


def init_dual_runtime_state(enable_3d_default: bool):
    """Create initial mutable state bundle for dual runtime loop."""
    return {
        "frame_idx": 0,
        "mp_ts_o_ms": 0,
        "mp_ts_l_ms": 0,
        "warned_fusion_linear_map": False,
        "ema_3d": None,
        "open_free_ema": None,
        "alpha_smooth": 0.18,
        "snap_state": None,
        "hud_cache": {
            "open": None,
            "free": None,
            "plan": None,
            "iso": None,
            "spread": None,
            "curl": None,
            "text": None,
        },
        "snap_vis_state": None,
        "snap_stable_frames": 0,
        "snap_hold_frames": 0,
        "enable_3d": bool(enable_3d_default),
    }

