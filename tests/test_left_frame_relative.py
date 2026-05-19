"""Per-frame deltas vs press-0 arm baseline."""

import numpy as np

from shared.left_hand_swarm_pose import (
    IDX_MIDDLE_MCP,
    IDX_THUMB_MCP,
    IDX_WRIST,
    LeftSwarmPoseState,
    update_left_swarm_pose,
)


def _synthetic_hand(wrist: np.ndarray) -> np.ndarray:
    h = np.zeros((21, 3), dtype=np.float64)
    w = np.asarray(wrist, dtype=np.float64).reshape(3)
    h[IDX_WRIST] = w
    h[IDX_MIDDLE_MCP] = w + np.array([0.0, -60.0, 5.0])
    h[IDX_THUMB_MCP] = w + np.array([-45.0, -40.0, 3.0])
    return h


def _step(state: LeftSwarmPoseState, h: np.ndarray, *, force_reset: bool = False) -> None:
    update_left_swarm_pose(
        h,
        state,
        trans_scale=0.012,
        rot_scale=1.0,
        trans_ema=0.35,
        rot_ema=0.35,
        max_offset_m=1.8,
        max_rot_rad=0.55,
        force_reset=force_reset,
        control_style="axis_locked",
        palm_basis="middle_thumb",
    )


def test_translation_delta_is_frame_not_arm():
    state = LeftSwarmPoseState(enabled=True)
    w0 = np.array([10.0, -160.0, 560.0])
    h0 = _synthetic_hand(w0)
    _step(state, h0, force_reset=True)
    _step(state, h0)
    assert float(np.linalg.norm(state.last_delta_cam_mm)) < 1e-3

    w1 = w0 + np.array([0.0, 0.0, 30.0])
    h1 = _synthetic_hand(w1)
    _step(state, h1)
    assert 12.0 <= float(np.linalg.norm(state.last_delta_cam_mm)) <= 35.0
    assert float(np.linalg.norm(state.last_delta_cam_arm_mm)) > 15.0


def test_arm_reset_clears_ema_not_cumulative_arm_delta():
    state = LeftSwarmPoseState(enabled=True)
    w_arm = np.array([0.0, -170.0, 560.0])
    _step(state, _synthetic_hand(w_arm), force_reset=True)
    w_far = w_arm + np.array([200.0, 0.0, 0.0])
    _step(state, _synthetic_hand(w_far))
    assert float(np.linalg.norm(state.ema_offset)) < 2.0
