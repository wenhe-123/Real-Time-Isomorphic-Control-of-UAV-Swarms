"""Per-frame deltas vs press-0 arm baseline."""

import numpy as np

from functions.mode_switch.hand_constants import INDEX_MCP_ID, MCP_IDS, MIDDLE_MCP_ID, WRIST_ID, THUMB_MCP_ID
from functions.swarm_motion.left_hand_swarm_pose import (
    LeftSwarmPoseState,
    update_left_swarm_pose,
)


def _synthetic_hand(wrist: np.ndarray) -> np.ndarray:
    h = np.zeros((21, 3), dtype=np.float64)
    w = np.asarray(wrist, dtype=np.float64).reshape(3)
    h[WRIST_ID] = w
    offsets = {
        INDEX_MCP_ID: (55.0, -18.0, 8.0),
        MIDDLE_MCP_ID: (58.0, 0.0, 10.0),
        THUMB_MCP_ID: (-42.0, -38.0, 8.0),
    }
    for idx in MCP_IDS:
        off = offsets.get(int(idx), (50.0, 22.0, 7.0))
        h[int(idx)] = w + np.array(off, dtype=np.float64)
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
        palm_basis="middle_thumb",
        trans_rot_coupling=0.0,
        trans_on_m=0.001,
        rot_on_rad=0.001,
        palm_center_depth_ema=1.0,
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
    assert float(np.linalg.norm(state.last_delta_cam_arm_mm)) > 10.0


def test_arm_reset_clears_ema_not_cumulative_arm_delta():
    state = LeftSwarmPoseState(enabled=True)
    w_arm = np.array([0.0, -170.0, 560.0])
    _step(state, _synthetic_hand(w_arm), force_reset=True)
    w_far = w_arm + np.array([200.0, 0.0, 0.0])
    _step(state, _synthetic_hand(w_far))
    assert float(np.linalg.norm(state.ema_offset)) < 2.0
