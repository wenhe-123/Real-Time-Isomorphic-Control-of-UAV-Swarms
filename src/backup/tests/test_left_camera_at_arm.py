"""Press-0 frozen camera→sim map (camera_at_arm world frame)."""

import numpy as np

from functions.swarm_motion.left_hand_swarm_pose import (
    LeftSwarmPoseState,
    build_sim_from_cam_matrices,
    left_cam_preset_rotation,
    palm_orthonormal_basis,
    update_left_swarm_pose,
)


def _synthetic_hand(wrist: np.ndarray) -> np.ndarray:
    from functions.mode_switch.hand_constants import INDEX_MCP_ID, MCP_IDS, MIDDLE_MCP_ID, WRIST_ID, THUMB_MCP_ID

    h = np.zeros((21, 3), dtype=np.float64)
    w = np.asarray(wrist, dtype=np.float64).reshape(3)
    h[WRIST_ID] = w
    offsets = {
        INDEX_MCP_ID: (40.0, 0.0, 0.0),
        MIDDLE_MCP_ID: (42.0, -5.0, 2.0),
        THUMB_MCP_ID: (-30.0, -25.0, 1.0),
    }
    for idx in MCP_IDS:
        off = offsets.get(int(idx), (35.0, 30.0, 0.0))
        h[int(idx)] = w + np.array(off, dtype=np.float64)
    return h


def test_build_sim_from_cam_camera_preset():
    M, Mt = build_sim_from_cam_matrices("camera", image_y_to_world_z=1.0)
    np.testing.assert_allclose(M, np.eye(3))
    v = np.array([10.0, -5.0, 20.0])
    np.testing.assert_allclose(M @ v, v)
    np.testing.assert_allclose(Mt @ v, v)


def test_frozen_map_used_after_arm():
    wrist0 = np.array([100.0, 200.0, 800.0])
    h0 = _synthetic_hand(wrist0)
    M, Mt = build_sim_from_cam_matrices("camera", image_y_to_world_z=1.0)
    state = LeftSwarmPoseState(enabled=True)
    state.reset_to_current(
        h0,
        sim_from_cam=M,
        sim_trans_from_cam=Mt,
        cam_preset_label="camera",
    )
    h1 = _synthetic_hand(wrist0 + np.array([50.0, 0.0, 0.0]))
    off, _ = update_left_swarm_pose(
        h1,
        state,
        trans_scale=0.01,
        rot_scale=0.0,
        trans_ema=1.0,
        rot_ema=0.0,
        max_offset_m=10.0,
        max_rot_rad=10.0,
        cam_delta_to_world=left_cam_preset_rotation("fwd_y"),
        cam_translation_to_world=left_cam_preset_rotation("fwd_y"),
        trans_rot_coupling=0.0,
        trans_on_m=0.001,
        rot_on_rad=0.001,
        palm_center_depth_ema=1.0,
    )
    # 50 mm cam +X → 0.5 m sim +X (identity map at arm)
    np.testing.assert_allclose(off, [0.5, 0.0, 0.0], atol=1e-6)


def test_frozen_fwd_y_in_out_on_sim_y():
    wrist0 = np.array([0.0, 0.0, 800.0])
    h0 = _synthetic_hand(wrist0)
    M, Mt = build_sim_from_cam_matrices("fwd_y", image_y_to_world_z=1.0)
    state = LeftSwarmPoseState(enabled=True)
    state.reset_to_current(h0, sim_from_cam=M, sim_trans_from_cam=Mt, cam_preset_label="fwd_y")
    h1 = _synthetic_hand(wrist0 + np.array([0.0, 0.0, 40.0]))
    off, _ = update_left_swarm_pose(
        h1,
        state,
        trans_scale=0.01,
        trans_ema=1.0,
        rot_scale=0.5,
        rot_ema=1.0,
        max_offset_m=10.0,
        max_rot_rad=10.0,
        axis_sign=(1.0, -1.0, 1.0),
        trans_rot_coupling=0.0,
        trans_on_m=0.001,
        rot_on_rad=0.001,
        palm_center_depth_ema=1.0,
    )
    # +40 mm cam Z → sim Y; axis_sign[1]=-1 → negative sim Y
    np.testing.assert_allclose(off, [0.0, -0.4, 0.0], atol=1e-5)
