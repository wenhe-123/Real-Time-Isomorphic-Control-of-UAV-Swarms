"""Palm middle_thumb axes aligned to world X/Y/Z via matrix + perm."""

import numpy as np

from functions.mode_switch.hand_constants import (
    INDEX_MCP_ID,
    MIDDLE_MCP_ID,
    MIDDLE_TIP_ID,
    RING_MCP_ID,
    THUMB_MCP_ID,
    THUMB_TIP_ID,
    WRIST_ID,
)
from backup.tests.left_hand_swarm_pose_test_api import (
    classify_pose_small_translate_vs_rotate,
    palm_center_translation_components,
    palm_components_in_camera_mm,
    palm_pose_change_rad,
    palm_vector_palm_to_world,
)
from backup.tests.left_hand_swarm_pose_legacy_gates import (
    axis_locked_gated_palm_components,
    cam_delta_palm_to_world_m,
    forward_palm_component_from_scale_only,
    palm_components_to_world_m,
    palm_translation_components_mm,
    scale_forward_triggered,
)
from functions.swarm_motion.left_hand_swarm_pose import (
    LEFT_CAM_PRESET_ROT,
    axis_locked_trans_rot_blend_weights,
    palm_center_mm,
    palm_orthonormal_basis_middle_y_thumb_x,
    palm_world_rotvec_from_basis_delta,
)


def test_palm_basis_is_orthonormal_right_handed():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([4.0, -58.0, 6.0])
    h[MIDDLE_TIP_ID] = wrist + np.array([5.0, -98.0, 10.0])
    h[THUMB_MCP_ID] = wrist + np.array([-44.0, -42.0, 4.0])
    h[THUMB_TIP_ID] = wrist + np.array([-62.0, -36.0, 6.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    assert abs(float(np.linalg.det(B)) - 1.0) < 0.02
    assert float(np.dot(B[:, 2], np.cross(B[:, 0], B[:, 1]))) > 0.99


def test_palm_center_is_not_wrist_only():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([0.0, -60.0, 5.0])
    h[THUMB_MCP_ID] = wrist + np.array([-45.0, -40.0, 3.0])
    pc = palm_center_mm(h)
    assert pc is not None
    assert float(np.linalg.norm(pc - wrist)) > 1.0


def test_scale_exclusive_yields_forward_only():
    fwd = forward_palm_component_from_scale_only(12.0, span_ref_mm=80.0, min_mm=8.0, min_rel=0.07)
    assert fwd is not None
    assert abs(float(fwd[0])) < 1e-6 and abs(float(fwd[1])) < 1e-6
    assert float(fwd[2]) > 8.0
    assert not scale_forward_triggered(3.0, span_ref_mm=80.0, min_mm=8.0, min_rel=0.07)


def test_lateral_pan_not_forward_only():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([0.0, -60.0, 5.0])
    h[THUMB_MCP_ID] = wrist + np.array([-45.0, -40.0, 3.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    gated = axis_locked_gated_palm_components(np.array([14.0, 3.0, 0.0]), B)
    assert abs(float(gated[0])) > 10.0
    assert abs(float(gated[2])) < 1e-5


def test_gated_palm_forward_maps_world_y():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([0.0, -60.0, 5.0])
    h[THUMB_MCP_ID] = wrist + np.array([-45.0, -40.0, 3.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    fwd = forward_palm_component_from_scale_only(10.0, span_ref_mm=80.0, min_mm=7.0, min_rel=0.07)
    dw = palm_components_to_world_m(fwd, trans_scale=0.001, axis_sign=(1.0, 1.0, -1.0))
    assert abs(float(dw[1])) > abs(float(dw[0]))
    assert abs(float(dw[1])) > abs(float(dw[2]))


def test_fingertip_motion_maps_world_up():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([0.0, -60.0, 5.0])
    h[MIDDLE_TIP_ID] = wrist + np.array([0.0, -100.0, 8.0])
    h[THUMB_MCP_ID] = wrist + np.array([-45.0, -40.0, 3.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    tip_dir = h[MIDDLE_TIP_ID] - wrist
    tip_dir = tip_dir / max(float(np.linalg.norm(tip_dir)), 1e-9)
    comp = palm_translation_components_mm(tip_dir * 40.0, B)
    world = palm_components_to_world_m(comp, trans_scale=0.012, axis_sign=(1.0, 1.0, 1.0))
    assert float(world[2]) > 0.35
    assert abs(float(world[0])) < 0.05
    assert abs(float(world[1])) < 0.05


def test_middle_maps_to_world_up():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([0.0, -60.0, 5.0])
    h[MIDDLE_TIP_ID] = wrist + np.array([0.0, -100.0, 8.0])
    h[THUMB_MCP_ID] = wrist + np.array([-45.0, -40.0, 3.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    mid = h[MIDDLE_TIP_ID] - wrist
    mid = mid / max(float(np.linalg.norm(mid)), 1e-9)
    assert abs(float(np.dot(B[:, 1], mid))) > 0.85
    comp = palm_components_in_camera_mm(B[:, 1], B)
    w_up = palm_vector_palm_to_world(comp)
    assert float(w_up[2]) > 0.85


def test_trans_rot_blend_keeps_secondary():
    mot, wr, wt = axis_locked_trans_rot_blend_weights(
        np.array([0.02, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.005]),
        trans_on_m=0.009,
        rot_on_rad=0.008,
        secondary_frac=0.3,
    )
    assert mot == "translate"
    assert wt == 1.0
    assert 0.0 < wr < 0.35
    mot2, wr2, wt2 = axis_locked_trans_rot_blend_weights(
        np.array([0.003, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.025]),
        trans_on_m=0.006,
        rot_on_rad=0.010,
        rv_cam_rad=np.array([0.0, 0.0, 0.025]),
        delta_cam_mm=np.array([3.0, 0.0, 0.0]),
        secondary_frac=0.3,
    )
    assert mot2 == "rotate"
    assert wr2 == 1.0
    assert 0.0 < wt2 < 0.35


def test_pose_small_vs_large_classify():
    B0 = np.eye(3)
    th = 0.12
    c, s = np.cos(th), np.sin(th)
    B1 = B0 @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    M = LEFT_CAM_PRESET_ROT["fwd_y"]
    rv = palm_world_rotvec_from_basis_delta(M, B1, B0)
    assert palm_pose_change_rad(B1, B0, Mc_rot=M) > 0.08
    assert (
        classify_pose_small_translate_vs_rotate(
            rv, np.array([0.002, 0.001, 0.002]), pose_rotate_rad=0.008, trans_on_m=0.009
        )
        == "rotate"
    )
    assert (
        classify_pose_small_translate_vs_rotate(
            np.zeros(3), np.array([0.0, 0.0, 0.02]), pose_rotate_rad=0.008, trans_on_m=0.009
        )
        == "translate"
    )


def test_center_translation_picks_components():
    wrist = np.array([0.0, 0.0, 500.0])
    h = np.zeros((21, 3), dtype=np.float64)
    h[WRIST_ID] = wrist
    h[MIDDLE_MCP_ID] = wrist + np.array([0.0, -60.0, 5.0])
    h[THUMB_MCP_ID] = wrist + np.array([-45.0, -40.0, 3.0])
    _, B = palm_orthonormal_basis_middle_y_thumb_x(h)
    delta_cam = np.asarray(B[:, 0], dtype=np.float64) * 12.0
    comp = palm_center_translation_components(delta_cam, B)
    assert abs(float(comp[0])) > 10.0 and abs(float(comp[2])) < 1e-5


def test_matrix_rot_uses_frozen_M():
    B0 = np.eye(3)
    th = 0.09
    c, s = np.cos(th), np.sin(th)
    B1 = B0 @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    M = LEFT_CAM_PRESET_ROT["fwd_y"]
    rv = palm_world_rotvec_from_basis_delta(M, B1, B0)
    assert float(np.linalg.norm(rv)) > 0.05
