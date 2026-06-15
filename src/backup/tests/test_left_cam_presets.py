"""Orthogonality checks for depth-camera → world presets (left-hand swarm)."""

import numpy as np

from functions.swarm_motion.left_hand_swarm_pose import (
    LEFT_CAM_PRESET_ROT,
    make_cam_translation_matrix,
)


def test_each_preset_is_orthogonal_proper_rotation():
    for name, M in LEFT_CAM_PRESET_ROT.items():
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-9), name
        det = float(np.linalg.det(M))
        assert abs(det - 1.0) < 1e-6, (name, det)


def test_translation_matrix_scales_world_z_row():
    M = LEFT_CAM_PRESET_ROT["fwd_y"]
    Mt0 = make_cam_translation_matrix(M, image_y_to_world_z=0.0)
    assert np.allclose(Mt0[2, :], 0.0)
    Mt1 = make_cam_translation_matrix(M, image_y_to_world_z=1.0)
    assert np.allclose(Mt1[2, :], M[2, :])
