"""Per-drone vs centroid rigid target application."""

import numpy as np

from shared.left_hand_swarm_pose import apply_rigid_to_targets, rotvec_to_R


def test_per_drone_pivot_rotates_morph_delta_about_arm_slot():
    ref = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    # Morph moved both drones +Y relative to arm-time ref (not a rigid orbit about centroid).
    targets = np.array([[1.0, 0.5, 0.0], [2.0, 0.5, 0.0]], dtype=np.float32)
    R = rotvec_to_R(np.array([0.0, 0.0, np.pi / 2.0]))
    out = apply_rigid_to_targets(targets, np.zeros(3), R, ref_drone_xyz=ref, pivot="per_drone")
    np.testing.assert_allclose(out[0, :3], [0.5, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out[1, :3], [1.5, 0.0, 0.0], atol=1e-5)


def test_centroid_pivot_orbits_common_center():
    targets = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    ref = targets.copy()
    R = rotvec_to_R(np.array([0.0, 0.0, np.pi / 2.0]))
    out = apply_rigid_to_targets(targets, np.zeros(3), R, ref_drone_xyz=ref, pivot="centroid")
    np.testing.assert_allclose(out[0, :3], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out[1, :3], [0.0, -1.0, 0.0], atol=1e-5)
