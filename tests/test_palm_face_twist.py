"""Palm–optical alignment and in-plane twist angle convention."""

import numpy as np

from shared.left_hand_swarm_pose import palm_optical_alignment_cos, translation_plane_dominates_depth


def test_palm_optical_alignment_cos_facing_camera():
    B = np.eye(3, dtype=np.float64)
    assert palm_optical_alignment_cos(B) == 1.0


def test_palm_optical_alignment_cos_edge_on():
    B = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    assert palm_optical_alignment_cos(B) == 0.0


def test_atan2_matches_in_plane_twist_angle():
    th = 0.37
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    assert np.arctan2(R[1, 0], R[0, 0]) == th


def test_planar_dom_triggers_for_mostly_image_plane_wrist_delta():
    """In-plane twist moves the wrist mostly in dx,dy; old behavior zeroed all rotation that frame."""
    d = np.array([12.0, 10.0, 0.5], dtype=np.float64)
    assert translation_plane_dominates_depth(d, planar_ratio=1.0, planar_min_mm=2.0)
