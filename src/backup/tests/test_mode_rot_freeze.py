"""Morph mode frozen during obvious left-hand rotation."""

import numpy as np

from functions.mode_switch.modes_runtime import (
    ModeState,
    obvious_left_rotation_for_mode_hold,
    obvious_left_translation_for_mode_hold,
    mode_frozen_for_rotation,
    tick_mode_rotation_freeze_latch,
    consume_mode_rotation_freeze_latch,
    update_mode_state,
)


class _Pose:
    enabled = True
    last_axis_motion = "rotate"
    last_rot_blend_w = 0.55
    last_rv_pose_world = np.array([0.0, 0.0, 0.08])


def _classify_always_4(_pts):
    return 4, 3, {}


def test_obvious_rotation_detected():
    assert obvious_left_rotation_for_mode_hold(_Pose())


def test_mode_held_while_latch_active():
    st = ModeState()
    st.morph_mode = 5
    st.last_mode_raw = 5
    st.mode_freeze_latch = 5
    pose = _Pose()
    pose.last_axis_motion = "none"
    assert mode_frozen_for_rotation(st, pose)
    raw, _ = update_mode_state(
        [(0.0, 0.0, 0.0)] * 21,
        mode_state=st,
        classify_mode_fn=_classify_always_4,
        debounce_frames=1,
        mode_smooth=0.2,
        hold_mode=True,
    )
    assert raw == 5
    assert st.morph_mode == 5


def test_latch_extends_on_rotation():
    st = ModeState()
    st.mode_freeze_latch = 2
    tick_mode_rotation_freeze_latch(st, _Pose(), latch_frames=10)
    assert st.mode_freeze_latch >= 10


class _TranslatePose:
    enabled = True
    last_axis_motion = "translate"
    last_rot_blend_w = 0.0
    last_rv_pose_world = np.array([0.0, 0.0, 1.0])


def test_translate_with_high_rv_does_not_freeze_mode():
    assert not obvious_left_rotation_for_mode_hold(_TranslatePose())


class _LargeTranslatePose:
    enabled = True
    last_axis_motion = "translate"
    last_trans_blend_w = 1.0
    last_delta_h_world = np.array([0.0, 0.05, 0.0])
    last_delta_cam_mm = np.array([0.0, 50.0, 0.0])
    last_delta_cam_arm_mm = np.array([0.0, 0.0, 0.0])


def test_obvious_large_translation_detected():
    assert obvious_left_translation_for_mode_hold(_LargeTranslatePose())


def test_mode_held_during_large_translation():
    st = ModeState()
    st.morph_mode = 5
    st.last_mode_raw = 5
    assert mode_frozen_for_rotation(st, _LargeTranslatePose())
    raw, _ = update_mode_state(
        [(0.0, 0.0, 0.0)] * 21,
        mode_state=st,
        classify_mode_fn=_classify_always_4,
        debounce_frames=1,
        mode_smooth=0.2,
        hold_mode=True,
    )
    assert raw == 5
    assert st.morph_mode == 5


def test_small_translate_drift_does_not_freeze_mode():
    pose = _LargeTranslatePose()
    pose.last_delta_h_world = np.array([0.0, 0.005, 0.0])
    pose.last_delta_cam_mm = np.array([5.0, 3.0, 2.0])
    assert not obvious_left_translation_for_mode_hold(pose)


def test_cumulative_arm_offset_does_not_freeze_mode_when_idle():
    """Hand displaced from arm but not moving this frame — mode must stay switchable."""
    pose = _LargeTranslatePose()
    pose.last_delta_h_world = np.array([0.0, 0.9, 0.0])
    pose.last_delta_cam_mm = np.zeros(3)
    pose.last_delta_cam_arm_mm = np.array([120.0, -80.0, 50.0])
    assert not obvious_left_translation_for_mode_hold(pose)


def test_steady_offset_world_cmd_does_not_freeze_without_frame_motion():
    pose = _LargeTranslatePose()
    pose.last_delta_cam_mm = np.array([2.0, 1.0, 0.0])
    pose.last_delta_h_world = np.array([0.0, 0.9, 0.0])
    assert not obvious_left_translation_for_mode_hold(pose)
