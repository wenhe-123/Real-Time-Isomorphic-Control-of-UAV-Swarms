"""Dual Orbbec + webcam morph mode fusion."""

from functions.mode_switch.dual_mode_fusion import (
    classify_mode_dual,
    fuse_dual_mode_raw,
    should_poll_webcam_for_dual,
)
from functions.mode_switch.mode_gesture_utils import mode_classify_confidence


def _classify_pts(pts):
    if pts is None:
        return 1, 0, {}
    return int(pts[0][0]), 1, {}


def test_poll_when_rotating_or_low_vis():
    assert should_poll_webcam_for_dual(
        orbbec_vis_min=0.8,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=True,
        show_preview=False,
    )
    assert should_poll_webcam_for_dual(
        orbbec_vis_min=0.3,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=False,
        show_preview=False,
    )
    assert not should_poll_webcam_for_dual(
        orbbec_vis_min=0.9,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=False,
        show_preview=False,
    )


def test_poll_only_when_rotating_or_low_vis():
    assert should_poll_webcam_for_dual(
        orbbec_vis_min=0.9,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=True,
        show_preview=False,
    )
    assert should_poll_webcam_for_dual(
        orbbec_vis_min=0.3,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=False,
        show_preview=False,
        dual_mode_assist=True,
    )
    assert not should_poll_webcam_for_dual(
        orbbec_vis_min=0.9,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=False,
        show_preview=False,
        dual_mode_assist=True,
    )


def test_m5_when_orbbec_m4_webcam_thumb_ok():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=4,
            mode_webcam=4,
            morph_mode=4,
            orbbec_vis_min=0.85,
            mode_vis_min=0.55,
            rotating=False,
            conf_orbbec=0.88,
            conf_webcam=0.75,
            orbbec_thumb_ok=False,
            webcam_thumb_ok=True,
        )
        == 5
    )


def test_m5_guard_when_rotating():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=4,
            mode_webcam=5,
            morph_mode=5,
            orbbec_vis_min=0.8,
            mode_vis_min=0.55,
            rotating=True,
        )
        == 5
    )


def test_webcam_primary_when_low_vis():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=2,
            mode_webcam=5,
            morph_mode=2,
            orbbec_vis_min=0.2,
            mode_vis_min=0.55,
            rotating=False,
        )
        == 5
    )


def test_webcam_wins_on_higher_confidence_when_vis_ok():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=4,
            mode_webcam=5,
            morph_mode=4,
            orbbec_vis_min=0.8,
            mode_vis_min=0.55,
            rotating=False,
            conf_orbbec=0.35,
            conf_webcam=0.82,
        )
        == 5
    )


def test_orbbec_kept_when_confidence_clearly_higher():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=3,
            mode_webcam=5,
            morph_mode=3,
            orbbec_vis_min=0.8,
            mode_vis_min=0.55,
            rotating=False,
            conf_orbbec=0.90,
            conf_webcam=0.40,
        )
        == 3
    )


def test_mode_classify_confidence_penalizes_m4_with_missing_thumb():
    dbg_m4_occluded = {
        "reason": "ok",
        "d_norm": [1.0, 1.0, 1.0, 1.0, 0.1],
        "max": 1.0,
        "gap": 0.38,
        "thumb_ok": False,
    }
    dbg_m5 = {
        "reason": "ok",
        "d_norm": [1.0, 1.0, 1.0, 1.0, 0.85],
        "max": 1.0,
        "gap": 0.38,
        "thumb_ok": True,
    }
    assert mode_classify_confidence(4, dbg_m4_occluded) < mode_classify_confidence(5, dbg_m5)
    # Low MP thumb visibility should crush Orbbec M4 trust even when d_norm gap looks fine.
    conf_vis = mode_classify_confidence(
        4, dbg_m4_occluded, thumb_tip_vis=0.45, hand_vis_min=0.80
    )
    assert conf_vis < 0.22


def test_poll_when_thumb_occluded():
    assert should_poll_webcam_for_dual(
        orbbec_vis_min=0.9,
        rot_vis_thresh=0.42,
        mode_vis_min=0.55,
        rotating=False,
        show_preview=False,
        orbbec_thumb_vis=0.55,
    )


def test_orbbec_m4_low_thumb_vis_loses_to_webcam_on_confidence():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=4,
            mode_webcam=5,
            morph_mode=4,
            orbbec_vis_min=0.80,
            mode_vis_min=0.55,
            rotating=False,
            conf_orbbec=0.62,
            conf_webcam=0.70,
            orbbec_thumb_vis=0.55,
            orbbec_thumb_ok=False,
        )
        == 5
    )


def test_classify_mode_dual_rotating_m5():
    pts4 = [(4.0, 0.0, 0.0)] * 21
    pts5 = [(5.0, 0.0, 0.0)] * 21
    mode, _ = classify_mode_dual(
        pts4,
        pts5,
        morph_mode=5,
        orbbec_vis_min=0.8,
        mode_vis_min=0.55,
        rotating=True,
        classify_mode_fn=_classify_pts,
    )
    assert mode == 5


def _classify_occluded_orbbec(_pts):
    return 4, 4, {
        "reason": "ok",
        "d_norm": [1.0, 1.0, 1.0, 1.0, 0.05],
        "max": 1.0,
        "gap": 0.38,
        "thumb_ok": False,
    }


def _classify_clear_webcam(_pts):
    return 5, 5, {
        "reason": "ok",
        "d_norm": [1.0, 1.0, 1.0, 1.0, 0.88],
        "max": 1.0,
        "gap": 0.38,
        "thumb_ok": True,
    }


def test_webcam_wins_when_orbbec_thumb_occluded():
    assert (
        fuse_dual_mode_raw(
            mode_orbbec=4,
            mode_webcam=5,
            morph_mode=4,
            orbbec_vis_min=0.85,
            mode_vis_min=0.55,
            rotating=False,
            conf_orbbec=0.88,
            conf_webcam=0.75,
            orbbec_thumb_vis=0.25,
        )
        == 5
    )


def test_resolve_webcam_left_hand_index_single_hand():
    from functions.dual_cam.mp_hand_utils import resolve_webcam_left_hand_index

    class _Cat:
        def __init__(self, name):
            self.category_name = name

    class _Res:
        hand_landmarks = [object()]
        handedness = [[_Cat("Right")]]

    assert resolve_webcam_left_hand_index(_Res()) == 0
    assert resolve_webcam_left_hand_index(_Res(), prefer_hand_idx=0) == 0


def test_classify_mode_dual_thumb_occlusion_prefers_webcam():
    pts_o = [(0.0, 0.0, 0.0)] * 21
    pts_w = [(0.0, 0.0, 0.0)] * 21

    def classify_fn(pts_in):
        if pts_in is pts_o:
            return _classify_occluded_orbbec(pts_in)
        return _classify_clear_webcam(pts_in)

    mode, _ = classify_mode_dual(
        pts_o,
        pts_w,
        morph_mode=4,
        orbbec_vis_min=0.85,
        mode_vis_min=0.55,
        rotating=False,
        classify_mode_fn=classify_fn,
        orbbec_thumb_vis=0.35,
    )
    assert mode == 5
