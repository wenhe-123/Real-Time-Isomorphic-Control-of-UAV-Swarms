"""Fuse Orbbec vs USB-webcam left-hand morph mode (M1..M5) when confidence is low or rotating."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

from functions.mode_switch.mode_gesture_utils import mode_classify_confidence

# Webcam wins when its classify confidence beats Orbbec by at least this margin.
DUAL_MODE_CONF_MARGIN = 0.04
# Orbbec below this and webcam higher → prefer webcam even with a small margin.
DUAL_MODE_ORBBEC_LOW_CONF = 0.75
# Orbbec thumb tip below this → treat depth occlusion; prefer webcam when available.
DUAL_MODE_ORBBEC_THUMB_VIS = 0.78


def should_poll_webcam_for_dual(
    *,
    orbbec_vis_min: float | None,
    rot_vis_thresh: float,
    mode_vis_min: float,
    rotating: bool,
    show_preview: bool,
    dual_mode_assist: bool = False,
    orbbec_thumb_vis: float | None = None,
) -> bool:
    """True when USB webcam should be read: preview, palm rotation, or low Orbbec visibility."""
    _ = dual_mode_assist
    if show_preview:
        return True
    if rotating:
        return True
    if orbbec_thumb_vis is not None and float(orbbec_thumb_vis) < float(DUAL_MODE_ORBBEC_THUMB_VIS):
        return True
    if orbbec_vis_min is not None:
        v = float(orbbec_vis_min)
        if v < float(rot_vis_thresh) or v < float(mode_vis_min):
            return True
    return False


def fuse_dual_mode_raw(
    *,
    mode_orbbec: int,
    mode_webcam: int | None,
    morph_mode: int,
    orbbec_vis_min: float | None,
    mode_vis_min: float,
    rotating: bool,
    conf_orbbec: float = 1.0,
    conf_webcam: float = 0.0,
    orbbec_thumb_vis: float | None = None,
    orbbec_thumb_ok: bool | None = None,
    webcam_thumb_ok: bool | None = None,
) -> int:
    """Choose mode_raw before debounce.

    Webcam is primary when Orbbec visibility is low, when rotating with M5 guard,
    when USB sees M5 / thumb-up while Orbbec is stuck at M4, or when confidence
    clearly favors the USB view (partial occlusion).
    """
    if mode_webcam is None:
        return int(mode_orbbec)

    mo = int(mode_orbbec)
    mw = int(mode_webcam)

    # Strong M5 assist: USB authority for thumb when Orbbec reads tier 4.
    if mo == 4 and mw == 5:
        return 5
    if mo == 4 and webcam_thumb_ok and not orbbec_thumb_ok:
        return 5
    if int(morph_mode) >= 5 and mw >= 5 and mo == 4:
        return 5

    low_vis = orbbec_vis_min is not None and float(orbbec_vis_min) < float(mode_vis_min)
    if low_vis:
        return mw

    thumb_occluded = (
        orbbec_thumb_vis is not None
        and float(orbbec_thumb_vis) < float(DUAL_MODE_ORBBEC_THUMB_VIS)
    )
    if thumb_occluded and mo in (4, 5):
        return mw

    if rotating and int(morph_mode) >= 5 and mo < 5 and mw >= 5:
        return 5

    co = float(conf_orbbec)
    cw = float(conf_webcam)
    if thumb_occluded:
        co *= 0.18
    elif (
        orbbec_thumb_vis is not None
        and float(orbbec_thumb_vis) < 0.85
        and mo in (4, 5)
    ):
        tv = float(orbbec_thumb_vis)
        co *= 0.15 + 0.85 * (tv / 0.85) ** 2.4
    if mo == 4 and not orbbec_thumb_ok:
        co *= 0.32
    if mo == 4 and mw >= 5:
        co *= 0.45
    if mo in (4, 5) and mw in (4, 5) and mo != mw:
        co *= 0.55

    if mo != mw:
        if cw > co + DUAL_MODE_CONF_MARGIN:
            return mw
        if co < DUAL_MODE_ORBBEC_LOW_CONF and cw > co:
            return mw
        if cw >= co:
            return mw
        # Orbbec M4 with weak trust: USB wins on a modest lead.
        if mo == 4 and co < 0.55 and cw + 0.04 >= co:
            return mw

    return mo


def classify_mode_dual(
    pts_orbbec: Sequence | None,
    pts_webcam: Sequence | None,
    *,
    morph_mode: int,
    orbbec_vis_min: float | None,
    mode_vis_min: float,
    rotating: bool,
    classify_mode_fn: Callable,
    classify_webcam_fn: Callable | None = None,
    orbbec_thumb_vis: float | None = None,
) -> Tuple[int, int]:
    """Classify mode from Orbbec 3D + optional USB 2D image plane, with dual fusion."""
    w_fn = classify_webcam_fn or classify_mode_fn
    tier_count = -1
    mode_o = int(morph_mode)
    conf_o = 0.0
    dbg_o: dict = {}
    o_thumb_ok: bool | None = None
    if pts_orbbec is not None:
        mode_o, tier_count, dbg_o = classify_mode_fn(pts_orbbec)
        conf_o = mode_classify_confidence(
            mode_o,
            dbg_o,
            thumb_tip_vis=orbbec_thumb_vis,
            hand_vis_min=orbbec_vis_min,
        )
        o_thumb_ok = dbg_o.get("thumb_ok")

    mode_w: int | None = None
    conf_w = 0.0
    w_thumb_ok: bool | None = None
    if pts_webcam is not None:
        mode_w, tier_w, dbg_w = w_fn(pts_webcam)
        conf_w = mode_classify_confidence(mode_w, dbg_w)
        w_thumb_ok = dbg_w.get("thumb_ok")
        if tier_count < 0:
            tier_count = tier_w

    fused = fuse_dual_mode_raw(
        mode_orbbec=mode_o,
        mode_webcam=mode_w,
        morph_mode=int(morph_mode),
        orbbec_vis_min=orbbec_vis_min,
        mode_vis_min=float(mode_vis_min),
        rotating=bool(rotating),
        conf_orbbec=conf_o,
        conf_webcam=conf_w,
        orbbec_thumb_vis=orbbec_thumb_vis,
        orbbec_thumb_ok=o_thumb_ok,
        webcam_thumb_ok=w_thumb_ok,
    )
    return int(fused), int(tier_count)
