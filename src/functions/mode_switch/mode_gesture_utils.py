"""Left-hand mode gestures: palm scale and finger-extension tiers → morph modes (webcam mode selection)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Tier 4 → 5: thumb must pass these (thumb is noisier than other tips; avoid false M5 with 4 fingers up).
# Slightly relaxed vs 2025 defaults so “five fingers open” reaches M5 more reliably on real depth/MP noise.
THUMB_PROMOTE_ABS_MIN = 0.60
THUMB_PROMOTE_REL_MX4 = 0.66
# Below this vs max of index..pinky, do not count as intentional thumb-up.
THUMB_PROMOTE_MAX_BELOW_MX4 = 0.36


def palm_center_and_scale(hand_points: Sequence[Tuple[float, float, float]], wrist_id: int, mcp_ids: Sequence[int]):
    palm_ids = [wrist_id] + list(mcp_ids)
    palm_pts = np.array(
        [hand_points[i] for i in palm_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if palm_pts.shape[0] == 0:
        return None, 1.0
    palm_center = palm_pts.mean(axis=0)
    wrist = np.array(hand_points[wrist_id], dtype=float)
    scale = float(np.mean(np.linalg.norm(palm_pts - wrist, axis=0))) + 1e-6
    return palm_center, scale


def classify_mode_from_fingers(
    hand_points: Sequence[Tuple[float, float, float]],
    *,
    mode_count_tip_ids: Sequence[int],
    mode_extend_min: float,
    mode_tier_gap: float,
    wrist_id: int,
    mcp_ids: Sequence[int],
    thumb_promote_abs_min: float = THUMB_PROMOTE_ABS_MIN,
    thumb_promote_rel_mx4: float = THUMB_PROMOTE_REL_MX4,
    thumb_promote_max_below_mx4: float = THUMB_PROMOTE_MAX_BELOW_MX4,
):
    """
    Extension tiers → mode. Index/middle/ring/pinky define tiers 1–4; thumb upgrades to 5 only
    when four fingers already read as tier 4 and thumb meets strict thresholds (reduces M4/M5 confusion).
    """
    pc, scale = palm_center_and_scale(hand_points, wrist_id, mcp_ids)
    if pc is None:
        return 1, 0, {"d_norm": [], "reason": "no_palm"}

    dists = []
    for tid in mode_count_tip_ids:
        if tid >= len(hand_points) or np.isnan(hand_points[tid][2]):
            dists.append(0.0)
        else:
            p = np.array(hand_points[tid], dtype=float)
            dists.append(float(np.linalg.norm(p - pc)))
    dn = np.array(dists, dtype=float) / scale
    n_tip = len(mode_count_tip_ids)

    if dn.size == 0 or float(np.max(dn)) < float(mode_extend_min):
        return 1, 0, {"d_norm": dn.tolist(), "reason": "fist_or_low"}

    have_thumb = n_tip >= 5

    # --- Tiers 1–4 from index/middle/ring/pinky only (thumb must not inflate the count).
    dn4 = dn[:4] if have_thumb else dn
    mx4 = float(np.max(dn4))
    gap = max(float(mode_tier_gap), 0.08 * mx4, 0.34 * mx4)
    tier = int(np.sum(dn4 >= mx4 - gap))
    tier = max(1, min(4, tier))

    if n_tip >= 4:
        if (
            tier == 3
            and len(dn4) >= 4
        ):
            pk = float(dn4[3])
            if pk >= float(mode_extend_min) * 0.72 and (mx4 - pk) <= 0.50:
                tier = 4

    if not have_thumb or tier < 4:
        tier = max(1, min(n_tip if not have_thumb else 4, tier))
        return tier, tier, {"d_norm": dn.tolist(), "max": mx4, "gap": gap, "reason": "ok", "tier_base": tier}

    # --- Tier 5: only if four fingers already look like tier 4, then require a deliberate thumb extension.
    th = float(dn[4])
    thumb_ok = th >= max(
        float(thumb_promote_abs_min),
        float(thumb_promote_rel_mx4) * mx4,
        mx4 - float(thumb_promote_max_below_mx4),
    )
    if tier == 4 and thumb_ok:
        tier = 5
    elif (
        tier == 4
        and have_thumb
        and th >= float(mode_extend_min) * 0.98
        and th >= 0.58 * mx4
    ):
        # Secondary: five-finger spread with thumb clearly extended but just below strict promote.
        tier = 5

    tier = max(1, min(n_tip, tier))
    return tier, tier, {
        "d_norm": dn.tolist(),
        "max": mx4,
        "gap": gap,
        "thumb": th,
        "thumb_ok": bool(thumb_ok) if have_thumb and tier >= 4 else None,
        "reason": "ok",
    }


def mode_classify_confidence(
    mode: int,
    debug: dict,
    *,
    thumb_tip_vis: float | None = None,
    hand_vis_min: float | None = None,
) -> float:
    """Score in [0, 1] for how trustworthy a mode_classify result is (higher = more confident).

    Optional ``thumb_tip_vis`` / ``hand_vis_min`` (Orbbec MP) down-weight tier calls when
    fingertips are occluded even if overall hand visibility still looks acceptable.
    """
    reason = str(debug.get("reason", ""))
    if reason == "no_palm":
        return 0.0
    if reason == "fist_or_low":
        return 0.12

    dn = debug.get("d_norm") or []
    if not dn:
        return 0.2

    mx4 = float(debug.get("max", 0.0))
    if mx4 < 0.05:
        return 0.15

    gap = float(debug.get("gap", 0.08 * mx4))
    sep = min(1.0, gap / max(0.10 * mx4, 1e-6))
    conf = 0.38 + 0.42 * sep

    mode_i = int(mode)
    if len(dn) >= 5:
        th = float(dn[4])
        thumb_ok = debug.get("thumb_ok")
        if mode_i == 4:
            # Four fingers up but thumb signal weak — often depth occlusion, not true M4.
            if th < THUMB_PROMOTE_ABS_MIN * 0.45:
                conf *= 0.12
            elif thumb_ok is False and th >= THUMB_PROMOTE_ABS_MIN * 0.70:
                conf *= 0.28
            elif thumb_ok is False:
                conf *= 0.35
        elif mode_i == 5:
            if thumb_ok:
                conf = min(1.0, conf + 0.18)
            elif th >= THUMB_PROMOTE_ABS_MIN:
                conf = min(1.0, conf + 0.08)
            elif thumb_ok is False:
                conf *= 0.42

    if thumb_tip_vis is not None:
        tv = float(np.clip(float(thumb_tip_vis), 0.0, 1.0))
        if mode_i in (4, 5):
            # Ramp: thumb vis 0.82+ ≈ no cut; below that crush M4/M5 trust on depth cam.
            if tv < 0.82:
                vis_scale = 0.10 + 0.90 * (tv / 0.82) ** 2.6
                conf *= float(vis_scale)
        elif tv < 0.55:
            conf *= 0.55 + 0.45 * (tv / 0.55)

    if hand_vis_min is not None:
        hv = float(np.clip(float(hand_vis_min), 0.0, 1.0))
        if hv < 0.80:
            conf *= 0.32 + 0.68 * (hv / 0.80) ** 1.8

    return float(np.clip(conf, 0.0, 1.0))
