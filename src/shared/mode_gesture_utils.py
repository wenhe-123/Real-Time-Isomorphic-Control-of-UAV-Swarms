"""Left-hand mode gestures: palm scale and finger-extension tiers → morph modes (webcam mode selection)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Tier 4 → 5: thumb must pass these (thumb is noisier than other tips; avoid false M5 with 4 fingers up).
THUMB_PROMOTE_ABS_MIN = 0.74
THUMB_PROMOTE_REL_MX4 = 0.78
# Below this vs max of index..pinky, do not count as intentional thumb-up.
THUMB_PROMOTE_MAX_BELOW_MX4 = 0.27


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
        float(THUMB_PROMOTE_ABS_MIN),
        float(THUMB_PROMOTE_REL_MX4) * mx4,
        mx4 - float(THUMB_PROMOTE_MAX_BELOW_MX4),
    )
    if tier == 4 and thumb_ok:
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
