"""Webcam mode pipeline defaults and helpers shared by online Orbbec/dual runtimes.

Left-hand mode 1–5 classification, HUD refresh thresholds, and topology analysis
defaults originally lived in ``runtime/hand_tracking_webcam_modes.py``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from functions.mode_switch.hand_constants import (
    FINGERTIP_IDS,
    MCP_IDS,
    MODE_COUNT_TIP_IDS,
    WRIST_ID,
)
from functions.mode_switch.mode_gesture_utils import (
    classify_mode_from_fingers as _shared_classify_mode_from_fingers,
    palm_center_and_scale as _shared_palm_center_and_scale,
)
from functions.mode_switch.topology_utils import (
    analyze_hand_topology_common,
    topology_label_from_alpha,
)

OPEN_GAMMA = 1.8
TOPO_ALPHA_PLANE = 0.67
TOPO_ALPHA_SPHERE = 0.33
MORPH_AXIS_LIM_MM = 200.0
NORM_AXIS_HALFLIM = 1.35
HAND_3D_SOURCE_MP = "mp"
HAND_3D_SOURCE_FUSED = "fused"
HAND_FRAME_SCALED = "scaled"
HAND_FRAME_PALM_PLANE = "palm_plane"

PLOT_EVERY_N_FRAMES = 2

PLANE_SNAP_ON = 0.82
PLANE_SNAP_OFF = 0.78
SPHERE_SNAP_ON = 0.12
SPHERE_SNAP_OFF = 0.18

HUD_UPDATE_EVERY_N_FRAMES = 10
HUD_OPEN_STEP = 0.03
HUD_METRIC_STEP = 0.05

SNAP_SHOW_AFTER_FRAMES = 6
SNAP_HOLD_AFTER_RELEASE_FRAMES = 10

MODE_EXTEND_MIN = 0.62
MODE_TIER_GAP = 0.38
MODE_DEBOUNCE_FRAMES = 7


def palm_center_and_scale(hand_points: Sequence[Tuple[float, float, float]]):
    """Compute palm centroid and scale using runtime wrist/MCP constants.

    Args:
        hand_points: Sequence of 21 hand landmarks.

    Returns:
        Tuple ``(palm_center, scale)`` from shared ``palm_center_and_scale``.
    """
    return _shared_palm_center_and_scale(hand_points, WRIST_ID, MCP_IDS)


def classify_mode_from_fingers(hand_points: Sequence[Tuple[float, float, float]]):
    """Classify morph mode M1–M5 using runtime extension thresholds.

    Args:
        hand_points: Hand landmarks (normalized or metric).

    Returns:
        Tuple ``(mode, tier_count, debug)`` from shared finger-tier classifier.
    """
    return _shared_classify_mode_from_fingers(
        hand_points,
        mode_count_tip_ids=MODE_COUNT_TIP_IDS,
        mode_extend_min=MODE_EXTEND_MIN,
        mode_tier_gap=MODE_TIER_GAP,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
    )


def classify_mode_from_fingers_webcam_image(hand_points: Sequence[Tuple[float, float, float]]):
    """Classify mode from 2D image-plane landmarks with relaxed thumb promote.

    Slightly relaxed thumb thresholds help M4→M5 on USB cameras.

    Args:
        hand_points: Image-plane pseudo-mm landmarks from USB MediaPipe.

    Returns:
        Tuple ``(mode, tier_count, debug)`` from shared finger-tier classifier.
    """
    return _shared_classify_mode_from_fingers(
        hand_points,
        mode_count_tip_ids=MODE_COUNT_TIP_IDS,
        mode_extend_min=MODE_EXTEND_MIN * 0.95,
        mode_tier_gap=MODE_TIER_GAP,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
        thumb_promote_abs_min=0.48,
        thumb_promote_rel_mx4=0.58,
        thumb_promote_max_below_mx4=0.44,
    )


def analyze_hand_topology(hand_points):
    """Run topology PCA analysis using runtime gamma and label thresholds.

    Args:
        hand_points: Sequence of 21 hand landmarks.

    Returns:
        Topology analysis dict, or ``None`` when too few valid points remain.
    """
    return analyze_hand_topology_common(
        hand_points,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
        fingertip_ids=FINGERTIP_IDS,
        open_gamma=OPEN_GAMMA,
        label_fn=lambda a: topology_label_from_alpha(
            a, plane_thr=TOPO_ALPHA_PLANE, sphere_thr=TOPO_ALPHA_SPHERE
        ),
    )


