"""Hand skeleton constants: MediaPipe 21-point edges, landmark IDs for drawing and control."""

from __future__ import annotations

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

WRIST_ID = 0
THUMB_MCP_ID = 2
THUMB_TIP_ID = 4
INDEX_MCP_ID = 5
INDEX_TIP_ID = 8
MIDDLE_MCP_ID = 9
MIDDLE_TIP_ID = 12
RING_MCP_ID = 13
RING_TIP_ID = 16
PINKY_MCP_ID = 17
PINKY_TIP_ID = 20

FINGERTIP_IDS = [THUMB_TIP_ID, INDEX_TIP_ID, MIDDLE_TIP_ID, RING_TIP_ID, PINKY_TIP_ID]
FINGERTIP_IDS_FOUR = (INDEX_TIP_ID, MIDDLE_TIP_ID, RING_TIP_ID, PINKY_TIP_ID)
# Mode gesture 1–5: index → thumb (order matters for classify_mode_from_fingers).
MODE_COUNT_TIP_IDS = [INDEX_TIP_ID, MIDDLE_TIP_ID, RING_TIP_ID, PINKY_TIP_ID, THUMB_TIP_ID]
MCP_IDS = [INDEX_MCP_ID, MIDDLE_MCP_ID, RING_MCP_ID, PINKY_MCP_ID]

# Wrist + five finger MCPs (palm frame origin / depth anchor).
PALM_CENTER_IDS = (
    WRIST_ID,
    THUMB_MCP_ID,
    INDEX_MCP_ID,
    MIDDLE_MCP_ID,
    RING_MCP_ID,
    PINKY_MCP_ID,
)

# Wrist-relative span proxy (mean distance to these landmarks).
HAND_SPAN_LANDMARK_IDS = (
    THUMB_MCP_ID,
    THUMB_TIP_ID,
    INDEX_MCP_ID,
    INDEX_TIP_ID,
    MIDDLE_MCP_ID,
    MIDDLE_TIP_ID,
    RING_MCP_ID,
    RING_TIP_ID,
    PINKY_TIP_ID,
)
