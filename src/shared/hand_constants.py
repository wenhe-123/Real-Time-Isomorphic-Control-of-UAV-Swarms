"""Hand skeleton constants: MediaPipe 21-point edges, fingertip/MCP/wrist IDs for drawing and topology."""

from __future__ import annotations

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]
WRIST_ID = 0
MCP_IDS = [5, 9, 13, 17]

