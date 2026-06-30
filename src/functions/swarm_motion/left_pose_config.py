"""Static presets and camera→sim embedding for left-hand swarm pose."""

from __future__ import annotations

import numpy as np

from functions.mode_switch.hand_constants import (
    INDEX_MCP_ID,
    INDEX_TIP_ID,
    MCP_IDS,
    MIDDLE_MCP_ID,
    MIDDLE_TIP_ID,
    RING_MCP_ID,
    RING_TIP_ID,
    THUMB_MCP_ID,
    THUMB_TIP_ID,
    WRIST_ID,
)

LEFT_PALM_BASIS_PRESETS: dict[str, tuple[int, int]] = {
    # +Y wrist→middle MCP; +Z = Y×(palm→thumb chain); +X = palm→thumb tip in plane ⊥Y.
    "middle_thumb": (MIDDLE_MCP_ID, THUMB_MCP_ID),
    "index_middle": (MIDDLE_MCP_ID, INDEX_MCP_ID),
    "index_ring": (RING_MCP_ID, INDEX_MCP_ID),
    "middle_ring": (RING_MCP_ID, MIDDLE_MCP_ID),
}


DEFAULT_LEFT_PALM_BASIS = "middle_thumb"


def palm_basis_pair_indices(preset: str) -> tuple[int, int]:
    key = str(preset).strip().lower()
    if key not in LEFT_PALM_BASIS_PRESETS:
        keys = ", ".join(sorted(LEFT_PALM_BASIS_PRESETS))
        raise ValueError(f"unknown left palm basis {preset!r}; choose one of: {keys}")
    a, b = LEFT_PALM_BASIS_PRESETS[key]
    return int(a), int(b)


LEFT_CAM_PRESET_ROT: dict[str, np.ndarray] = {
    # Depth-camera axes = simulation axes (X right, Y down, Z forward). Use with ``camera_at_arm``.
    "camera": np.eye(3, dtype=np.float64),
    # Older iso_swarm mapping: −optical Z → world X, +cam X → world Y, −cam Y → world Z.
    "legacy": np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    # Optical Z (depth) → world Y “into scene”; optical X (image right) → world X; optical Y → −world Z.
    "fwd_y": np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    # Negate depth column vs ``legacy`` (try if near/far feels inverted).
    "flip_depth": np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
}


def build_sim_from_cam_matrices(
    preset: str,
    *,
    image_y_to_world_z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M_rot, M_trans)`` embedding depth-camera mm into simulation coordinates."""
    M_rot = left_cam_preset_rotation(preset)
    M_trans = make_cam_translation_matrix(M_rot, image_y_to_world_z=float(image_y_to_world_z))
    return M_rot, M_trans


def left_cam_preset_rotation(preset: str) -> np.ndarray:
    """Return a copy of the orthonormal rotation for ``preset`` (``camera`` | ``legacy`` | ``fwd_y`` | ``flip_depth``)."""
    key = str(preset).strip().lower()
    if key not in LEFT_CAM_PRESET_ROT:
        keys = ", ".join(sorted(LEFT_CAM_PRESET_ROT))
        raise ValueError(f"unknown left-cam preset {preset!r}; choose one of: {keys}")
    return np.asarray(LEFT_CAM_PRESET_ROT[key], dtype=np.float64).copy()


def make_cam_translation_matrix(M_rot: np.ndarray, *, image_y_to_world_z: float = 0.0) -> np.ndarray:
    """World mm delta from camera mm delta: ``M_trans @ v`` with same XY rows as ``M_rot``.

    World-Z row is ``M_rot[2,:] * image_y_to_world_z`` so ``0`` disables altitude from camera Y
    (avoids near/far coupling into ``wz``). ``1`` restores full third row like rotation.
    """
    M = np.asarray(M_rot, dtype=np.float64).reshape(3, 3).copy()
    s = float(np.clip(image_y_to_world_z, 0.0, 1.0))
    M[2, :] *= s
    return M


_PALM_ROOT_IDS = (WRIST_ID, THUMB_MCP_ID, *MCP_IDS)


_PALM_PLANE_MIN_POINTS = 4


_PALM_PLANE_OUTLIER_MM = 40.0


_PALM_PLANE_MAX_MEDIAN_RESID_MM = 38.0


_PALM_PLANE_Z_MIN_MM = 180.0


_PALM_PLANE_Z_MAX_MM = 1600.0


# Depths (mm) for two-point ray through a color pixel when intersecting the palm plane.
_PALM_PLANE_RAY_Z0_MM = 350.0
_PALM_PLANE_RAY_Z1_MM = 950.0


_PALM_PLANE_CORE_IDS = (WRIST_ID, THUMB_MCP_ID, *MCP_IDS)


_PALM_PLANE_OPTIONAL_TIP_IDS = (MIDDLE_TIP_ID, INDEX_TIP_ID, RING_TIP_ID)
