"""Debug trace for palm +Y sign (wrist→middle MCP line)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MiddleYSignDebug:
    """One frame of +Y half-space decision (direction line + sign lock)."""

    reason: str = ""
    axial_mm: float | None = None
    seg_axial_mm: dict[str, float] = field(default_factory=dict)
    ey_raw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ey_out: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dot_out_prev: float | None = None
    dot_raw_prev_raw: float | None = None
    dot_out_ref: float | None = None
    flipped: bool = False
    anchor_used: str = "none"
    wrist_z_mm: float | None = None
    mcp_z_mm: float | None = None
    pip_z_mm: float | None = None
    tip_z_mm: float | None = None


def _vec3_s(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})"


def print_middle_y_sign_debug(
    dbg: MiddleYSignDebug,
    *,
    frame_idx: int,
    force: bool = False,
) -> None:
    """Print +Y sign decision; ``force`` or ``dbg.flipped`` always prints."""
    if not force and not dbg.flipped:
        return
    tag = "FLIP" if dbg.flipped else "y-sign"
    segs = " ".join(f"{k}={v:+.1f}mm" for k, v in dbg.seg_axial_mm.items())
    print(
        f"[middle-Y {tag}] frame={int(frame_idx)} reason={dbg.reason!r}"
        f" anchor={dbg.anchor_used}"
        f" axial={dbg.axial_mm if dbg.axial_mm is not None else 'n/a'}"
        f" segs=[{segs}]",
        flush=True,
    )
    print(
        f"  raw={_vec3_s(dbg.ey_raw)} out={_vec3_s(dbg.ey_out)}"
        f" dot(out,prev)={dbg.dot_out_prev if dbg.dot_out_prev is not None else 'n/a'}"
        f" dot(raw,prev_raw)={dbg.dot_raw_prev_raw if dbg.dot_raw_prev_raw is not None else 'n/a'}"
        f" dot(out,ref)={dbg.dot_out_ref if dbg.dot_out_ref is not None else 'n/a'}",
        flush=True,
    )
    print(
        f"  depth Z mm wrist={dbg.wrist_z_mm} mcp={dbg.mcp_z_mm}"
        f" pip={dbg.pip_z_mm} tip={dbg.tip_z_mm}",
        flush=True,
    )
