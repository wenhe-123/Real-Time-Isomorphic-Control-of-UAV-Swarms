"""Mode runtime core: mode defs, left/right state machines, open smoothing, SNAP HUD, unified 3D plot update."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shared.plot_3d_utils import (
    apply_hand_axis_limits,
    finalize_dual_3d_axes,
    plot_hand_points_connections,
    setup_hand_axis,
    setup_topology_axis,
)


@dataclass(frozen=True)
class ModeDef:
    mode_id: int
    name: str
    hint: str


MODE_1 = ModeDef(mode_id=1, name="SphereLike", hint="1=sph")
MODE_2 = ModeDef(mode_id=2, name="CylinderLike", hint="2=cyl")
MODE_3 = ModeDef(mode_id=3, name="CubeLike", hint="3=cube")
MODE_4 = ModeDef(mode_id=4, name="SquareColumn", hint="4=sqZ")
MODE_5 = ModeDef(mode_id=5, name="AsymSuper", hint="5=asym")
MODES = {
    MODE_1.mode_id: MODE_1,
    MODE_2.mode_id: MODE_2,
    MODE_3.mode_id: MODE_3,
    MODE_4.mode_id: MODE_4,
    MODE_5.mode_id: MODE_5,
}
ALL_MODES_HINT = " ".join([MODE_1.hint, MODE_2.hint, MODE_3.hint, MODE_4.hint, MODE_5.hint])


@dataclass
class ModeState:
    mode_raw: int = 1
    mode_ema: float = 1.0
    morph_mode: int = 1
    mode_raw_prev: Optional[int] = None
    mode_stable_frames: int = 0
    last_mode_raw: int = 1


@dataclass
class RightHandState:
    last_right_pts: Optional[List] = None
    last_open_out: Optional[float] = None
    open_free_ema: Optional[float] = None
    snap_state: Optional[str] = None


@dataclass
class SnapVisualState:
    snap_vis_state: Optional[str] = None
    snap_stable_frames: int = 0
    snap_hold_frames: int = 0


@dataclass
class RuntimeState:
    frame_idx: int = 0
    warned_fusion_linear_map: bool = False
    ema_3d: Any = None
    enable_3d: bool = True
    hud_cache: Dict[str, Any] = field(
        default_factory=lambda: {
            "open": None,
            "free": None,
            "plan": None,
            "iso": None,
            "spread": None,
            "curl": None,
            "text": None,
        }
    )


def update_mode_state(
    pts_left,
    *,
    mode_state,
    classify_mode_fn: Callable,
    debounce_frames: int,
    mode_smooth: float,
) -> Tuple[int, int]:
    tier_count = -1
    if pts_left is None:
        mode_state.mode_raw = mode_state.last_mode_raw
        return mode_state.mode_raw, tier_count

    mode_raw, tier_count, _dbg = classify_mode_fn(pts_left)
    mode_state.last_mode_raw = mode_raw
    mode_state.mode_raw = mode_raw

    if mode_state.mode_raw_prev is None:
        mode_state.mode_raw_prev = mode_raw
        mode_state.mode_stable_frames = debounce_frames
    elif mode_raw == mode_state.mode_raw_prev:
        mode_state.mode_stable_frames += 1
    else:
        mode_state.mode_raw_prev = mode_raw
        mode_state.mode_stable_frames = 0

    if mode_state.mode_stable_frames >= debounce_frames:
        # Use the stable raw gesture directly. EMA made transition time depend on
        # mode distance (e.g. 1->4 much slower than 1->5), so stable-direct keeps
        # all mode pairs on the same timing path.
        mode_state.mode_ema = float(mode_raw)
        mode_state.morph_mode = int(mode_raw)
        mode_state.morph_mode = max(1, min(max(MODES.keys()), mode_state.morph_mode))

    return mode_raw, tier_count


def update_open_state(
    pts_right,
    *,
    right_state,
    analyze_topology_fn: Callable,
    open_smooth: float,
    plane_snap_on: float,
    plane_snap_off: float,
    sphere_snap_on: float,
    sphere_snap_off: float,
) -> Optional[float]:
    if pts_right is None:
        return right_state.last_open_out

    tmp = analyze_topology_fn(pts_right)
    if tmp is None:
        return None

    if right_state.open_free_ema is None:
        right_state.open_free_ema = float(tmp["morph_alpha"])
    else:
        right_state.open_free_ema = (
            open_smooth * float(tmp["morph_alpha"]) + (1.0 - open_smooth) * right_state.open_free_ema
        )

    open_free = float(right_state.open_free_ema)
    if right_state.snap_state == "plane":
        if open_free < plane_snap_off:
            right_state.snap_state = None
    elif right_state.snap_state == "sphere":
        if open_free > sphere_snap_off:
            right_state.snap_state = None
    else:
        if open_free > plane_snap_on:
            right_state.snap_state = "plane"
        elif open_free < sphere_snap_on:
            right_state.snap_state = "sphere"

    open_out = open_free
    if right_state.snap_state == "plane":
        open_out = 1.0
    elif right_state.snap_state == "sphere":
        open_out = 0.0
    right_state.last_open_out = float(open_out)
    return open_out


def update_snap_visual_state(
    snap_state: Optional[str],
    *,
    snap_visual_state,
    snap_show_after_frames: int,
    snap_hold_after_release_frames: int,
):
    if snap_state is None:
        snap_visual_state.snap_stable_frames = 0
        if snap_visual_state.snap_vis_state is not None:
            snap_visual_state.snap_hold_frames += 1
            if snap_visual_state.snap_hold_frames >= snap_hold_after_release_frames:
                snap_visual_state.snap_vis_state = None
                snap_visual_state.snap_hold_frames = 0
        return

    snap_visual_state.snap_hold_frames = 0
    if snap_visual_state.snap_vis_state == snap_state:
        snap_visual_state.snap_stable_frames = min(
            snap_show_after_frames,
            snap_visual_state.snap_stable_frames + 1,
        )
    else:
        snap_visual_state.snap_stable_frames += 1
        if snap_visual_state.snap_stable_frames >= snap_show_after_frames:
            snap_visual_state.snap_vis_state = snap_state
            snap_visual_state.snap_stable_frames = 0


def update_3d_plot_modes(
    ax_hand,
    ax_topo,
    hands_3d,
    *,
    morph_mode: int,
    morph_alpha_smoothed,
    control_label: str,
    shape_normalized: bool,
    hand_frame: str,
    hand_3d_source: str,
    hand_frame_palm_plane: str,
    hand_connections,
    norm_axis_halflim: float,
    morph_axis_lim_mm: float,
    analyze_hand_topology_fn: Callable,
    draw_mode1_fn: Callable,
    draw_mode2_fn: Callable,
    draw_mode3_fn: Callable,
    clamp01_fn: Callable,
):
    src = "MediaPipe" if hand_3d_source == "mp" else "depth+MP fused"
    title = f"Hand 3D ({src}) joints 0..20"
    if control_label:
        title += f" | {control_label}"
    setup_hand_axis(
        ax_hand,
        title,
        shape_normalized=shape_normalized,
        hand_frame=hand_frame,
        palm_plane_frame=hand_frame_palm_plane,
    )

    mode_titles = {
        1: "M1 superellipsoid",
        2: "M2 superellipsoid",
        3: "M3 superellipsoid",
        4: "M4 superellipsoid",
        5: "M5 superellipsoid",
    }
    setup_topology_axis(ax_topo, mode_titles.get(morph_mode, "Morph"))

    analyses = []
    for hand_points in hands_3d:
        arr = np.array(hand_points, dtype=float)
        if arr.size == 0:
            continue
        valid = ~np.isnan(arr[:, 2])
        if not np.any(valid):
            continue

        plot_hand_points_connections(ax_hand, arr, valid, hand_connections)
        apply_hand_axis_limits(
            ax_hand,
            arr,
            valid,
            shape_normalized=shape_normalized,
            norm_axis_halflim=norm_axis_halflim,
            morph_axis_lim_mm=morph_axis_lim_mm,
        )

        analysis = analyze_hand_topology_fn(hand_points)
        if analysis is None:
            morph_fb = 0.55 if morph_alpha_smoothed is None else float(morph_alpha_smoothed)
            morph_fb = clamp01_fn(morph_fb)
            r_draw = 200.0
            if morph_mode == 1:
                draw_mode1_fn(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
            elif morph_mode == 2:
                draw_mode2_fn(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
            else:
                draw_mode3_fn(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
            ax_topo.text(
                -r_draw,
                -r_draw,
                r_draw * 0.92,
                f"open={morph_fb:.2f}\n(no topology)",
                color="tab:orange",
            )
            continue
        analyses.append(analysis)

        r = max(analysis["radius"], 1.0)
        morph_alpha = analysis["morph_alpha"] if morph_alpha_smoothed is None else morph_alpha_smoothed
        r_vis = max(140.0, 2.2 * r)
        if morph_mode == 1:
            draw_mode1_fn(ax_topo, radius=r_vis, open_alpha=morph_alpha, show_refs=True)
        elif morph_mode == 2:
            draw_mode2_fn(ax_topo, radius=r_vis, open_alpha=morph_alpha, show_refs=True)
        else:
            draw_mode3_fn(ax_topo, radius=r_vis, open_alpha=morph_alpha, show_refs=True)

        span_note = f"  span={analysis['span_ratio']:.1f}" if shape_normalized else ""
        ax_topo.text(
            -r_vis,
            -r_vis,
            r_vis * 0.92,
            f"open={morph_alpha:.2f}  plan={analysis['planarity']:.2f}  iso={analysis['isotropy']:.2f}{span_note}",
            color="tab:purple",
        )

    finalize_dual_3d_axes(ax_hand, ax_topo, morph_axis_lim_mm=morph_axis_lim_mm)
    return analyses


def build_modes_hud_lines(
    *,
    morph_mode: int,
    topo_label: str,
    open_disp: float,
    free_disp: float,
    spread: float,
    planarity: float,
    isotropy: float,
):
    return [
        f"M{morph_mode}  open:{open_disp:.2f}  free:{free_disp:.2f}  topo:{topo_label}",
        f"spread:{spread:.2f}  plan:{planarity:.2f}  iso:{isotropy:.2f}",
    ]


MODE_SMOOTH = 0.22
OPEN_SMOOTH = 0.18


def process_left_mode(
    keypoints_3d: List,
    idx_left: Optional[int],
    mode_state: ModeState,
) -> Tuple[int, int]:
    """Update left-hand mode state and return (mode_raw, tier_count)."""
    import runtime.hand_tracking_webcam_modes as hm

    pts_left = None
    if idx_left is not None and idx_left < len(keypoints_3d):
        pts_left = keypoints_3d[idx_left]
    return update_mode_state(
        pts_left,
        mode_state=mode_state,
        classify_mode_fn=hm.classify_mode_from_fingers,
        debounce_frames=hm.MODE_DEBOUNCE_FRAMES,
        mode_smooth=MODE_SMOOTH,
    )


def process_right_open(
    keypoints_3d: List,
    idx_right: Optional[int],
    right_state: RightHandState,
) -> Tuple[List, Optional[float]]:
    """Update right-hand open state and return (hands_3d_for_plot, open_out)."""
    import runtime.hand_tracking_orbbec as ob

    hands_3d: List = []
    open_out: Optional[float] = None
    if idx_right is not None and idx_right < len(keypoints_3d):
        pts_right = keypoints_3d[idx_right]
        right_state.last_right_pts = list(pts_right)
        hands_3d = [pts_right]
        open_out = update_open_state(
            pts_right,
            right_state=right_state,
            analyze_topology_fn=ob.analyze_hand_topology,
            open_smooth=OPEN_SMOOTH,
            plane_snap_on=ob.PLANE_SNAP_ON,
            plane_snap_off=ob.PLANE_SNAP_OFF,
            sphere_snap_on=ob.SPHERE_SNAP_ON,
            sphere_snap_off=ob.SPHERE_SNAP_OFF,
        )
    else:
        if right_state.last_right_pts is not None:
            hands_3d = [right_state.last_right_pts]
        open_out = right_state.last_open_out
    return hands_3d, open_out


def draw_bottom_status(
    frame,
    morph_mode: int,
    mode_raw: int,
    tier_count: int,
    idx_left: Optional[int],
    idx_right: Optional[int],
    open_out: Optional[float],
):
    hint_parts = []
    if idx_left is None:
        hint_parts.append("no LEFT (mode)")
    if idx_right is None:
        hint_parts.append("no RIGHT (open frozen)")
    hint = "  |  ".join(hint_parts) if hint_parts else "L=mode R=open"
    otxt = f"{open_out:.2f}" if open_out is not None else "-"
    cv2.putText(
        frame,
        f"M{morph_mode} raw:{mode_raw}  open:{otxt}  tier:{tier_count if tier_count >= 0 else '-'}  {hint}"[:95],
        (16, frame.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def overlay_mode_open_wrist_labels(
    *,
    frame,
    result,
    idx_left: Optional[int],
    idx_right: Optional[int],
    morph_mode: int,
    open_out: Optional[float],
    overlay_wrist_labels_fn: Callable,
) -> None:
    """Overlay compact labels at detected wrist(s): left=M#, right=open value.

    The actual drawing implementation is injected via ``overlay_wrist_labels_fn`` to keep this module
    independent from the concrete runtime (Orbbec/webcam/dual).
    """
    wrist_lbl: Dict[int, str] = {}
    if idx_left is not None:
        wrist_lbl[int(idx_left)] = f"M{int(morph_mode)}"
    if idx_right is not None:
        wrist_lbl[int(idx_right)] = f"open {float(open_out):.2f}" if open_out is not None else "open —"
    if wrist_lbl:
        overlay_wrist_labels_fn(frame, result, wrist_lbl)


def update_snap_visual_state_for_modes(snap_state: Optional[str], snap_vis_state: SnapVisualState):
    import runtime.hand_tracking_orbbec as ob

    update_snap_visual_state(
        snap_state,
        snap_visual_state=snap_vis_state,
        snap_show_after_frames=ob.SNAP_SHOW_AFTER_FRAMES,
        snap_hold_after_release_frames=ob.SNAP_HOLD_AFTER_RELEASE_FRAMES,
    )


def update_hud_cache(
    runtime: RuntimeState,
    frame_idx: int,
    analyses: List[Dict[str, Any]],
    hands_3d: List,
    hand_frame: str,
    morph_mode: int,
    open_out: Optional[float],
    open_free_ema: Optional[float],
    open_remap: Optional[Tuple[float, float]],
    snap_vis_state: Optional[str],
):
    import runtime.hand_tracking_orbbec as ob
    import runtime.hand_tracking_webcam_modes as hm

    a0 = analyses[0]
    topo_lbl = ob._topology_label_from_alpha(float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"]))
    open_disp = open_out if open_out is not None else a0["morph_alpha"]
    free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
    if open_remap is not None:
        lo_r, hi_r = open_remap
        open_disp = ob._remap_open_display(open_disp, lo_r, hi_r)
        free_disp = ob._remap_open_display(free_disp, lo_r, hi_r)

    need_refresh = (frame_idx % hm.HUD_UPDATE_EVERY_N_FRAMES) == 0 or runtime.hud_cache["open"] is None
    if not need_refresh:
        if abs(float(open_disp) - float(runtime.hud_cache["open"])) > hm.HUD_OPEN_STEP:
            need_refresh = True
        if abs(float(free_disp) - float(runtime.hud_cache["free"])) > hm.HUD_OPEN_STEP:
            need_refresh = True
        if abs(float(a0["planarity"]) - float(runtime.hud_cache["plan"])) > hm.HUD_METRIC_STEP:
            need_refresh = True
        if abs(float(a0["isotropy"]) - float(runtime.hud_cache["iso"])) > hm.HUD_METRIC_STEP:
            need_refresh = True
        if abs(float(a0["finger_spread"]) - float(runtime.hud_cache["spread"])) > hm.HUD_METRIC_STEP:
            need_refresh = True

    if not need_refresh:
        return

    runtime.hud_cache["open"] = float(open_disp)
    runtime.hud_cache["free"] = float(free_disp)
    runtime.hud_cache["plan"] = float(a0["planarity"])
    runtime.hud_cache["iso"] = float(a0["isotropy"])
    runtime.hud_cache["spread"] = float(a0["finger_spread"])
    lines = build_modes_hud_lines(
        morph_mode=morph_mode,
        topo_label=topo_lbl,
        open_disp=float(open_disp),
        free_disp=float(free_disp),
        spread=float(a0["finger_spread"]),
        planarity=float(a0["planarity"]),
        isotropy=float(a0["isotropy"]),
    )
    if hand_frame == ob.HAND_FRAME_PALM_PLANE and hands_3d:
        cm = ob._palm_plane_curl_metrics(hands_3d[0])
        if cm and cm.get("mean_r_xy_four") is not None:
            tr = cm.get("thumb_r_xy")
            thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
            curl_s = f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}{thumb_s}"
            runtime.hud_cache["curl"] = curl_s
            lines.append(curl_s)
        else:
            runtime.hud_cache["curl"] = None
    else:
        runtime.hud_cache["curl"] = None
    runtime.hud_cache["text"] = lines

