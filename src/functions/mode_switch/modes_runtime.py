"""Mode runtime core: mode defs, left/right state machines, open smoothing, SNAP HUD, unified 3D plot update."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# While left palm rotation or large translation is active, morph mode (M1..M5) is frozen.
DEFAULT_MODE_ROT_FREEZE_BLEND = 0.12
DEFAULT_MODE_ROT_FREEZE_RV_DEG = 3.5
DEFAULT_MODE_TRANS_FREEZE_BLEND = 0.35
DEFAULT_MODE_TRANS_FREEZE_CMD_M = 0.028
DEFAULT_MODE_TRANS_FREEZE_ARM_MM = 36.0
DEFAULT_MODE_TRANS_FREEZE_FRAME_IDLE_MM = 5.0
DEFAULT_MODE_ROT_FREEZE_LATCH_FRAMES = 18
from functions.display_sim.plot_3d_utils import (
    apply_hand_axis_limits,
    finalize_dual_3d_axes,
    plot_hand_points_connections,
    setup_hand_axis,
    setup_topology_axis,
)
from functions.mode_switch.dual_mode_fusion import classify_mode_dual
from functions.mode_switch.hand_constants import FINGERTIP_IDS_FOUR
from functions.mode_switch.hand_frame_utils import palm_plane_curl_metrics as _palm_plane_curl_metrics_shared
from functions.mode_switch.topology_utils import remap_open_display, topology_label_from_alpha
from functions.mode_switch.webcam_mode_defaults import (
    HAND_FRAME_PALM_PLANE,
    HUD_METRIC_STEP,
    HUD_OPEN_STEP,
    HUD_UPDATE_EVERY_N_FRAMES,
    MODE_DEBOUNCE_FRAMES,
    PLANE_SNAP_OFF,
    PLANE_SNAP_ON,
    SNAP_HOLD_AFTER_RELEASE_FRAMES,
    SNAP_SHOW_AFTER_FRAMES,
    SPHERE_SNAP_OFF,
    SPHERE_SNAP_ON,
    TOPO_ALPHA_PLANE,
    TOPO_ALPHA_SPHERE,
    analyze_hand_topology,
    classify_mode_from_fingers,
    classify_mode_from_fingers_webcam_image,
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


def topology_label_from_morph_alpha(alpha: float) -> str:
    return topology_label_from_alpha(alpha, plane_thr=TOPO_ALPHA_PLANE, sphere_thr=TOPO_ALPHA_SPHERE)


def palm_plane_curl_metrics(points_21):
    return _palm_plane_curl_metrics_shared(points_21, fingertip_ids_four=FINGERTIP_IDS_FOUR)


@dataclass
class ModeState:
    mode_raw: int = 1
    mode_ema: float = 1.0
    morph_mode: int = 1
    mode_raw_prev: Optional[int] = None
    mode_stable_frames: int = 0
    last_mode_raw: int = 1
    mode_freeze_latch: int = 0


def obvious_left_rotation_for_mode_hold(
    left_pose_state,
    *,
    blend_thresh: float = DEFAULT_MODE_ROT_FREEZE_BLEND,
    rv_deg_min: float = DEFAULT_MODE_ROT_FREEZE_RV_DEG,
) -> bool:
    """True when the previous pose step classified obvious palm rotation (freeze M1..M5)."""
    if left_pose_state is None or not bool(getattr(left_pose_state, "enabled", False)):
        return False
    lm = str(getattr(left_pose_state, "last_axis_motion", ""))
    if lm != "rotate":
        return False
    lw = float(getattr(left_pose_state, "last_rot_blend_w", 0.0))
    if lw >= float(blend_thresh):
        return True
    for attr in ("last_rv_pose_world", "last_rv_cmd_world"):
        rv = getattr(left_pose_state, attr, None)
        if rv is None:
            continue
        if float(np.linalg.norm(np.asarray(rv, dtype=np.float64).reshape(3))) >= np.deg2rad(float(rv_deg_min)):
            return True
    return False


def obvious_left_translation_for_mode_hold(
    left_pose_state,
    *,
    blend_thresh: float = DEFAULT_MODE_TRANS_FREEZE_BLEND,
    cmd_m_min: float = DEFAULT_MODE_TRANS_FREEZE_CMD_M,
    arm_mm_min: float = DEFAULT_MODE_TRANS_FREEZE_ARM_MM,
    frame_idle_mm: float = DEFAULT_MODE_TRANS_FREEZE_FRAME_IDLE_MM,
) -> bool:
    """True when the previous pose step had obvious palm translation (freeze M1..M5).

    Uses palm center delta vs the previous frame only. ``last_delta_h_world`` and arm
    offset must not count — the hand can stay displaced while idle without locking mode.
    """
    _ = cmd_m_min
    if left_pose_state is None or not bool(getattr(left_pose_state, "enabled", False)):
        return False
    if str(getattr(left_pose_state, "last_axis_motion", "")) != "translate":
        return False
    tw = float(getattr(left_pose_state, "last_trans_blend_w", 0.0))
    if tw < float(blend_thresh):
        return False
    dc = getattr(left_pose_state, "last_delta_cam_mm", None)
    if dc is None:
        return False
    dc_norm = float(np.linalg.norm(np.asarray(dc, dtype=np.float64).reshape(3)))
    if dc_norm < float(frame_idle_mm):
        return False
    return dc_norm >= float(arm_mm_min)


def left_swarm_motion_holds_mode(
    left_pose_state,
    *,
    rot_blend_thresh: float = DEFAULT_MODE_ROT_FREEZE_BLEND,
    rv_deg_min: float = DEFAULT_MODE_ROT_FREEZE_RV_DEG,
    trans_blend_thresh: float = DEFAULT_MODE_TRANS_FREEZE_BLEND,
    trans_cmd_m_min: float = DEFAULT_MODE_TRANS_FREEZE_CMD_M,
    trans_arm_mm_min: float = DEFAULT_MODE_TRANS_FREEZE_ARM_MM,
) -> bool:
    """True when prior left-swarm step was obvious rotation or large translation."""
    return bool(
        obvious_left_rotation_for_mode_hold(
            left_pose_state,
            blend_thresh=float(rot_blend_thresh),
            rv_deg_min=float(rv_deg_min),
        )
        or obvious_left_translation_for_mode_hold(
            left_pose_state,
            blend_thresh=float(trans_blend_thresh),
            cmd_m_min=float(trans_cmd_m_min),
            arm_mm_min=float(trans_arm_mm_min),
        )
    )


def mode_frozen_for_rotation(
    mode_state: ModeState,
    left_pose_state,
    *,
    blend_thresh: float = DEFAULT_MODE_ROT_FREEZE_BLEND,
    rv_deg_min: float = DEFAULT_MODE_ROT_FREEZE_RV_DEG,
    trans_blend_thresh: float = DEFAULT_MODE_TRANS_FREEZE_BLEND,
    trans_cmd_m_min: float = DEFAULT_MODE_TRANS_FREEZE_CMD_M,
    trans_arm_mm_min: float = DEFAULT_MODE_TRANS_FREEZE_ARM_MM,
) -> bool:
    """Latch or live left-swarm motion flag — morph mode must not change this frame."""
    if int(mode_state.mode_freeze_latch) > 0:
        return True
    return left_swarm_motion_holds_mode(
        left_pose_state,
        rot_blend_thresh=float(blend_thresh),
        rv_deg_min=float(rv_deg_min),
        trans_blend_thresh=float(trans_blend_thresh),
        trans_cmd_m_min=float(trans_cmd_m_min),
        trans_arm_mm_min=float(trans_arm_mm_min),
    )


def tick_mode_rotation_freeze_latch(
    mode_state: ModeState,
    left_pose_state,
    *,
    latch_frames: int = DEFAULT_MODE_ROT_FREEZE_LATCH_FRAMES,
    blend_thresh: float = DEFAULT_MODE_ROT_FREEZE_BLEND,
    rv_deg_min: float = DEFAULT_MODE_ROT_FREEZE_RV_DEG,
    trans_blend_thresh: float = DEFAULT_MODE_TRANS_FREEZE_BLEND,
    trans_cmd_m_min: float = DEFAULT_MODE_TRANS_FREEZE_CMD_M,
    trans_arm_mm_min: float = DEFAULT_MODE_TRANS_FREEZE_ARM_MM,
) -> None:
    """After pose update: extend latch so mode stays frozen through brief motion gaps."""
    if left_swarm_motion_holds_mode(
        left_pose_state,
        rot_blend_thresh=float(blend_thresh),
        rv_deg_min=float(rv_deg_min),
        trans_blend_thresh=float(trans_blend_thresh),
        trans_cmd_m_min=float(trans_cmd_m_min),
        trans_arm_mm_min=float(trans_arm_mm_min),
    ):
        mode_state.mode_freeze_latch = max(
            int(mode_state.mode_freeze_latch),
            int(latch_frames),
        )


def consume_mode_rotation_freeze_latch(mode_state: ModeState) -> None:
    """Call once per frame when mode freeze was applied."""
    if int(mode_state.mode_freeze_latch) > 0:
        mode_state.mode_freeze_latch = int(mode_state.mode_freeze_latch) - 1


def clear_mode_rotation_freeze_latch(mode_state: ModeState) -> None:
    """Drop morph-mode freeze latch (e.g. on left-swarm arm / disarm)."""
    mode_state.mode_freeze_latch = 0


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
    mode_vis_min: float = 0.0,
    hand_visibility_min: float | None = None,
    hold_mode: bool = False,
    pts_left_webcam=None,
    dual_mode_assist: bool = False,
    rotating: bool = False,
    orbbec_thumb_vis: float | None = None,
) -> Tuple[int, int]:
    """Update left-hand morph mode from finger geometry.

    When ``mode_vis_min > 0`` and ``hand_visibility_min`` is below it, **hold** the current
    ``morph_mode`` unless ``dual_mode_assist`` and ``pts_left_webcam`` allow USB-webcam classify.

    When ``hold_mode`` is true (e.g. L-swarm rotation or large translation active), skip classify entirely so
    finger geometry changes during palm twist do not change M1..M5.
    """
    tier_count = -1
    if pts_left is None and pts_left_webcam is None:
        mode_state.mode_raw = mode_state.last_mode_raw
        return mode_state.mode_raw, tier_count

    if bool(hold_mode):
        if bool(dual_mode_assist) and pts_left_webcam is not None:
            mode_raw, tier_count, _dbg = classify_mode_from_fingers_webcam_image(pts_left_webcam)
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
                next_mode = int(mode_raw)
                mode_state.mode_ema = float(next_mode)
                mode_state.morph_mode = max(1, min(max(MODES.keys()), next_mode))
            return mode_state.mode_raw, tier_count
        mode_state.mode_raw = int(mode_state.last_mode_raw)
        return mode_state.mode_raw, tier_count

    vmin = float(mode_vis_min)
    has_webcam_pts = pts_left_webcam is not None
    if (
        vmin > 0.0
        and hand_visibility_min is not None
        and float(hand_visibility_min) < vmin
        and not (bool(dual_mode_assist) and has_webcam_pts)
    ):
        mode_state.mode_raw = int(mode_state.last_mode_raw)
        return mode_state.mode_raw, tier_count

    if bool(dual_mode_assist) and has_webcam_pts:
        mode_raw, tier_count = classify_mode_dual(
            pts_left,
            pts_left_webcam,
            morph_mode=int(mode_state.morph_mode),
            orbbec_vis_min=hand_visibility_min,
            mode_vis_min=vmin,
            rotating=bool(rotating),
            classify_mode_fn=classify_mode_fn,
            classify_webcam_fn=classify_mode_from_fingers_webcam_image,
            orbbec_thumb_vis=orbbec_thumb_vis,
        )
    else:
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
        next_mode = int(mode_raw)
        mode_state.mode_ema = float(next_mode)
        mode_state.morph_mode = next_mode
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
    topology_analysis: Optional[Dict[str, Any]] = None,
    snap_soft_k: float = 0.34,
    snap_soft_max_step: float = 0.16,
    follow_k: float = 0.45,
    follow_max_step: float = 0.20,
) -> Optional[float]:
    if pts_right is None:
        return right_state.last_open_out

    tmp = topology_analysis if topology_analysis is not None else analyze_topology_fn(pts_right)
    if tmp is None:
        return right_state.last_open_out

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

    prev_open = right_state.last_open_out
    open_out = open_free if prev_open is None else float(prev_open)
    if right_state.snap_state == "plane":
        target = 1.0
        delta = float(target - open_out)
        step = float(snap_soft_k) * delta
        max_step = max(1e-4, float(snap_soft_max_step))
        step = float(np.clip(step, -max_step, max_step))
        open_out = float(np.clip(open_out + step, 0.0, 1.0))
    elif right_state.snap_state == "sphere":
        target = 0.0
        delta = float(target - open_out)
        step = float(snap_soft_k) * delta
        max_step = max(1e-4, float(snap_soft_max_step))
        step = float(np.clip(step, -max_step, max_step))
        open_out = float(np.clip(open_out + step, 0.0, 1.0))
    else:
        # Keep continuity when leaving snap hysteresis region: quickly follow
        # open_free, but never jump in one frame.
        delta = float(open_free - open_out)
        step = float(follow_k) * delta
        max_step = max(1e-4, float(follow_max_step))
        step = float(np.clip(step, -max_step, max_step))
        open_out = float(np.clip(open_out + step, 0.0, 1.0))
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
    topo_radius_override_mm: float | None = None,
):
    src = "MediaPipe" if hand_3d_source == "mp" else "depth+MP fused"
    title = f"Hand 3D ({src}) joints 0..20"
    if control_label:
        title += f" | {control_label}"
    if ax_hand is not None:
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

        if ax_hand is not None:
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
            if topo_radius_override_mm is not None and float(topo_radius_override_mm) > 0.0:
                r_draw = float(topo_radius_override_mm)
            else:
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

        morph_alpha = analysis["morph_alpha"] if morph_alpha_smoothed is None else morph_alpha_smoothed
        if topo_radius_override_mm is not None and float(topo_radius_override_mm) > 0.0:
            r_vis = float(topo_radius_override_mm)
        else:
            r = max(analysis["radius"], 1.0)
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

    if not analyses:
        morph_fb = 0.55 if morph_alpha_smoothed is None else float(morph_alpha_smoothed)
        morph_fb = clamp01_fn(morph_fb)
        if topo_radius_override_mm is not None and float(topo_radius_override_mm) > 0.0:
            r_draw = float(topo_radius_override_mm)
        else:
            r_draw = 200.0
        draw_fn = draw_mode1_fn if morph_mode == 1 else draw_mode2_fn if morph_mode == 2 else draw_mode3_fn
        draw_fn(ax_topo, radius=r_draw, open_alpha=morph_fb, show_refs=True)
        ax_topo.text(
            -r_draw,
            -r_draw,
            r_draw * 0.92,
            f"open={morph_fb:.2f}\n(no hand — morph only)",
            color="tab:gray",
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
    *,
    mp_result=None,
    mode_vis_min: float = 0.0,
    hold_mode: bool = False,
    debounce_frames: int | None = None,
    webcam_mp_result=None,
    webcam_idx_left: Optional[int] = None,
    dual_mode_assist: bool = False,
    rotating: bool = False,
    orbbec_thumb_vis: float | None = None,
) -> Tuple[int, int]:
    """Update left-hand mode state and return (mode_raw, tier_count)."""
    from functions.dual_cam.mp_hand_utils import extract_image_plane_points_mm_result
    from functions.swarm_motion.left_hand_swarm_pose import mp_hand_visibility_scores

    pts_left = None
    if idx_left is not None and idx_left < len(keypoints_3d):
        pts_left = keypoints_3d[idx_left]
    vis_min = None
    if mp_result is not None and idx_left is not None and float(mode_vis_min) > 0.0:
        _, vis_min = mp_hand_visibility_scores(mp_result, int(idx_left))
    pts_webcam = None
    if webcam_mp_result is not None and webcam_idx_left is not None:
        pts_webcam = extract_image_plane_points_mm_result(
            webcam_mp_result, int(webcam_idx_left)
        )
    db = int(debounce_frames) if debounce_frames is not None else int(MODE_DEBOUNCE_FRAMES)
    return update_mode_state(
        pts_left,
        mode_state=mode_state,
        classify_mode_fn=classify_mode_from_fingers,
        debounce_frames=db,
        mode_smooth=MODE_SMOOTH,
        mode_vis_min=float(mode_vis_min),
        hand_visibility_min=vis_min,
        hold_mode=bool(hold_mode),
        pts_left_webcam=pts_webcam,
        dual_mode_assist=bool(dual_mode_assist),
        rotating=bool(rotating),
        orbbec_thumb_vis=orbbec_thumb_vis,
    )


def process_right_open(
    keypoints_3d: List,
    idx_right: Optional[int],
    right_state: RightHandState,
    *,
    mp_result=None,
    open_vis_min: float = 0.0,
) -> Tuple[List, Optional[float]]:
    """Update right-hand open state and return (hands_3d_for_plot, open_out)."""
    from functions.swarm_motion.left_hand_swarm_pose import mp_hand_visibility_scores

    hands_3d: List = []
    open_out: Optional[float] = None
    if idx_right is not None and idx_right < len(keypoints_3d):
        vis_min = None
        if mp_result is not None and float(open_vis_min) > 0.0:
            _, vis_min = mp_hand_visibility_scores(mp_result, int(idx_right))
            if vis_min is not None and float(vis_min) < float(open_vis_min):
                if right_state.last_right_pts is not None:
                    hands_3d = [right_state.last_right_pts]
                return hands_3d, right_state.last_open_out
        pts_right = keypoints_3d[idx_right]
        right_state.last_right_pts = list(pts_right)
        hands_3d = [pts_right]
        open_out = update_open_state(
            pts_right,
            right_state=right_state,
            analyze_topology_fn=analyze_hand_topology,
            open_smooth=OPEN_SMOOTH,
            plane_snap_on=PLANE_SNAP_ON,
            plane_snap_off=PLANE_SNAP_OFF,
            sphere_snap_on=SPHERE_SNAP_ON,
            sphere_snap_off=SPHERE_SNAP_OFF,
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
    update_snap_visual_state(
        snap_state,
        snap_visual_state=snap_vis_state,
        snap_show_after_frames=SNAP_SHOW_AFTER_FRAMES,
        snap_hold_after_release_frames=SNAP_HOLD_AFTER_RELEASE_FRAMES,
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
    a0 = analyses[0]
    topo_lbl = topology_label_from_morph_alpha(
        float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"])
    )
    open_disp = open_out if open_out is not None else a0["morph_alpha"]
    free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
    if open_remap is not None:
        lo_r, hi_r = open_remap
        open_disp = remap_open_display(open_disp, lo_r, hi_r)
        free_disp = remap_open_display(free_disp, lo_r, hi_r)

    need_refresh = (frame_idx % HUD_UPDATE_EVERY_N_FRAMES) == 0 or runtime.hud_cache["open"] is None
    if not need_refresh:
        if abs(float(open_disp) - float(runtime.hud_cache["open"])) > HUD_OPEN_STEP:
            need_refresh = True
        if abs(float(free_disp) - float(runtime.hud_cache["free"])) > HUD_OPEN_STEP:
            need_refresh = True
        if abs(float(a0["planarity"]) - float(runtime.hud_cache["plan"])) > HUD_METRIC_STEP:
            need_refresh = True
        if abs(float(a0["isotropy"]) - float(runtime.hud_cache["iso"])) > HUD_METRIC_STEP:
            need_refresh = True
        if abs(float(a0["finger_spread"]) - float(runtime.hud_cache["spread"])) > HUD_METRIC_STEP:
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
    if hand_frame == HAND_FRAME_PALM_PLANE and hands_3d:
        cm = palm_plane_curl_metrics(hands_3d[0])
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

