"""Orbbec live-loop helpers: open EMA + SNAP, HUD cache refresh, periodic topology print, SNAP visual state."""

from __future__ import annotations


def compute_open_out(
    *,
    hands_3d,
    open_free_ema,
    snap_state,
    alpha_smooth: float,
    analyze_topology_fn,
    plane_snap_on: float,
    plane_snap_off: float,
    sphere_snap_on: float,
    sphere_snap_off: float,
):
    open_out = None
    if not hands_3d:
        return open_out, open_free_ema, snap_state
    tmp = analyze_topology_fn(hands_3d[0])
    if tmp is None:
        return open_out, open_free_ema, snap_state

    if open_free_ema is None:
        open_free_ema = float(tmp["morph_alpha"])
    else:
        open_free_ema = alpha_smooth * float(tmp["morph_alpha"]) + (1.0 - alpha_smooth) * open_free_ema

    open_free = float(open_free_ema)
    if snap_state == "plane":
        if open_free < plane_snap_off:
            snap_state = None
    elif snap_state == "sphere":
        if open_free > sphere_snap_off:
            snap_state = None
    else:
        if open_free > plane_snap_on:
            snap_state = "plane"
        elif open_free < sphere_snap_on:
            snap_state = "sphere"

    open_out = open_free
    if snap_state == "plane":
        open_out = 1.0
    elif snap_state == "sphere":
        open_out = 0.0
    return open_out, open_free_ema, snap_state


def update_snap_visual_state(
    *,
    snap_state,
    snap_vis_state,
    snap_stable_frames: int,
    snap_hold_frames: int,
    snap_show_after_frames: int,
    snap_hold_after_release_frames: int,
):
    if snap_state is None:
        snap_stable_frames = 0
        if snap_vis_state is not None:
            snap_hold_frames += 1
            if snap_hold_frames >= snap_hold_after_release_frames:
                snap_vis_state = None
                snap_hold_frames = 0
    else:
        snap_hold_frames = 0
        if snap_vis_state == snap_state:
            snap_stable_frames = min(snap_show_after_frames, snap_stable_frames + 1)
        else:
            snap_stable_frames += 1
            if snap_stable_frames >= snap_show_after_frames:
                snap_vis_state = snap_state
                snap_stable_frames = 0
    return snap_vis_state, snap_stable_frames, snap_hold_frames


def refresh_hud_cache(
    *,
    hud_cache: dict,
    frame_idx: int,
    a0: dict,
    hands_3d,
    hand_frame: str,
    hand_frame_palm_plane: str,
    open_out,
    open_free_ema,
    open_remap,
    snap_vis_state,
    hud_update_every_n_frames: int,
    hud_open_step: float,
    hud_metric_step: float,
    topology_label_fn,
    remap_open_display_fn,
    palm_plane_curl_metrics_fn,
):
    topo_lbl = topology_label_fn(float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"]))
    open_disp = open_out if open_out is not None else a0["morph_alpha"]
    free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
    if open_remap is not None:
        lo_r, hi_r = open_remap
        open_disp = remap_open_display_fn(open_disp, lo_r, hi_r)
        free_disp = remap_open_display_fn(free_disp, lo_r, hi_r)

    need_refresh = (frame_idx % hud_update_every_n_frames) == 0 or hud_cache["open"] is None
    if not need_refresh:
        if abs(float(open_disp) - float(hud_cache["open"])) > hud_open_step:
            need_refresh = True
        if abs(float(free_disp) - float(hud_cache["free"])) > hud_open_step:
            need_refresh = True
        if abs(float(a0["planarity"]) - float(hud_cache["plan"])) > hud_metric_step:
            need_refresh = True
        if abs(float(a0["isotropy"]) - float(hud_cache["iso"])) > hud_metric_step:
            need_refresh = True
        if abs(float(a0["finger_spread"]) - float(hud_cache["spread"])) > hud_metric_step:
            need_refresh = True
        if hand_frame == hand_frame_palm_plane and hands_3d:
            cm = palm_plane_curl_metrics_fn(hands_3d[0])
            curl_s = None
            if cm and cm.get("mean_r_xy_four") is not None:
                tr = cm.get("thumb_r_xy")
                thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
                curl_s = f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}{thumb_s}"
            if curl_s != hud_cache.get("curl"):
                need_refresh = True

    if not need_refresh:
        return

    hud_cache["open"] = float(open_disp)
    hud_cache["free"] = float(free_disp)
    hud_cache["plan"] = float(a0["planarity"])
    hud_cache["iso"] = float(a0["isotropy"])
    hud_cache["spread"] = float(a0["finger_spread"])
    snap_txt = f"  SNAP:{snap_vis_state.upper()}" if snap_vis_state is not None else ""
    lines = [
        f"Topo:{topo_lbl}{snap_txt}",
        f"open:{open_disp:.2f}  free:{free_disp:.2f}",
        f"spread:{a0['finger_spread']:.2f}  plan:{a0['planarity']:.2f}  iso:{a0['isotropy']:.2f}",
    ]
    if hand_frame == hand_frame_palm_plane and hands_3d:
        cm = palm_plane_curl_metrics_fn(hands_3d[0])
        if cm and cm.get("mean_r_xy_four") is not None:
            tr = cm.get("thumb_r_xy")
            thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
            curl_s = f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}{thumb_s}"
            hud_cache["curl"] = curl_s
            lines.append(curl_s)
        else:
            hud_cache["curl"] = None
    else:
        hud_cache["curl"] = None
    hud_cache["text"] = lines


def print_periodic_topology_status(
    *,
    frame_idx: int,
    a0: dict,
    hands_3d,
    hand_frame: str,
    hand_frame_palm_plane: str,
    open_out,
    open_free_ema,
    open_remap,
    topology_label_fn,
    remap_open_display_fn,
    palm_plane_curl_metrics_fn,
):
    if frame_idx % 30 != 0:
        return
    out_v = open_out if open_out is not None else a0["morph_alpha"]
    free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
    topo_print = topology_label_fn(float(free_v))
    if open_remap is not None:
        lo_r, hi_r = open_remap
        out_show = remap_open_display_fn(out_v, lo_r, hi_r)
        free_show = remap_open_display_fn(free_v, lo_r, hi_r)
        open_part = f"open={out_show:.3f} raw={out_v:.3f} free={free_show:.3f} raw_free={free_v:.3f}"
    else:
        open_part = f"open_out={out_v:.3f} free={free_v:.3f}"
    curl_part = ""
    if hand_frame == hand_frame_palm_plane and hands_3d:
        cm = palm_plane_curl_metrics_fn(hands_3d[0])
        if cm and cm.get("mean_r_xy_four") is not None:
            tr = cm.get("thumb_r_xy")
            thumb_p = f" thumb_r={tr:.3f}" if tr is not None else ""
            curl_part = f" curl_rxy4={cm['mean_r_xy_four']:.3f} curl_z4={cm['mean_abs_z_four']:.3f}{thumb_p}"
    print(
        "topology="
        f"{topo_print} inst={a0['morph_alpha']:.3f} "
        f"{open_part} "
        f"spread={a0['finger_spread']:.3f} "
        f"planarity={a0['planarity']:.3f} "
        f"isotropy={a0['isotropy']:.3f}"
        f"{curl_part}"
    )

