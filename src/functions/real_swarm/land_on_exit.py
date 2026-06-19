"""Axswarm-filtered exit landing for physical Crazyflies (no high-level goto)."""

from __future__ import annotations

import time

import numpy as np

from functions.mode_switch.online_frame_gesture import GestureFrameResult
from functions.runtime.online_boot import OnlineBoot, sync_armed_flags
from functions.runtime.online_runtime_config import OnlineRuntimeConfig
from functions.swarm_motion.online_frame_filter import filter_online_targets
from functions.swarm_motion.spacing_guard import enforce_min_separation_xy


def _idle_gesture_result(*, frame_idx: int = 0) -> GestureFrameResult:
    return GestureFrameResult(
        idx_l=None,
        idx_r=None,
        hands_3d=None,
        hands_3d_all=[],
        open_out=None,
        mode_raw="-",
        tier_count=-1,
        pts_l=None,
        pts_l_pose_mm=None,
        palm_center_depth_mm=None,
        palm_center_color_px=None,
        mph=0,
        mpw=0,
        orbbec_vis_min_now=None,
        prefetch_B=None,
        prefetch_res=None,
        prefetch_wfr=None,
        webcam_frame_idx=int(frame_idx),
    )


def stream_real_swarm_land_on_exit(
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    *,
    max_duration_s: float = 45.0,
) -> None:
    """Stream ``ground_layout`` through axswarm until near home.

    Caller must ``real_executor.close()`` (``swarm.close()``) in a ``finally`` block (see ``online_control``).
    """
    ex = boot.real_executor
    if ex is None:
        return
    if not ex.opts.land_on_exit or not ex.physical_armed:
        return

    print("Real swarm axswarm-filtered exit landing to ground ...")
    boot.prearm_phase_box[0] = "ground"
    boot.prearm_has_flown_box[0] = True
    sync_armed_flags(boot)

    elapsed0 = time.monotonic() - boot.start_time
    if boot.axswarm_rt is not None:
        boot.axswarm_rt.enter_recover(elapsed0, hold_s=10.0)
        track0 = ex.get_sim_track_positions(boot.prev_cmd_target, boot.n_drones)
        pos0 = track0 if track0 is not None else boot.prev_cmd_target
        boot.axswarm_rt.sync_gesture(
            np.asarray(pos0, dtype=np.float32),
            np.zeros((boot.n_drones, 3), dtype=np.float32),
        )

    gest = _idle_gesture_result(frame_idx=boot.frame_idx)
    land_tol = float(ex.opts.max_pos_error_m)
    period_s = 1.0 / max(float(cfg.fps), 1.0)
    t_start = time.monotonic()

    while time.monotonic() - t_start < float(max_duration_s):
        if not ex.mocap_ok():
            time.sleep(period_s)
            continue

        elapsed = time.monotonic() - boot.start_time
        raw_target = enforce_min_separation_xy(
            boot.ground_layout.copy(),
            float(cfg.min_separation_m),
            float(boot.ground_z),
            iters=10,
        )
        track_pos = ex.get_sim_track_positions(boot.prev_cmd_target, boot.n_drones)
        filt, boot.raw_target_filt, boot.prev_open_for_snap, boot.prev_gesture_control_enabled = (
            filter_online_targets(
                boot=boot,
                cfg=cfg,
                gest=gest,
                raw_target=raw_target,
                morph_targets_before_left_m=raw_target,
                elapsed=elapsed,
                track_pos=track_pos,
            )
        )
        boot.prev_cmd_target = filt.cmd_target.copy()
        ex.send_sim_layout(filt.cmd_target)

        if track_pos is not None:
            phys = np.asarray(track_pos[: ex.n_physical], dtype=np.float32)
            goal = np.asarray(boot.ground_layout[: ex.n_physical], dtype=np.float32)
            err = float(np.max(np.linalg.norm(phys - goal, axis=1)))
            if err <= land_tol:
                print(f"Exit land: within {err:.2f}m of ground (tol {land_tol:.2f}m).")
                break
        time.sleep(period_s)
    else:
        print("[WARN] Exit axswarm land timed out.")


def try_stream_real_swarm_land_on_exit(
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    *,
    max_duration_s: float = 45.0,
) -> None:
    """Best-effort exit land (swarmGPT ``try`` body); errors are logged, not re-raised."""
    try:
        stream_real_swarm_land_on_exit(boot, cfg, max_duration_s=max_duration_s)
    except Exception as exc:
        print(f"[WARN] Real swarm exit landing failed: {exc}")
