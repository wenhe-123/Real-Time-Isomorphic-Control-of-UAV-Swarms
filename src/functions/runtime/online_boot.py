"""Startup: Orbbec, Crazyflow sim, prearm, axswarm, hotkeys (before main loop)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyk4a import Config, FPS, PyK4A

from crazyflow.control import Control
from crazyflow.sim import Sim
from functions.dual_cam.online_input_keys import (
    OnlineKeyContext,
    OnlineKeyQueue,
    format_hotkey_install_hint,
    probe_global_hotkey_backends,
    process_online_control_keys,
    try_install_hotkey_dependencies,
)
from functions.dual_cam.mp_hand_utils import orbbec_resolve_swap_mp_hands
from functions.display_sim.online_plot import init_3d_plot
from functions.display_sim.gesture_report_debug import (
    ReportDebugFigures,
    ReportDebugPanels,
    close_report_debug_figures,
    init_report_debug_figures,
)
from functions.mode_switch.modes_runtime import ModeState, RightHandState
from functions.mode_switch.morph_shape_control import LpShapePipelineState
from functions.open_close.morph_world import ScaleConfig, fixed_morph_points, normalize_morph_points_at_hover
from functions.runtime.live_target import LiveTargetState
from functions.runtime.online_runtime_config import OnlineRuntimeConfig
from functions.runtime.online_defaults import (
    _DEFAULT_GROUND_Z,
    _ONLINE_ORBBEC_FPS,
    _TRAIL_BUFFER_MAXLEN,
)
from functions.runtime.pipeline_tuning import PipelineTuning
from functions.swarm_motion.axswarm_runtime import AxswarmOnlineConfig, AxswarmSafetyFilter, resolve_axswarm_settings_path
from functions.swarm_motion.left_hand_swarm_pose import (
    LeftSwarmPoseState,
    left_cam_preset_rotation,
    make_cam_translation_matrix,
    palm_basis_pair_indices,
)
from functions.swarm_motion.left_pose_tuning import LeftPoseTuning
from functions.swarm_motion.prearm import (
    sim_chessboard_ground_layout,
    vertical_takeoff_layout,
)
from functions.swarm_motion.spacing_guard import closest_pair

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from functions.real_swarm.executor import RealSwarmExecutor


@dataclass
class OnlineBoot:
    k4a: PyK4A
    calib: Any
    sim: Sim | None
    real_executor: RealSwarmExecutor | None
    n_drones: int
    axswarm_rt: AxswarmSafetyFilter
    mode_state: ModeState
    right_state: RightHandState
    lp_shape: LpShapePipelineState
    live_target: LiveTargetState
    scale: ScaleConfig
    pipe: PipelineTuning
    use_depth_fusion: bool
    cmd_target: np.ndarray
    prev_cmd_target: np.ndarray
    raw_target_filt: np.ndarray
    ground_layout: np.ndarray
    ground_z: float
    prearm_hover_layout: np.ndarray
    prearm_hover_z: float
    prearm_vertical_layout: np.ndarray
    prearm_takeoff_z: float
    trail_rgba: list
    pos_buffer: deque
    center_trace_prev: dict[str, float | None]
    plot_enabled: bool
    fig: Any
    ax_hand: Any
    ax_topo: Any
    left_pose_state: LeftSwarmPoseState
    key_queue: OnlineKeyQueue
    gesture_control_enabled_box: list[bool]
    prearm_climb_enabled_box: list[bool]
    prearm_phase_box: list[str]
    prearm_vertical_leg_box: list[str]
    prearm_has_flown_box: list[bool]
    left_pose_reset_req_box: list[bool]
    left_pose_runtime_armed_box: list[bool]
    orbbec_swap_mp_hands: bool
    left_use_camera_at_arm: bool
    left_M_rot: np.ndarray | None
    left_M_trans: np.ndarray | None
    left_cam_preset: str
    left_palm_basis: str
    left_rot_pivot_key: str
    left_dual_webcam_rot_eff: bool
    left_rot_webcam_vis_thresh: float
    left_unwind_s: float
    left_pose_tuning: LeftPoseTuning
    start_time: float
    render_enabled: bool = True
    orbbec_flip_depth_warned: bool = False
    prev_open_for_snap: float | None = None
    gesture_control_enabled: bool = False
    prev_gesture_control_enabled: bool = False
    prearm_climb_enabled: bool = False
    prev_prearm_climb_enabled: bool = False
    prearm_phase: str = "ground"
    prev_prearm_phase: str = "ground"
    prearm_vertical_leg: str = "climb"
    prearm_has_flown: bool = False
    left_pose_reset_req: bool = False
    left_pose_runtime_armed: bool = False
    frame_idx: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def print_pipeline_mode(pipe: PipelineTuning, *, draw_hand_debug: bool, mp_delegate_key: str) -> None:
    print(f"MediaPipe HandLandmarker delegate: {mp_delegate_key} (GPU falls back to CPU if unavailable)")
    if draw_hand_debug:
        print("Orbbec hand overlay: ON (--draw-hand-debug).")
    elif pipe.depth_fusion_enabled:
        print(
            f"Orbbec hand 3D: depth fusion ON (weight={pipe.depth_fusion_weight:.2f}, "
            f"mode_debounce={pipe.mode_debounce_frames})."
        )
    else:
        print(
            "Orbbec hand 3D: MediaPipe world only (no per-joint depth fusion in draw_hand). "
            "Depth still used for L-move via left_hand_pose_matrix_depth_mm. "
            "Use --draw-hand-debug for skeleton on preview; --debug-webcam-pipeline for demo tuning."
        )


def boot_online_control(
    *,
    live_target: LiveTargetState,
    cfg: OnlineRuntimeConfig,
) -> OnlineBoot:
    """Open Orbbec + sim or real swarm, run prearm, wire axswarm and hotkeys."""
    n_drones = int(cfg.point_count)
    scale = cfg.scale
    orbbec_fps = FPS.FPS_15 if _ONLINE_ORBBEC_FPS == "15" else FPS.FPS_30
    k4a = PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=orbbec_fps,
        )
    )
    k4a.start()
    calib = k4a.calibration

    n_drones = int(cfg.point_count)
    real_executor: RealSwarmExecutor | None = None
    sim: Sim | None = None
    motion_freq_hz = 100.0

    if cfg.drones_config is not None:
        from functions.real_swarm.executor import RealSwarmExecutor

        real_executor = RealSwarmExecutor(
            config_path=Path(cfg.drones_config),
            morph_point_count=n_drones,
        )
        motion_freq_hz = float(real_executor.ctrl_freq)
        print("Real-swarm mode: Crazyflow MuJoCo disabled; cmd_target → Crazyflie setpoints.")
    else:
        sim = Sim(
            n_worlds=1,
            n_drones=n_drones,
            control=Control.state,
            drone_model=str(cfg.drone_model),
        )
        sim.reset()
        motion_freq_hz = float(sim.freq)

    center_trace_every = max(1, int(cfg.center_trace_every))
    center_trace_prev: dict[str, float | None] = {
        "hand_z": None,
        "raw_z": None,
        "safe_z": None,
        "smooth_z": None,
    }
    if cfg.center_trace:
        print(
            "Center trace: hand (cam mm), hand_rel_m, raw/safe/smooth target centroids, sim; "
            f"flags depth-jump & target-jump every {center_trace_every} frame(s)."
        )

    zeros = jnp.zeros((n_drones, 3))

    axswarm_online = AxswarmOnlineConfig.from_runtime_config(cfg, scale)
    axswarm_rt = AxswarmSafetyFilter.from_online_config(n_drones, axswarm_online)
    yaml_path = resolve_axswarm_settings_path(
        axswarm_online.settings_path,
        axswarm_online.project_root,
    )
    env = np.asarray(axswarm_rt.settings.collision_envelope, dtype=np.float64)
    pos_lo = np.asarray(axswarm_rt._pos_lo, dtype=np.float64)
    pos_hi = np.asarray(axswarm_rt._pos_hi, dtype=np.float64)
    print(
        f"Axswarm yaml ({yaml_path}): freq={axswarm_rt.mpc_hz:.1f} Hz, "
        f"vel_max={float(axswarm_rt.settings.vel_max):.2f} m/s, "
        f"acc_max={float(axswarm_rt.settings.acc_max):.2f} m/s², "
        f"collision_env={env.tolist()}, min_sep={axswarm_rt.min_separation_m:.2f} m, "
        f"pos=[{pos_lo.tolist()}, {pos_hi.tolist()}]."
    )
    print(
        f"Planner: gesture setpoints + axswarm safety filter @ {axswarm_rt.mpc_hz:.1f} Hz, "
        f"n={n_drones} (active from startup)."
    )

    z_ground = float(_DEFAULT_GROUND_Z)
    if real_executor is not None:
        ground_layout = real_executor.get_sim_ground_layout(
            n_drones, min_separation_m=float(cfg.min_separation_m)
        )
        z_ground = float(np.median(ground_layout[: real_executor.n_physical, 2]))
        print(
            f"Startup layout: TOML home positions (sim frame, z≈{z_ground:.2f}m). "
            "Press 1: vertical takeoff → formation → vertical descend → ground."
        )
    else:
        ground_layout = sim_chessboard_ground_layout(
            n_drones,
            min_separation_m=float(cfg.min_separation_m),
            z_ground=z_ground,
            xy_half_extent_m=float(scale.xy_radius) * 0.85,
        )
        d_g, pi_g, pj_g = closest_pair(ground_layout)
        print(
            f"Startup layout: chessboard ground (z={z_ground:.2f}m, "
            f"spacing=({pi_g},{pj_g}) {d_g:.2f}m). "
            "Press 1: vertical takeoff → formation → vertical descend → ground."
        )
    prearm_hover_z = float(
        np.clip(float(cfg.prearm_hover_z), float(z_ground) + 0.15, 10.0)
    )
    scale = replace(
        scale,
        hover_z=prearm_hover_z,
        z_center=prearm_hover_z,
        z_min=prearm_hover_z - float(scale.z_amplitude),
        z_max=prearm_hover_z + float(scale.z_amplitude),
    )
    _hover_mm = fixed_morph_points(
        n_drones,
        float(cfg.morph_radius_mm),
        int(live_target.mode),
        float(live_target.open_alpha),
        None,
    )
    prearm_hover_layout = normalize_morph_points_at_hover(
        _hover_mm,
        scale,
    )
    live_target.set(
        prearm_hover_layout,
        int(live_target.mode),
        float(live_target.open_alpha),
    )
    d_hold, pi_h, pj_h = closest_pair(prearm_hover_layout)
    _layout_c = prearm_hover_layout.mean(axis=0)
    _mm_c = _hover_mm.mean(axis=0)
    print(
        f"Pre-SPACE: morph layout → hover z={prearm_hover_z:.2f}m, "
        f"sim_centroid=({_layout_c[0]:+.2f},{_layout_c[1]:+.2f},{_layout_c[2]:+.2f})m "
        f"z∈[{float(prearm_hover_layout[:, 2].min()):.2f},"
        f"{float(prearm_hover_layout[:, 2].max()):.2f}]m "
        f"(mm sample centroid=({_mm_c[0]:+.1f},{_mm_c[1]:+.1f},{_mm_c[2]:+.1f})mm), "
        f"spacing=({pi_h},{pj_h}) {d_hold:.2f}m"
    )
    takeoff_z_lo = float(z_ground) + 0.02
    takeoff_z_hi = max(takeoff_z_lo + 0.02, float(prearm_hover_z) - 0.02)
    prearm_takeoff_z = float(
        np.clip(float(cfg.prearm_takeoff_z), takeoff_z_lo, takeoff_z_hi)
    )
    prearm_vertical_layout = vertical_takeoff_layout(
        ground_layout,
        takeoff_z=prearm_takeoff_z,
        min_separation_m=float(cfg.min_separation_m),
    )
    if real_executor is not None:
        n_phys = int(real_executor.n_physical)
        prearm_vertical_layout[:n_phys, :2] = ground_layout[:n_phys, :2]
        prearm_vertical_layout[:n_phys, 2] = prearm_takeoff_z
    d_v, pi_v, pj_v = closest_pair(prearm_vertical_layout)
    if real_executor is not None:
        print(
            f"Vertical takeoff target: z={prearm_takeoff_z:.2f}m "
            f"(TOML home XY fixed for {real_executor.n_physical} drone(s), "
            f"spacing=({pi_v},{pj_v}) {d_v:.2f}m)."
        )
    else:
        print(
            f"Vertical takeoff target: z={prearm_takeoff_z:.2f}m (ground XY fixed, "
            f"spacing=({pi_v},{pj_v}) {d_v:.2f}m)."
        )
    boot_cmd = ground_layout.copy()
    if real_executor is not None:
        track0 = real_executor.get_sim_track_positions(ground_layout, n_drones)
        if track0 is not None:
            boot_cmd = track0
    if sim is not None:
        sim.data = sim.data.replace(
            states=sim.data.states.replace(
                pos=jnp.asarray(ground_layout[None, :, :], device=sim.device),
                vel=zeros[None, :, :],
                ang_vel=zeros[None, :, :],
            )
        )
    axswarm_rt.reset(boot_cmd, np.zeros((n_drones, 3), dtype=np.float32))
    mode_label = "real Crazyflie" if real_executor is not None else "MuJoCo sim (cmd+step)"
    print(
        f"Mode: gesture targets → axswarm filter → cmd_target; {mode_label}. "
        "Planner: Axswarm safety_filter."
    )
    print(
        f"Axswarm safety filter active from startup (ground hold_z={z_ground:.2f}m). "
        f"Press 1: z={prearm_takeoff_z:.2f}m → formation z={prearm_hover_z:.2f}m (before SPACE) "
        f"→ z={prearm_takeoff_z:.2f}m → ground; SPACE → gesture control."
    )

    left_cam_preset = str(cfg.left_cam_preset).strip().lower()
    left_cam_y_to_world_z = float(np.clip(cfg.left_cam_y_to_world_z, 0.0, 1.0))
    left_palm_basis = str(cfg.left_palm_basis).strip().lower()
    palm_basis_pair_indices(left_palm_basis)
    left_cam_motion = bool(cfg.left_swarm_depth_frame_motion) and calib is not None
    left_world_frame_key = str(cfg.left_world_frame).strip().lower()
    left_use_camera_at_arm = left_world_frame_key == "camera_at_arm" and left_cam_motion
    left_M_rot = left_cam_preset_rotation(left_cam_preset) if left_cam_motion else None
    left_M_trans = (
        make_cam_translation_matrix(left_M_rot, image_y_to_world_z=left_cam_y_to_world_z)
        if left_M_rot is not None
        else None
    )
    orbbec_swap_mp_hands = orbbec_resolve_swap_mp_hands(
        hand_swap=cfg.orbbec_hand_swap,
        flip_horizontal=bool(cfg.orbbec_flip_horizontal),
        use_orbbec=True,
    )

    panels = cfg.report_panels or ReportDebugPanels()
    report_debug_figs: ReportDebugFigures | None = None
    if panels.any_enabled() and int(cfg.plot_every_n) > 0:
        report_debug_figs = init_report_debug_figures(panels)
        plot_enabled = True
        fig = report_debug_figs.fig_morph
        ax_topo = report_debug_figs.ax_morph
        ax_hand = None
        print(
            f"Debug report panels: {', '.join(panels.enabled_labels()) or 'none'} "
            f"(plot-every={cfg.plot_every_n}; use one panel at a time for screenshots)."
        )
    else:
        plot_enabled, fig, ax_hand, ax_topo = init_3d_plot(cfg.plot_every_n, "Online Control Orbbec + 3D")
    print("Orbbec input started. Left hand = MODE, right hand = OPEN. Press q/Enter to stop.")
    if cfg.orbbec_flip_horizontal:
        print(
            "Orbbec horizontal flip is ON (ego view vs mirror). "
            "Use --no-orbbec-flip-horizontal if depth/3D looks wrong."
        )
    if cfg.orbbec_use_transformed_depth:
        print(
            "Orbbec transformed_depth is ON (K4A alignment). If the process aborts, use "
            "--no-orbbec-use-transformed-depth (default off for Femto Bolt)."
        )
    if orbbec_swap_mp_hands:
        print(
            "Orbbec: swapping MediaPipe left/right for mode vs open hand "
            f"(policy {str(cfg.orbbec_hand_swap).strip().lower()!r}; auto follows horizontal flip). "
            "Override: --orbbec-hand-swap off | on | auto."
        )
    if left_cam_motion:
        wf = (
            f"world_frame={left_world_frame_key!r} (cam→sim map locked at press 0)"
            if left_use_camera_at_arm
            else f"world_frame={left_world_frame_key!r} (cam→sim map from CLI each frame)"
        )
        print(
            f"Left-swarm depth: cam→sim preset={left_cam_preset!r} "
            f"(camera|legacy|fwd_y|flip_depth), {wf}, "
            f"palm_basis={left_palm_basis!r}."
        )
    print(
        "Ground hold until 1; 1×4 cycle: climb → formation → descend → ground; "
        "SPACE arms/disarms gesture control."
    )

    left_pose_state = LeftSwarmPoseState(enabled=bool(cfg.left_swarm_pose))
    left_rot_pivot_key = str(cfg.left_rot_pivot).strip().lower()
    if left_rot_pivot_key not in ("per_drone", "centroid"):
        left_rot_pivot_key = "per_drone"
    left_dual_webcam_rot_eff = bool(cfg.left_dual_webcam_rot)
    left_rot_webcam_vis_thresh = float(np.clip(cfg.left_rot_webcam_vis_thresh, 0.05, 0.99))
    left_pose_tuning = LeftPoseTuning.from_config(
        cfg, direct_follow=bool(cfg.left_rot_direct_follow)
    )
    if left_pose_state.enabled:
        print(
            "Left-hand whole formation: press 0 to START (zero pose = current hand; "
            "current palm center + pose are tracked relative to press-0), "
            f"0 again to restore morph frame (~{float(cfg.left_unwind_s):.1f}s)."
        )
        if bool(cfg.left_rot_direct_follow):
            print(
                "Left rotation direct-follow: palm-local axes drive swarm axes "
                "(palm normal twist -> world Z yaw); planar/tau damping disabled."
            )
    gesture_control_enabled = False
    _hk_probe = (
        try_install_hotkey_dependencies()
        if cfg.install_hotkey_deps
        else probe_global_hotkey_backends()
    )
    key_queue = OnlineKeyQueue()
    if cfg.global_hotkeys:
        key_queue.start(use_global=True, use_stdin=True)
        _hk = key_queue.mode
        if "pynput" not in _hk and "keyboard" not in _hk:
            print("[WARN] Global hotkeys not available in this Python environment.")
            print(format_hotkey_install_hint(_hk_probe))
        elif _hk == "off":
            print("[WARN] No hotkey backends started.")
            print(format_hotkey_install_hint(_hk_probe))
        else:
            print(
                f"Hotkeys: {_hk} — 1 climb→formation→descend→ground, SPACE arm/disarm, "
                "0/q without Orbbec focus."
            )
        print(
            "Left control: palm_centroid — press-0 palm center + palm basis; "
            f"frozen cam→sim preset {left_cam_preset!r} when camera_at_arm."
        )

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))
    trail_rgba = [colors[d].tolist() for d in range(n_drones)]

    return OnlineBoot(
        k4a=k4a,
        calib=calib,
        sim=sim,
        real_executor=real_executor,
        n_drones=n_drones,
        axswarm_rt=axswarm_rt,
        mode_state=ModeState(),
        right_state=RightHandState(),
        lp_shape=LpShapePipelineState(),
        live_target=live_target,
        scale=scale,
        pipe=cfg.pipe,
        use_depth_fusion=bool(cfg.pipe.depth_fusion_enabled),
        cmd_target=boot_cmd.copy(),
        prev_cmd_target=boot_cmd.copy(),
        raw_target_filt=boot_cmd.copy(),
        ground_layout=ground_layout.copy(),
        ground_z=z_ground,
        prearm_hover_layout=prearm_hover_layout.copy(),
        prearm_hover_z=prearm_hover_z,
        prearm_vertical_layout=prearm_vertical_layout.copy(),
        prearm_takeoff_z=prearm_takeoff_z,
        trail_rgba=trail_rgba,
        pos_buffer=deque(maxlen=_TRAIL_BUFFER_MAXLEN),
        center_trace_prev=center_trace_prev,
        plot_enabled=plot_enabled,
        fig=fig,
        ax_hand=ax_hand,
        ax_topo=ax_topo,
        left_pose_state=left_pose_state,
        key_queue=key_queue,
        gesture_control_enabled_box=[gesture_control_enabled],
        prearm_climb_enabled_box=[False],
        prearm_phase_box=["ground"],
        prearm_vertical_leg_box=["climb"],
        prearm_has_flown_box=[False],
        left_pose_reset_req_box=[False],
        left_pose_runtime_armed_box=[False],
        orbbec_swap_mp_hands=orbbec_swap_mp_hands,
        left_use_camera_at_arm=left_use_camera_at_arm,
        left_M_rot=left_M_rot,
        left_M_trans=left_M_trans,
        left_cam_preset=left_cam_preset,
        left_palm_basis=left_palm_basis,
        left_rot_pivot_key=left_rot_pivot_key,
        left_dual_webcam_rot_eff=left_dual_webcam_rot_eff,
        left_rot_webcam_vis_thresh=left_rot_webcam_vis_thresh,
        left_unwind_s=float(cfg.left_unwind_s),
        left_pose_tuning=left_pose_tuning,
        start_time=time.monotonic(),
        extras={
            "report_debug_figs": report_debug_figs,
            "report_panels": panels if panels.any_enabled() else None,
        },
    )


def make_key_poller(boot: OnlineBoot, *, global_hotkeys: bool):
    ctx = OnlineKeyContext.from_boot(boot)

    def _poll_keys(cv_key: int | None = None) -> bool:
        return process_online_control_keys(
            boot.key_queue if bool(global_hotkeys) else None,
            global_hotkeys=bool(global_hotkeys),
            ctx=ctx,
            cv_key=cv_key,
        )

    return _poll_keys


def sync_armed_flags(boot: OnlineBoot) -> None:
    boot.gesture_control_enabled = bool(boot.gesture_control_enabled_box[0])
    boot.prearm_climb_enabled = bool(boot.prearm_climb_enabled_box[0])
    boot.prearm_phase = str(boot.prearm_phase_box[0])
    boot.prearm_vertical_leg = str(boot.prearm_vertical_leg_box[0])
    boot.prearm_has_flown = bool(boot.prearm_has_flown_box[0])
    boot.left_pose_reset_req = bool(boot.left_pose_reset_req_box[0])
    boot.left_pose_runtime_armed = bool(boot.left_pose_runtime_armed_box[0])
