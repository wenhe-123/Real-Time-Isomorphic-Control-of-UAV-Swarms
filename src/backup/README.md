# backup — unused / legacy code (not used by `online_control_dual.py`)

Active entry point: `src/online_control_dual.py` → `online_control.py`.

## Layout

| Folder | Contents |
|--------|----------|
| `entrypoints/` | Standalone scripts: old Orbbec online control, offline replay, spacing check, LED demo |
| `pipelines/` | Gesture-only demos (webcam / Orbbec / dual-camera fusion) |
| `runtime/` | Webcam-only modes demo, dual Orbbec+webcam fusion tracker (not live dual rotation) |
| `shared/` | Helpers only used by backup runtime/pipelines |
| `sampling/3region_mapping/` | Alternate morph sampling backend (`ISO_SWARM_SAMPLING_BACKEND=3region_mapping`) |
| `legacy/` | Older copies, mode state experiments, compatibility shims |
| `data/logs/` | Sample morph logs (`text1.txt` … `text4.txt`) |
| `data/orbbec/` | Orbbec SDK log dumps |
| `data/plots/` | Offline replay PNG output |
| `tests/` | Pytest suite (left-hand / palm basis unit tests) |
| `tools/` | One-off Orbbec / Crazyflow test scripts |

## Run backup scripts

From `iso_swarm` with `pixi shell` (PYTHONPATH=src):

```bash
python src/backup/entrypoints/offline_control.py --log src/backup/data/logs/text3.txt --plot-only
python src/backup/pipelines/orbbec_main.py
python src/backup/runtime/hand_tracking_webcam_modes.py
python src/backup/runtime/hand_tracking_orbbec_demo.py
python src/backup/pipelines/webcam_main.py
python src/backup/entrypoints/check_formation_spacing.py --n 24 --open 0
pytest -v src/backup/tests
```

## Active `src/` tree (dual path)

- `online_control_dual.py`, `online_control.py`, `online_control_{loop,cli,defaults,state,targets,profiler}.py`
- `backup/runtime/hand_draw_utils.py`, `backup/runtime/orbbec_live_steps.py` (demo-only; not in `shared/`)
- `backup/runtime/demo_defaults.py` (re-exports webcam/orbbec demo constants for backup scripts)
- `shared/display_sim/orbbec_hand.py` (Orbbec library for online control)
- `shared/mode_switch/` (M1–M5, topology, dual-mode fusion)
- `shared/open_close/` (morph surface sampling)
- `shared/swarm_motion/` (formation translate/rotate, spacing, axswarm)
- `shared/display_sim/` (3D plot, LED, depth fusion, pose overlay)
- `shared/dual_cam/` (Orbbec + USB webcam)
- `shared/online_pipeline_defaults.py` (production pipeline defaults)
- `debug/pipeline_tuning.py` (``--debug-webcam-pipeline`` / ``--debug-3d-plot`` overrides)
- `pure_angular/` (default sampling)
- `shared/` (except moved modules)
