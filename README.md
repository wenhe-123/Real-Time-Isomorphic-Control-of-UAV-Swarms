# Iso-Swarm: Real-Time Isomorphic Control of UAV Swarms via Hand Geometry Mapping

Real-time hand-gesture control for UAV swarms using Orbbec RGB-D sensing, MediaPipe hand tracking, geometric topology analysis, and Crazyflow simulation.

```text
Hand Geometry → PCA → Topology → Morph α → Swarm Control → (optional) axswarm safety filter
```

---

## Requirements

| Item | Required for main demo | Notes |
|------|------------------------|-------|
| **Linux x86_64** (or macOS arm64) | Yes | `pixi` platforms in `pyproject.toml` |
| [pixi](https://pixi.sh) | Yes | Manages Python env + editable deps |
| **Orbbec depth camera** + [K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases/) | Yes | Main entry uses `pyk4a` |
| **axswarm-amswarm** (sibling folder) | Yes for `pixi install` | Default planner uses axswarm MPC |
| GPU | No | JAX runs on CPU by default |

Not included in this repository: Orbbec SDK binaries, axswarm source (separate checkout), thesis/report assets.

---

## Workspace layout

`pixi` expects **axswarm next to iso_swarm** (see `pyproject.toml`):

```text
workspace/
├── iso_swarm/                         ← this repo
│   ├── hand_landmarker.task           ← MediaPipe model (tracked in git)
│   ├── crazyflow/                     ← simulator (vendored)
│   ├── submodules/                    ← drone-models, drone-controllers
│   ├── src/
│   │   ├── online_control_dual.py     ← primary entry
│   │   ├── online_control.py
│   │   └── functions/                 ← active library code
│   └── data/                          ← local recordings only (gitignored)
└── axswarm-amswarm/
    └── axswarm-amswarm/               ← axswarm Python package + params/settings.yaml
        ├── axswarm/
        └── params/settings.yaml
```

Legacy scripts live on the **`backup-archive`** git branch (`src/backup/`), not on `main`.

---

## Installation (full steps)

### 1. Clone iso_swarm with submodules

```bash
mkdir -p ~/workspace && cd ~/workspace

git clone --recurse-submodules \
  git@github.com:wenhe-123/Iso-Swarm-Real-Time-Isomorphic-Control-of-UAV-Swarms-via-Hand--Geometry-Mapping.git \
  iso_swarm

cd iso_swarm

# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

HTTPS:

```bash
git clone --recurse-submodules \
  https://github.com/wenhe-123/Iso-Swarm-Real-Time-Isomorphic-Control-of-UAV-Swarms-via-Hand--Geometry-Mapping.git \
  iso_swarm
```

### 2. Install axswarm (required for `pixi install`)

Place the axswarm package at the path `pixi` references:

```bash
cd ~/workspace
# Example: clone or copy your axswarm repo so this path exists:
#   axswarm-amswarm/axswarm-amswarm/axswarm/solve.py
mkdir -p axswarm-amswarm
# git clone <your-axswarm-repo-url> axswarm-amswarm/axswarm-amswarm
```

Verify:

```bash
test -f axswarm-amswarm/axswarm-amswarm/axswarm/solve.py && echo OK
```

Alternative: edit `pyproject.toml` → `[tool.pixi.pypi-dependencies]` → `axswarm = { path = "..." }` to match your layout, then run `pixi install`.

### 3. Create the Python environment

```bash
cd ~/workspace/iso_swarm
pixi install
pixi shell
```

`pixi shell` sets `PYTHONPATH=src` automatically.

### 4. Orbbec SDK + pyk4a

1. Download and unpack [OrbbecSDK-K4A-Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases/).

2. Add to `~/.bashrc` (adjust `ORBBEC_ROOT`):

```bash
export ORBBEC_ROOT=~/OrbbecSDK_K4A_Wrapper_v1.10.4_linux_202504141102
export PATH="$PATH:$ORBBEC_ROOT/bin"
export LD_LIBRARY_PATH="$ORBBEC_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

3. Test the camera:

```bash
cd "$ORBBEC_ROOT/bin"
./k4aviewer
```

4. Inside `pixi shell`:

```bash
pip install pyk4a opencv-python matplotlib
```

(`mediapipe`, `jax`, `mujoco`, etc. are already pulled in by pixi.)

---

## Run

Always start from the **iso_swarm repo root** (so `hand_landmarker.task` resolves).

```bash
cd ~/workspace/iso_swarm
pixi shell

# Default: Orbbec input, axswarm safety filter, 24 drones
python src/online_control_dual.py

# Profile per-frame timing
python src/online_control_dual.py --mp-detect-every 1 --profile-frame

# Skip axswarm MPC (raw gesture targets; still needs Orbbec)
python src/online_control_dual.py --planner direct

# Change swarm size (minimum 8)
python src/online_control_dual.py --point-count 48
```

### Controls (in the Orbbec window)

| Key | Action |
|-----|--------|
| `Space` | Enable / disable gesture control |
| `0` | Arm / disarm left-hand whole-formation pose |
| `1`–`3` | Morph mode (M1 / M2 / M3) |
| `q` or `Enter` | Quit |

### Report / debug screenshots (one panel at a time)

```bash
python src/online_control_dual.py --debug-report-morph --plot-every 8
python src/online_control_dual.py --debug-report-hand --plot-every 8
python src/online_control_dual.py --debug-report-pca --plot-every 8
python src/online_control_dual.py --debug-report-landmarks --plot-every 8
python src/online_control_dual.py --debug-report-palm --plot-every 8 --left-pose-frame-viz
```

Avoid `--debug-report-viz` (opens all five Matplotlib windows; heavy).

### Tests

```bash
pixi run -e tests tests
# runs: pytest -v src/backup/tests  (on backup-archive branch)
```

On `main`, unit tests may be absent; check out `backup-archive` for the legacy test suite.

---

## Code layout (`main` branch)

```text
src/
  online_control_dual.py    # primary CLI entry
  online_control.py         # Orbbec + Crazyflow main loop
  functions/
    display_sim/            # Orbbec hand pipeline, 3D plots, report debug figures
    dual_cam/               # frame capture, webcam fusion helpers
    mode_switch/            # morph modes, topology, shape control
    runtime/                # boot, CLI args, pipeline tuning
    swarm_motion/           # left-hand pose, axswarm filter, spacing
  debug/                    # offline replay / debug tools
crazyflow/                  # MuJoCo swarm simulator (editable install)
submodules/                 # drone-models, drone-controllers
hand_landmarker.task        # MediaPipe hand model (project root)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pixi install` fails on axswarm path | Clone axswarm to `../axswarm-amswarm/axswarm-amswarm` or fix path in `pyproject.toml` |
| `hand_landmarker.task` not found | Run from `iso_swarm/` root, or pass `--model /path/to/hand_landmarker.task` |
| Orbbec / pyk4a import errors | Set `PATH` and `LD_LIBRARY_PATH` to SDK `bin/` and `lib/` |
| Depth looks wrong | Try `--no-orbbec-flip-horizontal` or `--no-orbbec-use-transformed-depth` |
| No camera (laptop only) | Check out `backup-archive` → `src/backup/runtime/hand_tracking_webcam_modes.py` |
| `MUJOCO_LOG.TXT` / `Log/OrbbecSDK.log.txt` appear | Auto-generated runtime logs; safe to delete (gitignored) |

---

## Local-only files (do not commit)

These are created at runtime and are listed in `.gitignore`:

- `MUJOCO_LOG.TXT` — MuJoCo warning log
- `Log/` — Orbbec SDK log directory
- `logs/` — rigid-pose traces, etc.
- `data/*.npz` — optional local trajectory dumps

---

## Method (short)

1. MediaPipe extracts 21 hand landmarks (Orbbec RGB + depth).
2. PCA / topology features → continuous morph parameter α ∈ [0, 1].
3. α maps hand geometry to swarm target formation (Crazyflow visualization).
4. Optional **axswarm** MPC filters targets for collision-aware motion (`--planner axswarm`, default in `online_control_dual.py`).

---

## Acknowledgements

Crazyflow · MediaPipe · Orbbec · axswarm (AMSwarm)

## License

MIT
