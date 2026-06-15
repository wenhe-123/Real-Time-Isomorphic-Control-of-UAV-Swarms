# Iso-Swarm

**Iso-Swarm** is a real-time hand-gesture-based control framework for UAV swarms.  
The project explores how human hand geometry can be mapped to swarm formation, deformation, translation, and rotation in an intuitive way.

Instead of using gestures only as discrete commands, Iso-Swarm treats the UAV swarm as a virtual structure that continuously follows the user's hand motion and hand shape.

---

## Overview

The system uses camera-based hand tracking to extract hand landmarks and convert them into swarm control parameters.

```text
camera input
→ hand landmark detection
→ hand geometry processing
→ superellipsoid-based formation generation
→ global translation and rotation
→ safety filtering
→ UAV target positions
```

**Two-hand interaction**

| Hand | Role |
|------|------|
| **Left** | Formation mode (M1–M5), global swarm translation and rotation |
| **Right** | Continuous open–close deformation |

Simulation and visualization use [Crazyflow](https://github.com/utiasDSL/crazyflow). Optional collision-aware filtering uses [axswarm](https://github.com/learnsyslab/axswarm/tree/amswarm) (AMSwarm, JAX).

---

## Requirements

- Linux x86_64 or macOS arm64
- [pixi](https://pixi.sh)
- Orbbec depth camera + SDK (see below)
- axswarm checkout as a sibling folder (see below)

GPU is optional (JAX runs on CPU by default).

---

## Installation

### 1. Clone iso_swarm

```bash
mkdir -p ~/workspace && cd ~/workspace

git clone --recurse-submodules \
  https://github.com/wenhe-123/Iso-Swarm-Real-Time-Isomorphic-Control-of-UAV-Swarms-via-Hand--Geometry-Mapping.git \
  iso_swarm

cd iso_swarm
git submodule update --init --recursive   # if submodules were skipped
```

### 2. Clone axswarm

`pixi` expects this layout (see `pyproject.toml`):

```text
workspace/
├── iso_swarm/
└── axswarm-amswarm/
    └── axswarm-amswarm/          # axswarm repo root (axswarm/, params/, …)
```

```bash
cd ~/workspace
mkdir -p axswarm-amswarm
git clone -b amswarm https://github.com/learnsyslab/axswarm.git \
  axswarm-amswarm/axswarm-amswarm

test -f axswarm-amswarm/axswarm-amswarm/axswarm/solve.py && echo OK
```

### 3. Python environment

```bash
cd ~/workspace/iso_swarm
pixi install
pixi shell
```

`pixi shell` sets `PYTHONPATH=src`.

### 4. Orbbec SDK + pyk4a

1. SDK: [orbbec/OrbbecSDK](https://github.com/orbbec/OrbbecSDK)  
2. For Python / K4A API used by this project, also install the [OrbbecSDK-K4A-Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases/) build matching your camera.

Add to `~/.bashrc` (adjust `ORBBEC_ROOT`):

```bash
export ORBBEC_ROOT=~/OrbbecSDK_K4A_Wrapper_v1.10.4_linux_202504141102
export PATH="$PATH:$ORBBEC_ROOT/bin"
export LD_LIBRARY_PATH="$ORBBEC_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

Test the device:

```bash
cd "$ORBBEC_ROOT/bin" && ./k4aviewer
```

Inside `pixi shell`:

```bash
pip install pyk4a opencv-python matplotlib
```

---

## Run

Run from the **iso_swarm repo root** (so `hand_landmarker.task` resolves).

```bash
cd ~/workspace/iso_swarm
pixi shell

python src/online_control_dual.py
python src/online_control_dual.py --mp-detect-every 1 --profile-frame
python src/online_control_dual.py --planner direct          # no axswarm MPC
python src/online_control_dual.py --no-left-dual-webcam-rot # Orbbec only, no USB webcam
```

| Key | Action |
|-----|--------|
| `Space` | Enable / disable gesture control |
| `0` | Arm / disarm left-hand whole-formation pose |
| `q` or `Enter` | Quit |

Legacy scripts and unit tests are on the **`backup-archive`** git branch.

---

## Code layout

```text
src/
  online_control_dual.py    # primary entry
  online_control.py         # main loop
  functions/
    display_sim/            # Orbbec hand pipeline, plots
    dual_cam/               # frame capture, optional USB webcam rotation
    mode_switch/            # morph modes, topology, shape control
    runtime/                # boot, CLI, pipeline tuning
    swarm_motion/           # left-hand pose, axswarm filter
crazyflow/                  # MuJoCo simulator
submodules/                 # drone-models, drone-controllers
hand_landmarker.task        # MediaPipe model (project root)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pixi install` fails on axswarm | Clone [learnsyslab/axswarm](https://github.com/learnsyslab/axswarm/tree/amswarm) to `../axswarm-amswarm/axswarm-amswarm` |
| Orbbec / pyk4a errors | Set `PATH` and `LD_LIBRARY_PATH`; see [OrbbecSDK](https://github.com/orbbec/OrbbecSDK) |
| USB webcam opens unexpectedly | Default dual-webcam rotation is on; use `--no-left-dual-webcam-rot` |
| Depth looks wrong | Try `--no-orbbec-flip-horizontal` or `--no-orbbec-use-transformed-depth` |

Runtime logs (`Log/`, `MUJOCO_LOG.TXT`, `data/*.npz`) are gitignored and safe to delete locally.

---

## Acknowledgements

Crazyflow · MediaPipe · Orbbec · [axswarm / AMSwarm](https://github.com/learnsyslab/axswarm/tree/amswarm)

## License

MIT
