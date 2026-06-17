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

- Linux x86_64 (real Crazyflies) or Linux / macOS arm64 (simulation)
- [pixi](https://pixi.sh)
- Orbbec depth camera (Femto Bolt / similar)

GPU is optional (JAX runs on CPU by default).

---

## Installation

```bash
git clone --recurse-submodules <repo-url> iso_swarm
cd iso_swarm
pixi install
pixi run setup    # once: download Orbbec K4A Wrapper + build pyk4a
```

Submodules: **crazyflow**, **drone-models**, **drone-controllers**, **axswarm** (`amswarm`).  
On Linux, `pixi install` also pulls **cflib2** (real Crazyflie control). macOS is simulation-only.

`pixi shell` sets `PYTHONPATH=src` and wires `ORBBEC_ROOT` when `third_party/orbbec/current` exists.

---

## Run

From the repo root (so `hand_landmarker.task` resolves):

```bash
pixi shell

# Simulation (MuJoCo + Orbbec)
python src/online_control_dual.py
python src/online_control_dual.py --no-left-dual-webcam-rot   # Orbbec only

# Real Crazyflies (Linux, Lighthouse — no ROS)
cp config/drones.example.toml config/drones.toml   # edit URIs + home
pixi run real-dual -- --drones-config config/drones.toml
```

Morph uses **8 virtual formation points** by default; physical drone count is only `[[drone]]` entries in `drones.toml` (e.g. 2 drones follow points 0 and 1).

**Startup layout:** simulation spawns on a ground chessboard; real drones use `home` in `drones.toml`. Axswarm safety filtering is active from launch.

| Key | Action |
|-----|--------|
| `1` | Toggle **ground ↔ hover** (TOML / chessboard ground → pre–Space hover layout; axswarm-filtered). Press again to descend. Blocked after `Space` arms gestures. |
| `Space` | Arm gesture control (hand-driven formation) |
| `0` | Arm / disarm left-hand whole-formation pose |
| `q` or `Enter` | Quit |

Legacy scripts and unit tests are on the **`backup-archive`** git branch.

---

## Code layout

```text
src/
  online_control_dual.py      # simulation entry
  online_control_real_dual.py # real Crazyflie entry
  online_control.py           # shared main loop
  functions/
    display_sim/              # Orbbec hand pipeline, plots
    real_swarm/               # Crazyflie bridge (cflib2)
submodules/
  crazyflow/  drone-models/  drone-controllers/  axswarm/
config/drones.toml            # real-swarm URIs, home, frame mapping
```

---

## Acknowledgements

[Crazyflow](https://github.com/utiasDSL/crazyflow) · [MediaPipe](https://developers.google.com/mediapipe) · [Orbbec K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper) · [axswarm / AMSwarm](https://github.com/learnsyslab/axswarm/tree/amswarm)

## License

MIT
