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

Simulation and visualization use [Crazyflow](https://github.com/utiasDSL/crazyflow). Collision-aware filtering uses [axswarm](https://github.com/learnsyslab/axswarm/tree/amswarm) (AMSwarm, JAX).

---

## Requirements

- **Linux x86_64** — Orbbec camera, real Crazyflies, ROS 2 mocap
- **Linux / macOS arm64** — simulation only (no real-swarm / deploy env)
- [pixi](https://pixi.sh) (recommended) or Python 3.12 + pip
- Orbbec depth camera (Femto Bolt / similar) on Linux

GPU is optional (JAX runs on CPU by default).

---

## Installation

**Simulation (Linux, Orbbec camera):**

```bash
pixi install && pixi run setup
```

**Real Crazyflies (+ ROS 2 mocap):**

```bash
pixi install -e deploy && pixi run -e deploy setup
```

`setup` downloads the Orbbec SDK and builds `pyk4a` for the active pixi environment.

---

## Run

From the repo root (so `hand_landmarker.task` resolves):

### Simulation (MuJoCo + Orbbec)

```bash
pixi run online-dual
```

### Real Crazyflies (Linux, motion capture via ROS 2)

```bash
# Terminal 1
pixi run -e deploy mocap

# Terminal 2
pixi run -e deploy real-dual -- --drones-config config/2drones.toml
```

Edit `config/2drones.toml` (or copy from `config/drones.example.toml`) for your drone URIs and room frame.

Morph uses **8 virtual formation points** by default (`online_control_real_dual.py`); physical drone count is the number of `[[drone]]` entries in `drones.toml` (e.g. 2 drones follow indices 0 and 1).

**Startup layout:** simulation spawns on a ground chessboard; real drones use `home` in `drones.toml`. Axswarm safety filtering is active from launch.

### Controls

| Key | Action |
|-----|--------|
| `1` | Toggle **ground ↔ hover** (axswarm-filtered). Blocked while gesture control is armed (`Space`). |
| `Space` | **Arm / disarm** gesture control (hand-driven formation). |
| `0` | Arm / disarm left-hand whole-formation pose |
| `q` or `Enter` | Quit (real swarm: lands at TOML `home` when `land_on_exit = true`) |

**Typical shutdown:** `Space` → `1` (descend if at hover) → `q`.

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
    real_swarm/               # Crazyflie bridge (cflib2 + mocap)
config/
  drones.example.toml         # template
  2drones.toml / 5drones.toml # lab presets (swarmGPT-style URIs)
  axswarm_settings.yaml       # axswarm MPC / collision defaults
scripts/
  setup_orbbec.sh             # download Orbbec K4A Wrapper
  setup_mocap.sh              # clone/build motion_capture_tracking (deploy)
```

---

## Acknowledgements

[Crazyflow](https://github.com/utiasDSL/crazyflow) · [swarmGPT](https://github.com/utiasDSL/swarmGPT) · [MediaPipe](https://developers.google.com/mediapipe) · [Orbbec K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper) · [axswarm / AMSwarm](https://github.com/learnsyslab/axswarm/tree/amswarm)

## License

MIT
