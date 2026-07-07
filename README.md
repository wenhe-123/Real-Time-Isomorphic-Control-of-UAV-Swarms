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
→ axswarm planning
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

Clone the repository, then install dependencies with [pixi](https://pixi.sh):

```bash
git clone https://github.com/wenhe-123/Real-Time-Isomorphic-Control-of-UAV-Swarms.git iso_swarm
cd iso_swarm
```

**Simulation (Linux, Orbbec camera):**

```bash
pixi install && pixi run setup
```

**Real Crazyflies (+ ROS 2 mocap):**

```bash
pixi install -e deploy && pixi run -e deploy setup
```

After pulling changes that touch `swarmgpt`, refresh the locked git revision:

```bash
pixi update -e deploy swarmgpt && pixi install -e deploy
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
# Terminal 1 — use deploy env only (do not source system ROS /opt/ros/jazzy)
pixi run -e deploy mocap

# Terminal 2 — live (mocap + drones required):
pixi run -e deploy real-dual -- --drones-config config/drones.toml
```

Edit `config/drones.toml` and `config/settings.yaml` for real-swarm hardware. Sim/real defaults: `config/online_defaults.yaml`; MPC limits: `config/axswarm_settings.yaml`.

### Controls

| Key | Action |
|-----|--------|
| `1` | Takeoff / descent (step through prearm phases) |
| `Space` | Arm / disarm gesture control |
| `0` | Arm / disarm left-hand whole-formation pose (L-move) |
| `q` or `Enter` | Quit (real: auto-land when configured) |

**Typical session:** `1` until hover formation → `Space` (gestures) → `Space` (disarm) → `1` to land → `q`.

## Code layout

```text
src/
  online_control_dual.py      # simulation entry (24 drones)
  online_control_real_dual.py # real Crazyflie entry (8 morph points)
  online_control.py           # shared main loop (prearm phases, gestures, filter)
  functions/
    display_sim/              # production pipeline (no debug drawing)
    dual_cam/                 # Orbbec capture, MP hand index helpers
    mode_switch/              # M1–M5 mode, open/close, morph topology
    swarm_motion/             # axswarm filter, prearm layouts, left-hand L-move pose
    runtime/                  # boot, online_defaults.yaml loader, runtime config
    real_swarm/               # executor (swarmGPT DroneSwarm + sim→room setpoints)
  debug/                      # optional overlays & diagnostics (off by default)
config/                       # online_defaults, drones, axswarm MPC
scripts/                      # Orbbec SDK, mocap, deploy helpers
```

---

## Acknowledgements

[Crazyflow](https://github.com/utiasDSL/crazyflow) · [swarmGPT](https://github.com/utiasDSL/swarmGPT) · [MediaPipe](https://developers.google.com/mediapipe) · [Orbbec K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper) · [axswarm / AMSwarm](https://github.com/learnsyslab/axswarm/tree/amswarm)

## License

MIT
