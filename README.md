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

# Terminal 2
pixi run -e deploy real-dual -- --drones-config config/2drones.toml
```

Edit `config/2drones.toml` (or copy from `config/drones.example.toml`) for your drone URIs and room frame.

- **Sim:** 24 virtual morph points (`online-dual` default)
- **Real:** 8 virtual morph points; physical count = `[[drone]]` entries in `drones.toml` (e.g. 2 drones follow indices 0 and 1)
- **Ground layout:** sim uses a chessboard at `z=0.05 m`; real uses each drone’s TOML `home` (XY fixed on vertical legs)

### Prearm sequence (press `1` four times per cycle)

Axswarm safety filter **engages on the first `1`** (not at idle on ground). Default MPC replan rate is **8 Hz** (`--axswarm-mpc-hz`, matches `config/axswarm_settings.yaml`).

| Press `1` | Phase | Target |
|-----------|--------|--------|
| 1 | **Vertical climb** | Ground XY, rise to `--prearm-takeoff-z` (default ≈ morph z₀, often ~1.4 m) |
| 2 | **Hover formation** | Spread to `--prearm-hover-z` (default 1.50 m) |
| 3 | **Vertical descend** | Shrink back to takeoff height (ground XY) |
| 4 | **Ground** | Return to startup layout |

Then cycle repeats from ground. All prearm moves and exit landing use the same axswarm-filtered `swarm.setpoint` stream as gesture control.

**Real swarm:** same key sequence and the same axswarm-filtered setpoint stream (no separate blocking `goto`).

### Controls

| Key | Action |
|-----|--------|
| `1` | Advance prearm: climb → formation → descend → ground (blocked while gesture armed) |
| `Space` | **Arm / disarm** gesture control (hand-driven formation; use after hover formation) |
| `0` | Arm / disarm left-hand whole-formation pose |
| `q` or `Enter` | Quit (real: lands at TOML `home` when `land_on_exit = true`) |

**Typical session:** `1` ×4 to reach hover formation → `Space` for gestures → `Space` to disarm → `1` ×2 to descend to vertical then ground → `q`.

**Useful flags:** `--prearm-takeoff-z`, `--prearm-hover-z`, `--axswarm-mpc-hz 8`, `--profile-frame`, `--sim-render-every 2`.

Legacy scripts and unit tests are on the **`backup-archive`** git branch.

---

## Code layout

```text
src/
  online_control_dual.py      # simulation entry (24 drones, axswarm @ 8 Hz)
  online_control_real_dual.py # real Crazyflie entry (8 morph points)
  online_control.py           # shared main loop (prearm phases, gestures, filter)
  functions/
    display_sim/              # Orbbec hand pipeline, Crazyflow step/render
    real_swarm/               # executor (swarmGPT DroneSwarm + sim→room setpoints)
    swarm_motion/             # axswarm filter, prearm layouts, spacing
config/
  drones.example.toml         # template
  2drones.toml / 5drones.toml # lab presets (swarmGPT-style URIs)
  axswarm_settings.yaml       # axswarm MPC / collision defaults (freq: 8 Hz)
scripts/
  setup_orbbec.sh             # download Orbbec K4A Wrapper
  setup_mocap.sh              # clone/build motion_capture_tracking (deploy)
  ensure_sqlite.sh            # deploy: libsqlite for swarm_gpt import (pixi activation)
```

---

## Acknowledgements

[Crazyflow](https://github.com/utiasDSL/crazyflow) · [swarmGPT](https://github.com/utiasDSL/swarmGPT) · [MediaPipe](https://developers.google.com/mediapipe) · [Orbbec K4A Wrapper](https://github.com/orbbec/OrbbecSDK-K4A-Wrapper) · [axswarm / AMSwarm](https://github.com/learnsyslab/axswarm/tree/amswarm)

## License

MIT
