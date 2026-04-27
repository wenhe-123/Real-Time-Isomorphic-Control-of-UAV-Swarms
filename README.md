# Iso-Swarm: Real-Time Isomorphic Control of UAV Swarms via Hand Geometry Mapping

## 🚀 Overview

Iso-Swarm is a real-time hand-gesture-based control system for UAV swarms using RGB-D sensing and geometric topology analysis.

Instead of discrete gestures, this project introduces a **continuous topology-based control paradigm**.

---

## 🧠 Core Idea

```text
Hand Geometry → PCA → Topology → Morph α → Swarm Control
```

* ✊ Fist → Sphere-like topology
* ✋ Open hand → Plane-like topology
* 🔄 Continuous interpolation via α

This enables smooth and intuitive swarm control.

---

## 🎯 Features

* Real-time hand tracking (MediaPipe)
* 3D topology analysis (PCA)
* Continuous control parameter `α`
* Plane ↔ Sphere isomorphic mapping
* Integration with Crazyflow simulator

---

## 📦 Installation

```bash
git clone --recurse-submodules https://github.com/wenhe-123/Real-Time-Isomorphic-Control-of-UAV-Swarms.git
cd iso-swarm

pixi install
pixi shell

pip install pyk4a mediapipe opencv-python matplotlib
```

---

## 📡 Orbbec Setup

Download Orbbec SDK (K4A Wrapper):

https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases/

Test device:

```bash
cd OrbbecSDK_K4A_Wrapper_xxx/bin
./k4aviewer
```

---

## ▶️ Run

```bash
pixi run python src/apps/webcam_modes_app.py
pixi run python src/apps/orbbec_app.py
pixi run python src/apps/dual_app.py
pixi run python src/apps/orbbec_modes_app.py
pixi run python src/apps/swarm_orbbec_app.py
```

---

## 🗂️ Code Layout

```text
src/
  apps/        # runnable application entry files (main commands)
  pipelines/   # pipeline wiring (device/cli/main flow)
  shared/      # reusable logic shared by pipelines
  tools/       # utility/testing scripts
  legacy/      # archived old scripts kept as backup
  backup/      # compatibility shims kept for rollback
```

Notes:

* `apps/` is the only supported runtime entry layer.
* `entrypoints/` compatibility layer has been removed.
* `hand_landmarker.task` stays at project root and is still used by defaults.

---

## 🧪 Method

* MediaPipe extracts 21 hand keypoints
* PCA analyzes 3D geometry
* Compute topology features:

  * Planarity
  * Finger spread
* Generate continuous parameter `α ∈ [0,1]`
* Map `α` to swarm behavior

---

## 🚁 Applications

* UAV swarm control
* Human-swarm interaction
* Gesture-based interfaces

---

## 📸 Demo

*(Add demo GIF here)*

---

## ⚠️ Notes

* Orbbec SDK is not included
* Set `LD_LIBRARY_PATH` to SDK `lib/` if needed

---

## 🙏 Acknowledgements

* Crazyflow
* MediaPipe
* Orbbec

---

## 📜 License

MIT
