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
python hand_swarm/hand_swarm_control_orbbec.py
```

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
