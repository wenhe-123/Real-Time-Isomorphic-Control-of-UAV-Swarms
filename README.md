# Iso-Swarm: Real-Time Isomorphic Control of UAV Swarms via Hand Geometry Mapping

![Crazyflow Logo](https://github.com/utiasDSL/crazyflow/raw/main/docs/img/logo.png)

---

## 🚀 Overview

This project presents a **real-time hand-gesture-based control system for UAV swarms** using RGB-D sensing and geometric topology analysis.

Built on top of the Crazyflow simulation framework, we extend it with:

* ✋ Hand tracking using MediaPipe
* 📡 RGB-D input from Orbbec camera
* 🧠 3D topology analysis 
* 🔄 Continuous gesture representation (plane ↔ sphere)
* 🚁 Swarm control mapping

---

## 🧠 Core Idea

Instead of discrete gesture classification, we introduce a **continuous geometric mapping**:

```text
Hand Geometry → PCA → Topology → Morph α → Swarm Control
```

* ✊ Fist → Sphere-like topology
* ✋ Open hand → Plane-like topology
* 🔄 Intermediate gestures → Continuous interpolation

This enables **smooth, real-time, and intuitive swarm control**.

---

## 🎯 Key Features

* ✅ Real-time hand tracking (MediaPipe)
* ✅ 3D hand topology reconstruction
* ✅ Continuous morph parameter `α`
* ✅ Plane-to-sphere isomorphic mapping
* ✅ Compatible with UAV swarm simulation (Crazyflow)

---

## 📦 Installation

### 1. Clone repository

```bash
git clone --recurse-submodules https://github.com/wenhe-123/Real-Time-Isomorphic-Control-of-UAV-Swarms.git
cd iso-swarm
```

---

### 2. Setup environment

```bash
pixi install
pixi shell
```

---

### 3. Install additional dependencies

```bash
pip install pyk4a mediapipe opencv-python matplotlib
python -m pip install mediapipe
```


---

## 📡 Orbbec Camera Setup

Download Orbbec SDK (K4A Wrapper):

👉 https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases/

Release v1.10.4 was used in this project 

After downloading:

```bash
cd OrbbecSDK_K4A_Wrapper_v1.10.4_linux_202504141102/bin
./k4aviewer
```

---

## ▶️ Run

```bash
python hand_swarm/hand_swarm_control_orbbec.py
```

## 🧪 Method Overview

### 1. Hand Tracking

* MediaPipe detects 21 hand keypoints
* RGB-D used for visualization and depth reference

### 2. Topology Analysis

* PCA on 3D hand keypoints
* Extract:

  * Planarity
  * Isotropy
  * Finger spread

### 3. Morph Mapping

```text
α ∈ [0, 1]
```

* α → 0 → sphere (fist)
* α → 1 → plane (open hand)

### 4. Swarm Control

* α controls swarm behavior (e.g., expansion, velocity)
* Normal vector defines direction
* Centroid defines spatial reference

---

## 🚁 Applications

* Swarm drone control
* Human-swarm interaction
* Continuous gesture interfaces
* XR / HCI research

---

## 📸 Demo

> (Add demo GIF here)

---

## ⚠️ Notes

* Orbbec SDK is **not included** in this repository
* Please install it manually
* Make sure `LD_LIBRARY_PATH` includes SDK `lib/`

---

## 🙏 Acknowledgements

* [Crazyflow](https://github.com/utiasDSL/crazyflow)
* MediaPipe
* Orbbec

---

## 📜 License

MIT
