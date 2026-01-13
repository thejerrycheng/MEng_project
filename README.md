# 🎥 **IRIS: Learning-Driven Task-Specific Cinema Robot Arm for Visuomotor Motion Control**

<p align="center">
  <img src="images/hero.png" width="70%">
</p>
<p align="center">
  <a href="#1️⃣-hardware-platform">1️⃣ Hardware</a> •
  <a href="#2️⃣-low-level-actuator-control">2️⃣ Actuator Control</a> •
  <a href="#3️⃣-mujoco-simulation">3️⃣ MuJoCo Simulation</a> •
  <a href="#4️⃣-ros-interface-and-rosbag-collection">4️⃣ ROS + Rosbags</a> •
  <a href="#5️⃣-data-processing">5️⃣ Data Processing</a> •
  <a href="#6️⃣-imitation-learning">6️⃣ Imitation Learning</a> •
  <a href="#7️⃣-sim-to-real-deployment">7️⃣ Sim-to-Real</a>
</p>

> **IRIS (Intelligent Robotic Imaging System)** is a low-cost, 3D-printed 6-DOF cinema robot arm that learns smooth, repeatable, and obstacle-aware camera motions through visuomotor imitation learning.
> This repository contains the complete **hardware, simulation, control, ROS data collection, and learning stack**, enabling seamless **sim-to-real and real-to-sim** workflows.

<p align="center">
  <img src="images/overview.png" width="100%">
</p>

---

## 🗂 Repository Structure

```
MEng_project/
├── mujoco_sim/              # MuJoCo simulation, kinematics, planners
├── classical_planner/       # RRT*, potential-field, trajectory generation
├── mpr_control/             # Unitree actuator SDK + low-level control
├── motor_control/           # Motor diagnostics and testing
├── meng_ws/                 # ROS workspace (hardware interface, teleop, logging)
├── bag_reader/              # Rosbag recording and dataset extraction tools
├── sim2real/                # Sim–real synchronization utilities
├── il_training/             # Visuomotor imitation learning training code
├── inverse_kinematics_sim/  # Analytical and numerical IK solvers
├── paper/                   # LaTeX source for accompanying paper
└── README.md
```

---

## 1️⃣ Hardware Platform

IRIS is a **fully 3D-printed 6-DOF robotic camera arm** driven by **Unitree GO-M8010-6 torque-controlled BLDC actuators**.
The design emphasizes low cost, modularity, and high backdrivability for kinesthetic teaching.

### 🔩 3D Models and CAD

- 📐 **Mechanical CAD (STEP + STL):**
  👉 [https://github.com/thejerrycheng/IRIS-Hardware](https://github.com/thejerrycheng/IRIS-Hardware)

- 🖨️ **Printable STL files:**
  👉 [https://github.com/thejerrycheng/IRIS-Hardware/tree/main/STL](https://github.com/thejerrycheng/IRIS-Hardware/tree/main/STL)

<p align="center">
  <img src="images/render.png" width="70%">
</p>

<p align="center">
  <img src="images/mechanical.png" width="40%">
</p>

---

## 2️⃣ Low-Level Actuator Control

Low-level torque, velocity, and position control is implemented using the **official Unitree GO-series actuator SDK**.
Motors communicate over **RS-485 at 1 kHz**, enabling synchronized multi-joint closed-loop control with gravity compensation and impedance control.

### 📘 Unitree SDK and Documentation

- **Official Actuator Documentation:**
  [https://support.unitree.com/home/en/Actuator](https://support.unitree.com/home/en/Actuator)

- **Unitree Actuator SDK Repository:**
  [https://github.com/unitreerobotics/unitree_actuator_sdk](https://github.com/unitreerobotics/unitree_actuator_sdk)

Place the SDK at:

```
mpr_control/unitree_actuator_sdk/
```

### ⚙️ Setup

```bash
cd mpr_control/unitree_actuator_sdk/python
pip install -r requirements.txt
```

### ▶️ Example Motor Demos

```bash
python example_goM8010_6_motor.py     # Motor diagnostics
python position_teleop.py             # Joint-space teleoperation
python torque_teleop.py               # Torque control demo
python velocity_teleop.py             # Velocity control demo
```

<p align="center">
  <img src="videos/motor.gif" width="35%">
</p>

---

## 3️⃣ MuJoCo Simulation

A physics-accurate **MuJoCo digital twin** is provided for:

- Kinematic verification
- Classical motion planning (RRT\*, potential fields)
- Trajectory preview
- Real–sim synchronization

### ⚙️ Setup

```bash
cd mujoco_sim
pip install -r requirements.txt
```

### ▶️ Run Simulation Demos

```bash
python circle_path_tracking.py
python line_path_tracking.py
```

<p align="center">
  <img src="videos/apf.gif" width="45%">
  <img src="videos/rrt.gif" width="42%">
</p>

```bash
python cinema_line_tracking.py --mode crane
```

<p align="center">
  <img src="videos/crane-ezgif.com-video-to-gif-converter.gif" width="60%">
</p>

```bash
python cinema_line_tracking.py --mode dolly
```

<p align="center">
  <img src="videos/dolly-ezgif.com-video-to-gif-converter.gif" width="60%">
</p>

```bash
python cinema_line_tracking.py --mode pan
```

<p align="center">
  <img src="videos/pan-ezgif.com-video-to-gif-converter.gif" width="60%">
</p>

### ▶️ Interactive Teleoperation

```bash
python teleop_ik.py     # Cartesian IK teleoperation
python teleop_fk.py     # Joint-space teleoperation
```

<p align="center">
  <img src="videos/teleop.gif" width="60%">
</p>

---

## 4️⃣ ROS Interface, Hardware Bringup, and Rosbag Collection

The ROS stack provides a unified interface for **real‑robot control, calibration, teleoperation, MuJoCo synchronization, and dataset recording**.
It bridges the Unitree actuator hardware, RealSense RGB‑D sensing, and the MuJoCo digital twin into a single synchronized pipeline.

### 🧩 System Overview

**Core ROS Nodes**

| Node                      | File                        | Role                                                                                                                |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `iris_hw_node`            | `iris_hw_node.py`           | Low‑level hardware driver. Streams Unitree actuator states, executes joint commands, and publishes `/joint_states`. |
| `keyboard_joint_teleop`   | `keyboard_joint_teleop.py`  | Keyboard teleoperation interface publishing `/arm/command`.                                                         |
| `teach_repeat_node`       | `teach_and_repeat.py`       | Kinesthetic teaching and playback of demonstrations.                                                                |
| `joint_state_calibrator`  | `calibrate_joint_states.py` | Converts raw motor states into calibrated kinematic joint states.                                                   |
| `home_calibration_node`   | `calibrate_home_state.py`   | Interactive home pose calibration and offset saving.                                                                |
| `mujoco_state_visualizer` | `mujoco_visualizer.py`      | Real‑to‑sim state synchronization in MuJoCo.                                                                        |

---

### 🚀 Hardware Bringup

Launch the IRIS hardware driver:

```bash
roslaunch unitree_arm_ros iris_bringup.launch
```

This starts:

- Unitree actuator streaming over RS‑485 at 200 Hz
- `/joint_states` publisher
- `/arm/command` subscriber
- Joint‑space impedance control with velocity limiting and safety timeout

**Published Topics**

| Topic           | Type                     | Description                                   |
| --------------- | ------------------------ | --------------------------------------------- |
| `/joint_states` | `sensor_msgs/JointState` | Raw motor‑side joint positions and velocities |

**Subscribed Topics**

| Topic          | Type                     | Description                                   |
| -------------- | ------------------------ | --------------------------------------------- |
| `/arm/command` | `sensor_msgs/JointState` | Desired joint targets from teleop or policies |

---

### 🎮 Keyboard Teleoperation

```bash
roslaunch unitree_arm_ros keyboard_teleop.launch
```

This launches:

- Wait‑for‑state node
- `keyboard_joint_teleop.py`

**Teleop Flow**

```
Keyboard Input → /arm/command → iris_hw_node → Actuators → /joint_states
```

---

### 🏠 Home Pose Calibration

Before first use, define the robot home configuration:

```bash
rosrun unitree_arm_ros calibrate_home_state.py
```

Procedure:

1. Manually place robot in upright home pose
2. Press ENTER in terminal
3. Joint offsets are saved to `config/calibration.yaml`

This file stores motor encoder offsets for repeatable kinematic alignment.

---

### 🧭 Kinematic Joint Calibration

Convert raw motor readings into kinematic joint coordinates:

```bash
rosrun unitree_arm_ros calibrate_joint_states.py
```

This node:

- Subscribes to `/joint_states`
- Applies saved encoder offsets
- Computes differential wrist pitch/roll mapping
- Publishes calibrated `/joint_states_calibrated`

**Calibrated Topics**

| Topic                      | Type                     | Description                                       |
| -------------------------- | ------------------------ | ------------------------------------------------- |
| `/joint_states_calibrated` | `sensor_msgs/JointState` | Kinematic joint states for controllers and MuJoCo |

---

### 🪞 Real → MuJoCo Synchronization

```bash
rosrun unitree_arm_ros mujoco_visualizer.py
```

This node mirrors the real robot configuration into the MuJoCo digital twin:

```
/joint_states → Calibration → MuJoCo qpos → Live Viewer
```

This enables real‑time verification of kinematic consistency and safety before executing learned policies.

<p align="center">
  <img src="images/sim2real.png" width="60%">
</p>

---

### ✋ Kinesthetic Teaching and Playback

```bash
rosrun unitree_arm_ros teach_and_repeat.py
```

Capabilities:

- Gravity‑compensated hand‑guiding
- High‑rate joint trajectory recording (200 Hz)
- Smooth replay with cosine interpolation
- CSV export for debugging or benchmarking

**Data Flow**

```
/joint_states → teach_and_repeat → CSV log
teach_and_repeat → /arm/command → iris_hw_node
```

---

### 📷 RGB‑D and TF Streaming

The RealSense camera publishes synchronized visual observations:

| Topic                          | Type                     |
| ------------------------------ | ------------------------ |
| `/camera/color/image_raw`      | `sensor_msgs/Image`      |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image`      |
| `/camera/color/camera_info`    | `sensor_msgs/CameraInfo` |
| `/tf`                          | `tf2_msgs/TFMessage`     |
| `/tf_static`                   | `tf2_msgs/TFMessage`     |

These streams provide timestamp‑aligned perception inputs for imitation learning.

---

### 💾 Rosbag Data Recording

Automated SSD‑backed data collection:

```bash
bash calibrated_data_collection -O NAME
```

Features:

- Records directly to external SSD
- Automatic filename tagging with goal and timestamp
- Automatic bag chunking every 100 s
- LZ4 compression

**Recorded Topics**

```
/arm/command
/joint_states
/tf
/tf_static
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_rect_raw
/camera/depth/camera_info
```

Each bag is later converted into episode folders for training.

---

### Human expert data collection demonstrations data collection

<p align="center">
  <img src="videos/data_collection_iris-ezgif.com-video-to-gif-converter.gif" width="60%">
</p>

### Semi-autonumous data collection deployment

<p align="center">
  <img src="videos/semi_automous_data_collection.gif" width="60%">
</p>

---

### 📦 Dataset Output Structure

```
processed_data/<bag_prefix>_episode_0001/
 ├── rgb/                 # RGB frames
 ├── depth/              # Depth frames
 └── robot/joint_states.csv
```

These episodes are directly consumed by the imitation learning pipeline.

---

## 5️⃣ Rosbag Data Processing and Episode Generation

Raw ROS bag recordings are converted into structured, learning-ready episodes using an interactive dataset builder.
This tool aligns RGB-D frames with robot joint states, performs timestamp interpolation, and exports synchronized multimodal trajectories for imitation learning.

---

### 🧠 Processing Pipeline Overview

```
Rosbag (.bag)
 ├── /camera/color/image_raw         → RGB frames
 ├── /camera/depth/image_rect_raw   → Depth frames
 └── /joint_states                 → Robot joint states
        ↓
Temporal alignment (RGB ↔ Depth)
        ↓
Joint-state interpolation at camera timestamps
        ↓
Interactive episode slicing
        ↓
Structured episode folders
```

---

### ⚙️ Running the Dataset Builder

```bash
cd bag_reader/scripts

python process_rosbag.py \
  --bag /media/jerry/SSD/rosbag_data/demo_name_YYYYMMDD_HHMMSS.bag \
  --out /media/jerry/SSD/processed_data
```

This launches an interactive episode editor window.

---

### 🎛 Interactive Episode Cutter Controls

| Key       | Action                    |
| --------- | ------------------------- |
| ← / →     | Move START frame          |
| ↑ / ↓     | Move END frame            |
| **p**     | Playback selected segment |
| **Enter** | Save current episode      |
| **ESC**   | Exit                      |

During preview, the UI displays:

- RGB frames
- Normalized depth frames
- Joint angles at start and end frames

This allows precise trimming of clean demonstration segments.

---

### 🔄 RGB–Depth Temporal Alignment

Because RGB and depth cameras publish asynchronously, the script:

1. Estimates the initial timestamp offset
2. Trims streams to equal length
3. Guarantees one-to-one RGB–Depth correspondence

---

### 📐 Joint-State Interpolation

Robot joint states are typically published at a higher rate than camera frames.
To synchronize modalities, joint positions are interpolated at camera timestamps:

[
\mathbf{q}(t_c) = \mathrm{interp}(t_c, {t_j, \mathbf{q}_j})
]

This ensures each exported image frame has a precisely aligned robot configuration.

---

### 📦 Output Episode Structure

```
processed_data/<bag_prefix>_episode_0001/
 ├── rgb/                  # RGB frames (frame_XXXX.png)
 ├── depth/                # Depth frames (frame_XXXX.png)
 ├── robot/
 │    └── joint_states.csv
 └── meta.json             # Episode timing metadata
```

---

### 📄 joint_states.csv Format

| Column      | Description                 |
| ----------- | --------------------------- |
| timestamp   | Camera frame timestamp      |
| pos_joint_i | Joint positions (rad)       |
| vel_joint_i | Joint velocities (optional) |
| eff_joint_i | Joint efforts (optional)    |

---

### 🧾 meta.json

Each episode also includes metadata:

```json
{
  "start_index": 120,
  "end_index": 360,
  "num_frames": 241,
  "t_start": 1736649201.23,
  "t_end": 1736649205.87
}
```

This allows exact temporal reconstruction for evaluation.

---

## 6️⃣ Imitation Learning

Visuomotor imitation learning is implemented using an **Action‑Conditioned Transformer (ACT)** that predicts short‑horizon future joint trajectories conditioned on RGB observations, robot joint states, and a target goal.

<p align="center">
  <img src="docs/media/act_architecture.png" width="85%">
</p>

---

### 🧩 Model Architecture

**Core components**

| Module        | File                          | Description                                                                                 |
| ------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| ACT Model     | `models/transformer_model.py` | Action‑Conditioned Transformer with ResNet34 visual encoder and transformer encoder–decoder |
| Loss Function | `losses/loss.py`              | Trajectory reconstruction + continuity + goal consistency loss                              |
| Kinematics    | `kinematics.py`               | Forward kinematics used for Cartesian evaluation                                            |

**Loss formulation**

The training objective combines three terms:

L = L_mse + λ_cont · L_cont + λ_goal · L_goal

Trajectory reconstruction: L*mse = || Δq̂*(1:F) − Δq\_(1:F) ||²
Continuity regularization: L_cont = || Δq̂_1 ||²
Goal consistency: L_goal = || (q_t + Δq̂_F) − q_goal ||²

This encourages smooth initial motion, accurate trajectory imitation, and convergence to the goal configuration.

---

### ⚙️ Dataset Interface

Training windows are loaded using:

`datasets/iris_dataset.py → EpisodeWindowDataset`

Each sample provides:

| Tensor     | Shape               | Description                      |
| ---------- | ------------------- | -------------------------------- |
| rgb        | (B, S, 3, 128, 128) | RGB observation sequence         |
| joints     | (B, S, 6)           | Joint states                     |
| goal_xyz   | (B, 3)              | Cartesian goal position          |
| fut_delta  | (B, F, 6)           | Ground‑truth future joint deltas |
| goal_joint | (B, 6)              | Target joint configuration       |

---

### 🏋️ Training

Train an ACT policy using the main training script:

```bash
cd il_training
python train.py \
  --data_dir /media/jerry/SSD/processed_data \
  --name iris_goal_exp1 \
  --epochs 80 \
  --batch_size 32
```

**Outputs**

```
checkpoints/best_iris_goal_exp1.pth
checkpoints/final_iris_goal_exp1.pth
plots/history_iris_goal_exp1.csv
plots/loss_iris_goal_exp1.png
```

---

### 📊 Training Curves

<p align="center">
  <img src="images/training_loss.png" width="60%">
</p>

---

### 🧪 Evaluation

Evaluate a trained model on held‑out test episodes:

```bash
cd il_training
python test.py
```

Evaluation pipeline:

1. Loads trained checkpoint
2. Predicts final joint state
3. Applies forward kinematics
4. Computes Cartesian goal error

---

### 🧾 Configuration Files

Training hyperparameters are defined in:

```
configs/train.yaml
```

Includes model size, training schedule, and loss weights.

---

### 🏁 Summary Pipeline

## Processed Episodes → Dataset → ACT Model → ACT Loss → Training → Checkpoints → Evaluation

## 7️⃣ Sim-to-Real Deployment

Planned or learned trajectories can be:

- Previewed in MuJoCo
- Executed on the real robot
- Logged via ROS
- Replayed in simulation

### ▶️ Execute Learned Policy

```bash
python il_training/deploymemnt.py --model models/best_act_iris_goal_exp1.pth
```

<p align="center">
  <img src="docs/media/real_robot.gif" width="75%">
</p>

---

## 💻 System Requirements

- Python ≥ 3.9
- MuJoCo ≥ 2.3
- ROS Noetic
- Intel RealSense RGB-D camera
- Unitree GO-M8010-6 actuators
- NVIDIA GPU recommended for IL training

---

## 📄 Citation

```
@article{cheng2026iris,
  title={IRIS: Learning-Driven Task-Specific Robot Arm for Visuomotor Motion Control},
  author={Cheng, Qilong and others},
  journal={Under Review},
  year={2026}
}
```

---

## 📧 Contact

**Qilong (Jerry) Cheng**
NYU Robotics
[qc1007@nyu.edu](mailto:qc1007@nyu.edu)
