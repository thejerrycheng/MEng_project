# 🎥 **IRIS: Learning-Driven Task-Specific Robot Arm for Visuomotor Motion Control**

> **IRIS (Intelligent Robotic Imaging System)** is a low-cost, 3D-printed 6-DOF cinema robot arm that learns smooth, repeatable, and obstacle-aware camera motions through visuomotor imitation learning.
> This repository contains the complete **hardware, simulation, control, ROS data collection, and learning stack**, enabling seamless **sim-to-real and real-to-sim** workflows.

<p align="center">
  <img src="docs/media/iris_hero.gif" width="85%">
</p>

<p align="center">
  <a href="#1-hardware-platform">1️⃣ Hardware</a> •
  <a href="#2-low-level-actuator-control">2️⃣ Actuator Control</a> •
  <a href="#3-mujoco-simulation">3️⃣ MuJoCo Simulation</a> •
  <a href="#4-ros-interface-and-rosbag-collection">4️⃣ ROS + Rosbags</a> •
  <a href="#5-data-processing">5️⃣ Data Processing</a> •
  <a href="#6-imitation-learning">6️⃣ Imitation Learning</a> •
  <a href="#7-sim-to-real-deployment">7️⃣ Sim-to-Real</a>
</p>

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
  <img src="videos/motor.gif" width="30%">
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
python cinema_line_tracking.py
python circle_path_tracking.py
python line_path_tracking.py
```

<p align="center">
  <img src="videos/apf.gif" width="50%">
</p>
<p align="center">
  <img src="videos/rrt.gif" width="50%">
</p>

### ▶️ Interactive Teleoperation

```bash
python teleop_ik.py     # Cartesian IK teleoperation
python teleop_fk.py     # Joint-space teleoperation
```

<p align="center">
  <img src="videos/apf.gif" width="70%">
</p>

---

## 4️⃣ ROS Interface and Rosbag Collection

ROS provides synchronized real-time streaming of:

- Robot joint states and commands
- RGB-D camera frames (Intel RealSense)
- TF transforms and timestamps
- MuJoCo real–sim mirroring

### ⚙️ Build ROS Workspace

```bash
cd meng_ws
catkin_make
source devel/setup.bash
```

### ▶️ Launch Hardware Interface

```bash
rosrun unitree_arm_ros unitree_hw_node.py
```

### ▶️ Keyboard Joint Teleoperation

```bash
rosrun unitree_arm_ros keyboard_joint_teleop.py
```

### ▶️ MuJoCo Visualization of Real Robot

```bash
rosrun unitree_arm_ros mujoco_visualizer.py
```

<p align="center">
  <img src="docs/media/ros_mujoco_sync.gif" width="50%">
</p>

<p align="center">
  <img src="docs/media/" width="50%">
</p>

### ▶️ Record Rosbag Demonstrations

```bash
cd bag_reader/scripts
bash iris_rosbag_record.sh -O demo_name
```

Recorded topics include:

```
/arm/command
/joint_states
/tf
/camera/color/image_raw
/camera/depth/image_rect_raw
```

<p align="center">
  <img src="videos/data_collection_iris-ezgif.com-video-to-gif-converter.gif" width="66%">
</p>
<p align="center">
  <img src="videos/semi_automous_data_collection.gif" width="50%">
</p>

---

## 5️⃣ Data Processing

Recorded rosbags are converted into structured datasets for imitation learning.

### ▶️ Process a Rosbag

```bash
cd bag_reader/scripts
bash process_bag.sh --bag demo_name_YYYYMMDD_HHMMSS
```

### 📦 Output Dataset Structure

```
processed_data/<bag_prefix>_episode_0001/
 ├── rgb/                 # RGB frames
 ├── depth/               # Depth frames
 └── robot/joint_states.csv
```

These episodes are directly consumed by the IL training pipeline.

---

## 6️⃣ Imitation Learning

Visuomotor imitation learning is implemented using an **Action-Conditioned Transformer (ACT)** that predicts future joint trajectories conditioned on RGB observations and robot state.

<p align="center">
  <img src="docs/media/act_architecture.png" width="85%">
</p>

### ▶️ Train Policy

```bash
python il_training/train_act.py \
  --data_root /media/jerry/SSD/processed_data \
  --bag_prefix 0.3_0.3_0.3_goal_20260111_181031 \
  --name iris_goal_exp1 \
  --epochs 200
```

### 📈 Outputs

```
models/best_act_iris_goal_exp1.pth
plots/loss_iris_goal_exp1.png
plots/loss_iris_goal_exp1.csv
```

<p align="center">
  <img src="docs/media/loss_curve.png" width="60%">
</p>

---

## 7️⃣ Sim-to-Real Deployment

Planned or learned trajectories can be:

- Previewed in MuJoCo
- Executed on the real robot
- Logged via ROS
- Replayed in simulation

### ▶️ Execute Learned Policy

```bash
python mpr_control/run_policy.py --model models/best_act_iris_goal_exp1.pth
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
