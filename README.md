# IRIS: Learning-Driven Task-Specific Robot Arm for Visuomotor Motion Control

This repository contains the full hardware, simulation, control, data collection, and learning stack for **IRIS (Intelligent Robotic Imaging System)** — a low-cost, 3D-printed 6-DOF robot arm designed for smooth, repeatable visuomotor motion generation.  
The codebase supports **MuJoCo simulation**, **real-robot control via Unitree BLDC actuators**, **ROS-based data collection**, **classical motion planning**, and **visuomotor imitation learning**, enabling seamless **sim-to-real and real-to-sim** workflows.

---

## Repository Structure

```

MEng_project/
├── mujoco_sim/              # MuJoCo simulation, kinematics, dynamics, planners
├── classical_planner/       # RRT*, potential-field, and trajectory generation
├── mpr_control/             # Real robot low-level control (Unitree actuator SDK)
├── motor_control/           # Motor-level control and diagnostics
├── meng_ws/                 # ROS workspace (hardware interface, teleop, logging)
├── bag_reader/              # ROS bag recording and dataset extraction tools
├── sim2real/                # Sim–real synchronization utilities
├── il_training/             # Visuomotor imitation learning training code
├── inverse_kinematics_sim/  # Analytical and numerical IK solvers
├── paper/                   # LaTeX source for the accompanying paper
└── README.md

```

---

## 1. MuJoCo Simulation Quick Start

The MuJoCo simulator is used for:

- Kinematic verification
- Classical motion planning (RRT\*, potential fields)
- Trajectory preview
- Real–sim synchronization

### Setup

```bash
cd mujoco_sim
bash setup_env.sh
pip install -r requirement.txt
```

### Launch a simulation demo script

```bash
python cinema_line_tracking.py  # Tracking cinematic path
python circle_path_tracking.py  # Trakcing a defined circle
python line_path_tracking.py    # Tracking a line
```

### Launch a teleoperation script

```bash
python teleop_ik.py         # Interactive IK teleoperation
python teleop_fk.py         # Interactive FK teleoperation
```

---

Here is a **concise, professional revision** with the SDK source and download reference added, suitable for a research repo README:

---

## 2. Actuator Control and SDK Download (Unitree BLDC Actuators)

Low-level torque and position control are implemented using the official **Unitree actuator SDK** for GO-series motors.

### SDK Download

The Unitree actuator SDK can be obtained from:

- Official documentation: https://support.unitree.com/home/en/Actuator
- SDK repository (GO-series motors): https://github.com/unitreerobotics/unitree_actuator_sdk

After downloading, place the SDK under:

```

mpr_control/unitree_actuator_sdk/

```

---

### Setup

```bash
cd mpr_control/unitree_actuator_sdk/python
pip install -r requirements.txt
```

---

### Run basic motor test

```bash
python example_goM8010_6_motor.py
python position_teleop.py
python torque_teleop.py
python velocity_teleop.py
```

---

## 3. IRIS Robot Control

### Demo Scripts

```bash
python teach_and_repeat.py   # Teach and repeat script
python calibratoin.py        # This calibrate and save the home position (by default the home is considered pointing upwards)
python repeatability_test.py # This test out the repeatabilty of the robot arm
```

---

### Execute planned trajectories

```bash
python position_tracking.py
```

---

Joint-space impedance control with gravity compensation is implemented on the host computer.
Motor commands and state feedback are communicated over an **RS-485 bus at 1 kHz**, enabling low-latency multi-joint closed-loop control.

---

## 3. ROS Interface and Real–Sim Synchronization

ROS is used for:

- Hardware state publishing
- Command streaming
- Rosbag data recording
- MuJoCo real–sim mirroring

### Build ROS workspace

```bash
cd meng_ws
catkin_make
source devel/setup.bash
```

### Launch hardware interface

```bash
rosrun unitree_arm_ros unitree_hw_node.py
```

### Teleoperation

```bash
rosrun unitree_arm_ros keyboard_ik_teleop.py
```

### Real–Sim bridge

```bash
rosrun unitree_arm_ros mujoco_ik_ros_node.py
```

---

## 4. Dataset Collection

Demonstrations are collected from:

- **Planner-generated trajectories** executed on hardware
- **Kinesthetic drag-and-teach human demonstrations**

### Record ROS bag

```bash
cd bag_reader/scripts
bash iris_rosbag_record.sh
```

### Convert rosbag to training dataset

```bash
python rosbag_reader.py
python process.py
```

Outputs synchronized:

- RGB frames
- Joint positions / velocities
- End-effector trajectories

---

## 5. Imitation Learning Training

Visuomotor imitation learning models are trained in `il_training/`.

### Prepare dataset split

```bash
cd il_training
python train_data_split.py
```

### Train RGB Transformer policy

```bash
python train_rgb_transformer.py
```

### Alternative CNN baselines

```bash
python train_rgb.py
python train_depth_cnn.py
```

Trained models predict short-horizon joint trajectories conditioned on RGB observations and robot state.

---

## 6. Sim-to-Real and Real-to-Sim Execution

Planned or learned trajectories can be:

- Previewed in MuJoCo
- Executed on hardware
- Logged and replayed in simulation

The `sim2real/` and ROS bridge nodes provide time-synchronized state and command streaming for reproducible experiments.

---

<!--
## 7. Paper

The accompanying system and learning framework are described in:

```
paper/
```

Build with:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
```

--- -->

## Requirements

- Python ≥ 3.8
- MuJoCo ≥ 2.3
- ROS Noetic
- NVIDIA GPU recommended for training
- Unitree GO-M8010-6 actuators for hardware experiments

---

## Contact

Qilong (Jerry) Cheng
[qc1007@nyu.edu](mailto:qc1007@nyu.edu)
