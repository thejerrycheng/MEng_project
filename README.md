# **IRIS: Learning-Driven Task-Specific Robot Arm for Visuomotor Motion Control**

This repository contains the full hardware, simulation, control, data collection, and learning stack for **IRIS (Intelligent Robotic Imaging System)** — a low-cost, 3D-printed 6-DOF robot arm designed for smooth, repeatable visuomotor motion generation.
The codebase supports **MuJoCo simulation**, **real-robot control via Unitree BLDC actuators**, **ROS-based data collection**, **classical motion planning**, and **visuomotor imitation learning**, enabling seamless **sim-to-real and real-to-sim** workflows.

---

## **Repository Structure**

```
MEng_project/
├── mujoco_sim/              # MuJoCo simulation, kinematics, planners
├── classical_planner/       # RRT*, potential-field, and trajectory generation
├── mpr_control/             # Low-level actuator SDK and motor control
├── motor_control/           # Motor diagnostics and testing
├── meng_ws/                 # ROS workspace (hardware interface, teleop, logging)
├── bag_reader/              # Rosbag recording and dataset extraction tools
├── sim2real/                # Sim–real synchronization utilities
├── il_training/             # Visuomotor imitation learning training code
├── inverse_kinematics_sim/  # Analytical and numerical IK solvers
├── paper/                   # LaTeX source for the accompanying paper
└── README.md
```

---

## **1. MuJoCo Simulation**

MuJoCo is used for:

* Kinematic verification
* Classical motion planning (RRT*, potential fields)
* Trajectory preview
* Real–sim synchronization

### **Setup**

```bash
cd mujoco_sim
bash setup_env.sh
pip install -r requirement.txt
```

### **Run simulation demos**

```bash
python cinema_line_tracking.py
python circle_path_tracking.py
python line_path_tracking.py
```

### **Interactive teleoperation**

```bash
python teleop_ik.py     # Cartesian IK teleoperation
python teleop_fk.py     # Joint-space teleoperation
```

---

## **2. Actuator Control and SDK (Unitree BLDC)**

Low-level torque and position control are implemented using the official **Unitree GO-series actuator SDK**.

### **SDK Download**

* Official documentation:
  [https://support.unitree.com/home/en/Actuator](https://support.unitree.com/home/en/Actuator)

* SDK repository:
  [https://github.com/unitreerobotics/unitree_actuator_sdk](https://github.com/unitreerobotics/unitree_actuator_sdk)

Place the SDK at:

```
mpr_control/unitree_actuator_sdk/
```

---

### **Setup**

```bash
cd mpr_control/unitree_actuator_sdk/python
pip install -r requirements.txt
```

---

### **Run basic motor tests**

```bash
python example_goM8010_6_motor.py
python position_teleop.py
python torque_teleop.py
python velocity_teleop.py
```

---

## **3. IRIS Real Robot Control**

Joint-space impedance control with gravity compensation runs on the host PC.
Motors communicate over **RS-485 at 1 kHz**, enabling low-latency multi-joint closed-loop control.

### **Demo scripts**

```bash
python teach_and_repeat.py
python calibration.py          # Save home pose calibration
python repeatability_test.py
```

### **Execute planned trajectories**

```bash
python position_tracking.py
```

---

## **4. ROS Interface and Real–Sim Synchronization**

ROS provides:

* Hardware state publishing
* Command streaming
* Teleoperation
* Rosbag data recording
* MuJoCo real–sim mirroring

### **Build ROS workspace**

```bash
cd meng_ws
catkin_make
source devel/setup.bash
```

### **Launch hardware interface**

```bash
rosrun unitree_arm_ros unitree_hw_node.py
```

### **Keyboard joint teleoperation**

```bash
rosrun unitree_arm_ros keyboard_joint_teleop.py
```

### **MuJoCo IK teleoperation**

```bash
rosrun unitree_arm_ros keyboard_ik_teleop.py
```

### **MuJoCo visualization of real robot**

```bash
rosrun unitree_arm_ros mujoco_visualizer.py
```

---

## **5. Rosbag Data Collection**

All real-robot demonstrations are recorded using ROS bags, including:

* RGB images
* Depth images
* Camera calibration
* Joint feedback
* Joint command targets
* TF transforms
* Timestamps

### **Record a demonstration**

```bash
cd bag_reader/scripts
bash iris_rosbag_record.sh -O demo_name
```

This produces:

```
rosbag_data/demo_name_YYYYMMDD_HHMMSS.bag
```

Recorded topics include:

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

Stop recording with **Ctrl+C**.

---

## **6. Rosbag Dataset Processing**

Recorded rosbags are converted into structured training datasets.

### **Process a bag**

```bash
cd bag_reader/scripts
bash process_bag.sh --bag demo_name_YYYYMMDD_HHMMSS
```

### **Outputs**

```
raw_data/demo_name_YYYYMMDD_HHMMSS/
 ├── rgb/                 # RGB frames (timestamp-named PNGs)
 ├── depth/               # Depth frames (16UC1 PNGs)
 ├── joint_states.csv     # Robot joint feedback
 ├── arm_command.csv      # Commanded joint targets
 ├── camera_info_color.json
 └── camera_info_depth.json
```

These datasets are directly consumed by the imitation learning pipeline.

---

## **7. Imitation Learning Training**

Visuomotor imitation learning models are trained in:

```
il_training/
```

### **Prepare dataset splits**

```bash
cd il_training
python train_data_split.py
```

### **Train RGB policy**

```bash
python train_rgb_transformer.py
```

### **CNN baselines**

```bash
python train_rgb.py
python train_depth_cnn.py
```

Models predict short-horizon joint trajectories conditioned on RGB observations and robot state.

---

## **8. Sim-to-Real and Real-to-Sim**

Planned or learned trajectories can be:

* Previewed in MuJoCo
* Executed on hardware
* Logged on ROS
* Replayed in simulation

Nodes in `sim2real/` and ROS bridge tools provide synchronized streaming for reproducible experiments.

---

## **Requirements**

* Python ≥ 3.8
* MuJoCo ≥ 2.3
* ROS Noetic
* Intel RealSense RGB-D camera
* Unitree GO-M8010-6 actuators
* NVIDIA GPU recommended for training

---

## **Contact**

**Qilong (Jerry) Cheng**
NYU Robotics
[qc1007@nyu.edu](mailto:qc1007@nyu.edu)
