# 🤖 MuJoCo Trajectory Tracking (6-DOF Arm)

This repository contains Python scripts for a 6-DOF robotic arm using **Inverse Dynamics Control**. The controller calculates the necessary torques to compensate for gravity, Coriolis, and centripetal forces while following geometric paths.

## 🛠 Prerequisites

- **MuJoCo:** `pip install mujoco`
- **MuJoCo Viewer:** `pip install mujoco-python-viewer`
- **NumPy:** `pip install numpy`

Ensure your folder structure looks like this:

```text
project_folder/
├── circle_path_tracking.py
├── line_path_tracking.py
└── assets/
    └── iris.xml (and associated .stl meshes)

```

---

## 1. Circular Path Tracking

`circle_path_tracking.py` moves the end-effector along a horizontal (X-Y plane) circle.

**Dynamic Constraint:** The end-effector's **Z-axis (blue)** always points towards the **center of the circle** (visualized as a red sphere).

### Usage

```bash
python circle_path_tracking.py --radius [R] --center [X Y Z]

```

### Examples

- **Standard Circle:**
  `python circle_path_tracking.py --radius 0.15 --center 0.45 0.0 0.4`
- **Small, High Circle:**
  `python circle_path_tracking.py -r 0.08 -c 0.4 0.1 0.5`

---

## 2. Line Path Tracking

`line_path_tracking.py` moves the end-effector back and forth along a 3D line segment.

**Dynamic Constraint:** The end-effector's **Z-axis (blue)** always points towards a **stationary target object** (visualized as a red cube).

### Usage

```bash
python line_path_tracking.py --start [X Y Z] --dir [X Y Z] --length [L] --object [X Y Z]

```

### Examples

- **Horizontal Tracking:** Track a 40cm line along the Y-axis while looking at a cube in the center.

```bash
python line_path_tracking.py --start 0.4 -0.2 0.4 --dir 0 1 0 --length 0.4 --object 0.8 0.0 0.45

```

- **Vertical/Diagonal Tracking:** Move upward and away while looking at an object on the floor.

```bash
python line_path_tracking.py --start 0.35 0.0 0.2 --dir 1 0 1 --length 0.3 --object 0.4 0.0 0.05

```

---

## 🎮 Visualization Legend

| Element                | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| **Large Green Sphere** | The active target point the robot is currently chasing. |
| **Green Dots**         | The pre-calculated desired path (Breadcrumbs).          |
| **Red Cube**           | The "Look-At" target object (only in Line tracking).    |
| **Red Sphere**         | The center of the circle (only in Circle tracking).     |
| **Blue/Red Spheres**   | Start (Blue) and End (Red) points of the line segment.  |

## 🧠 Control Logic Notes

Both scripts use an **Inverse Dynamics** approach:

1. **Inverse Kinematics:** A damped Jacobian Pseudo-inverse maps Cartesian errors (Position + Orientation) to joint space accelerations.
2. **Look-At Algorithm:** A custom rotation matrix is generated every frame to align the end-effector Z-axis with the target vector.
3. **mj_inverse:** MuJoCo’s recursive Newton-Euler algorithm calculates the torques () required to achieve the desired joint accelerations ().
