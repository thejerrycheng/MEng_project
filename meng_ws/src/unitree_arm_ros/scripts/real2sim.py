#!/usr/bin/env python3
import os
import yaml
import numpy as np
import rospy
import mujoco
import mujoco.viewer

from sensor_msgs.msg import JointState

# ==================================================
# Paths (ABSOLUTE, mesh-safe)
# ==================================================
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"MuJoCo XML not found: {XML_PATH}")
if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(f"Calibration file not found: {CALIB_PATH}")

# ==================================================
# Joint naming
# ==================================================

# REAL ROBOT / ROS JOINT STATES (motor space)
REAL_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",   # wrist motor A
    "joint_6",   # wrist motor B
]

# MUJOCO GENERALIZED COORDINATES (kinematic space)
MJ_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "wrist_pitch",
    "wrist_roll",
]

NUM_JOINTS = 6

# ==================================================
# Differential wrist conventions (EXPLICIT)
# ==================================================

# Flip signs if MuJoCo motion is opposite to real robot
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0

# 180° roll correction at home (frame mismatch)
ROLL_HOME_OFFSET = np.pi

# ==================================================
def load_calibration(path):
    with open(path, "r") as f:
        calib = yaml.safe_load(f)
    return calib["joint_offsets"]

# ==================================================
class MujocoStateVisualizer:
    def __init__(self):
        rospy.init_node("mujoco_state_visualizer")

        # Load calibration offsets
        self.offsets = load_calibration(CALIB_PATH)

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Home pose (zero after calibration)
        self.home_qpos = np.zeros(NUM_JOINTS)

        # State buffers
        self.qpos = np.zeros(NUM_JOINTS)
        self.received_state = False

        # ROS subscriber
        self.sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_state_callback,
            queue_size=1
        )

        rospy.loginfo("==============================================")
        rospy.loginfo(" MuJoCo Real → Sim State Visualizer ")
        rospy.loginfo("==============================================")
        rospy.loginfo("Joint mapping:")
        for r, m in zip(REAL_JOINT_NAMES, MJ_JOINT_NAMES):
            rospy.loginfo(f"  {r} → {m}")
        rospy.loginfo(f"WRIST_PITCH_SIGN = {WRIST_PITCH_SIGN}")
        rospy.loginfo(f"WRIST_ROLL_SIGN  = {WRIST_ROLL_SIGN}")
        rospy.loginfo(f"ROLL_HOME_OFFSET = {ROLL_HOME_OFFSET} rad")

    # ------------------------------------------------
    def joint_state_callback(self, msg):
        """
        Convert real robot joint states → MuJoCo qpos
        - calibration offsets
        - differential wrist mapping
        - sign conventions
        - roll frame correction
        """
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        # -------- Base joints --------
        for i in range(4):
            name = REAL_JOINT_NAMES[i]
            if name in name_to_idx:
                raw = msg.position[name_to_idx[name]]
                self.qpos[i] = raw - self.offsets.get(name, 0.0)

        # -------- Differential wrist --------
        if "joint_5" in name_to_idx and "joint_6" in name_to_idx:
            q5 = msg.position[name_to_idx["joint_5"]] - self.offsets.get("joint_5", 0.0)
            q6 = msg.position[name_to_idx["joint_6"]] - self.offsets.get("joint_6", 0.0)

            # Differential decomposition
            wrist_pitch = WRIST_PITCH_SIGN * 0.5 * (q5 - q6)
            wrist_roll  = WRIST_ROLL_SIGN  * 0.5 * (q5 + q6) + ROLL_HOME_OFFSET

            self.qpos[4] = wrist_pitch
            self.qpos[5] = wrist_roll

        self.received_state = True

    # ------------------------------------------------
    def run(self):
        rate = rospy.Rate(60)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                if self.received_state:
                    self.data.qpos[:NUM_JOINTS] = self.qpos - self.home_qpos
                    mujoco.mj_forward(self.model, self.data)

                viewer.sync()
                rate.sleep()

# ==================================================
if __name__ == "__main__":
    MujocoStateVisualizer().run()
