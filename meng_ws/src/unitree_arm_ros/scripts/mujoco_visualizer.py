#!/usr/bin/env python3
import os
import yaml
import numpy as np
import rospy
import mujoco
import mujoco.viewer

from sensor_msgs.msg import JointState

# ==================================================
# Absolute Paths
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
# Joint Naming
# ==================================================
REAL_JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","joint_5","joint_6"
]

MJ_JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","wrist_pitch","wrist_roll"
]

NUM_JOINTS = 6

# ==================================================
# Differential Wrist Mapping
# ==================================================
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0
ROLL_HOME_OFFSET = np.pi   # frame alignment

# ==================================================
def load_calibration(path):
    with open(path,"r") as f:
        calib = yaml.safe_load(f)
    return calib["joint_offsets"]

# ==================================================
class MujocoStateVisualizer:
    def __init__(self):
        rospy.init_node("mujoco_state_visualizer")

        # Load calibration
        self.offsets = load_calibration(CALIB_PATH)

        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Buffers
        self.qpos = np.zeros(NUM_JOINTS)
        self.received_state = False

        # ROS Subscriber
        rospy.Subscriber("/joint_states", JointState, self.joint_cb, queue_size=1)

        rospy.loginfo("==============================================")
        rospy.loginfo(" MuJoCo Real → Sim State Visualizer Running ")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def joint_cb(self, msg):
        name_to_idx = {n:i for i,n in enumerate(msg.name)}

        # Base joints
        for i in range(4):
            jname = REAL_JOINT_NAMES[i]
            if jname in name_to_idx:
                raw = msg.position[name_to_idx[jname]]
                self.qpos[i] = raw - self.offsets.get(jname,0.0)

        # Differential wrist
        if "joint_5" in name_to_idx and "joint_6" in name_to_idx:
            q5 = msg.position[name_to_idx["joint_5"]] - self.offsets.get("joint_5",0.0)
            q6 = msg.position[name_to_idx["joint_6"]] - self.offsets.get("joint_6",0.0)

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
                    self.data.qpos[:NUM_JOINTS] = self.qpos
                    mujoco.mj_forward(self.model, self.data)

                viewer.sync()
                rate.sleep()

# ==================================================
if __name__ == "__main__":
    MujocoStateVisualizer().run()
