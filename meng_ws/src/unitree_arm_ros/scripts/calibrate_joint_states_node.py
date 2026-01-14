#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

# --- Output Topic ---
TOPIC_NAME = "/joint_states_calibrated"

# --- Names ---
REAL_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

CALIB_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

# --- Differential Kinematics Constants (MATCHING VISUALIZER) ---
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0

# --- HOME OFFSETS (MATCHING VISUALIZER) ---
# The visualizer uses +PI for the roll alignment
ROLL_HOME_OFFSET = np.pi   
PITCH_HOME_OFFSET = 0.0 

# ==================================================
def load_calibration(path):
    try:
        with open(path, "r") as f:
            calib = yaml.safe_load(f)
        rospy.loginfo(f"Loaded offsets from {path}")
        return calib["joint_offsets"]
    except Exception as e:
        rospy.logerr(f"Could not load calibration file: {e}")
        return {}

class JointCalibrator:
    def __init__(self):
        rospy.init_node("joint_state_calibrator")

        self.offsets = load_calibration(CALIB_PATH)
        
        self.pub = rospy.Publisher(TOPIC_NAME, JointState, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self.cb, queue_size=10)

        rospy.loginfo("--------------------------------------------------")
        rospy.loginfo(f"Publishing to: {TOPIC_NAME}")
        rospy.loginfo(f"Roll Offset:  {ROLL_HOME_OFFSET:.4f}")
        rospy.loginfo("--------------------------------------------------")

    def cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        # 1. Zero out the raw values using calibration file
        q_zeroed = {}
        for jname in REAL_JOINT_NAMES:
            raw = msg.position[name_to_idx[jname]] if jname in name_to_idx else 0.0
            off = self.offsets.get(jname, 0.0)
            q_zeroed[jname] = raw - off

        # 2. Base Joints
        q_out = np.zeros(6)
        q_out[0] = q_zeroed["joint_1"]
        q_out[1] = q_zeroed["joint_2"]
        q_out[2] = q_zeroed["joint_3"]
        q_out[3] = q_zeroed["joint_4"]
        q_out[4] = q_zeroed["joint_5"]
        q_out[5] = q_zeroed["joint_6"]

        # 4. Publish
        out_msg = JointState()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.name = CALIB_JOINT_NAMES
        out_msg.position = q_out.tolist()
        
        self.pub.publish(out_msg)

if __name__ == "__main__":
    JointCalibrator()
    rospy.spin()