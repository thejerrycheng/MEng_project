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
    "joint_4", "wrist_pitch", "wrist_roll"
]

# --- Differential Kinematics Constants ---
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0

# --- HOME OFFSETS (Tune these if flipped) ---
# If the gripper is rotated 180 deg around its axis (upside down), change ROLL to np.pi
ROLL_HOME_OFFSET = 0.0  

# If the wrist is pointing DOWN/BACKWARDS instead of UP, change PITCH to np.pi
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
        rospy.loginfo(f"Roll Offset:  {ROLL_HOME_OFFSET}")
        rospy.loginfo(f"Pitch Offset: {PITCH_HOME_OFFSET}")
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

        # 3. Differential Wrist Math
        q5 = q_zeroed["joint_5"]
        q6 = q_zeroed["joint_6"]

        # Calculate Pitch and Roll with added HOME OFFSETS
        wrist_pitch = (WRIST_PITCH_SIGN * 0.5 * (q5 - q6)) + PITCH_HOME_OFFSET
        wrist_roll  = (WRIST_ROLL_SIGN  * 0.5 * (q5 + q6)) + ROLL_HOME_OFFSET

        q_out[4] = wrist_pitch
        q_out[5] = wrist_roll

        # 4. Publish
        out_msg = JointState()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.name = CALIB_JOINT_NAMES
        out_msg.position = q_out.tolist()
        
        self.pub.publish(out_msg)

        # Debug print (throttle to every 2 seconds roughly)
        # if rospy.get_time() % 2 < 0.1:
        #    print(f"P: {wrist_pitch:.2f} | R: {wrist_roll:.2f}")

if __name__ == "__main__":
    JointCalibrator()
    rospy.spin()