#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from sensor_msgs.msg import JointState

CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

REAL_JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","joint_5","joint_6"
]

# MuJoCo / kinematic target naming
CALIB_JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","wrist_pitch","wrist_roll"
]

WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0
ROLL_HOME_OFFSET = np.pi


def load_calibration(path):
    with open(path,"r") as f:
        calib = yaml.safe_load(f)
    return calib["joint_offsets"]


class JointCalibrator:
    def __init__(self):
        rospy.init_node("joint_state_calibrator")

        self.offsets = load_calibration(CALIB_PATH)

        self.pub = rospy.Publisher("/joint_states_calibrated", JointState, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self.cb, queue_size=10)

        rospy.loginfo("Joint calibration relay running...")
        rospy.loginfo("Publishing /joint_states_calibrated")

    def cb(self, msg):
        name_to_idx = {n:i for i,n in enumerate(msg.name)}

        # --- Base joints ---
        q = np.zeros(6)

        for i in range(4):
            jname = REAL_JOINT_NAMES[i]
            raw = msg.position[name_to_idx[jname]]
            q[i] = raw - self.offsets.get(jname,0.0)

        # --- Differential wrist ---
        q5 = msg.position[name_to_idx["joint_5"]] - self.offsets.get("joint_5",0.0)
        q6 = msg.position[name_to_idx["joint_6"]] - self.offsets.get("joint_6",0.0)

        wrist_pitch = WRIST_PITCH_SIGN * 0.5 * (q5 - q6)
        wrist_roll  = WRIST_ROLL_SIGN  * 0.5 * (q5 + q6) + ROLL_HOME_OFFSET

        q[4] = wrist_pitch
        q[5] = wrist_roll

        # --- Publish calibrated JointState ---
        out = JointState()
        out.header = msg.header
        out.name = CALIB_JOINT_NAMES
        out.position = q.tolist()
        self.pub.publish(out)


if __name__ == "__main__":
    JointCalibrator()
    rospy.spin()
