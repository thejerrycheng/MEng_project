#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

# --- INPUT: The Ground Truth World Frame Commands ---
# Your planner/controller should publish to this topic
INPUT_TOPIC = "/joint_commands_calibrated"

# --- OUTPUT: The Raw Actuator Commands ---
# The Unitree Driver usually listens to this topic
OUTPUT_TOPIC = "/joint_commands"
# --- Names ---
# The names coming from your planner (World Frame)
WORLD_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

# The names expected by the Actuators (Raw Frame)
ACTUATOR_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

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

class JointCommandCalibrator:
    def __init__(self):
        rospy.init_node("joint_command_calibrator")

        # Load the SAME calibration file used for state estimation
        self.offsets = load_calibration(CALIB_PATH)
        
        # Publisher to the Robot Driver
        self.pub = rospy.Publisher(OUTPUT_TOPIC, JointState, queue_size=10)
        
        # Subscriber to the High-Level Controller
        rospy.Subscriber(INPUT_TOPIC, JointState, self.cb, queue_size=10)

        rospy.loginfo("--------------------------------------------------")
        rospy.loginfo(" COMMAND CALIBRATOR RUNNING")
        rospy.loginfo(f" Listening to: {INPUT_TOPIC} (World Frame)")
        rospy.loginfo(f" Publishing to: {OUTPUT_TOPIC} (Actuator Frame)")
        rospy.loginfo(" Logic: Raw_Cmd = World_Cmd + Offset")
        rospy.loginfo("--------------------------------------------------")
 
    def cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        
        # Dictionary to store calculated raw commands
        raw_cmds = {}

        # 1. Apply Inverse Calibration (Add Offset)
        for jname in WORLD_JOINT_NAMES:
            # Get the world command (default to 0.0 if missing)
            world_val = msg.position[name_to_idx[jname]] if jname in name_to_idx else 0.0
            
            # Get the offset (same file as state calibrator)
            off = self.offsets.get(jname, 0.0)
            
            # REVERSE LOGIC: Add the offset to get back to raw motor frame
            raw_cmds[jname] = world_val + off

        # 2. Construct Output Array (Ordered for Actuators)
        q_out = np.zeros(6)
        q_out[0] = raw_cmds["joint_1"]
        q_out[1] = raw_cmds["joint_2"]
        q_out[2] = raw_cmds["joint_3"]
        q_out[3] = raw_cmds["joint_4"]
        q_out[4] = raw_cmds["joint_5"]
        q_out[5] = raw_cmds["joint_6"]

        # 3. Publish Raw Command
        out_msg = JointState()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.name = ACTUATOR_JOINT_NAMES
        out_msg.position = q_out.tolist()
        
        # Pass through velocity/effort if they exist, but generally
        # offsets don't affect velocity (derivative of constant is 0)
        if len(msg.velocity) > 0:
            out_msg.velocity = msg.velocity 
        if len(msg.effort) > 0:
            out_msg.effort = msg.effort

        self.pub.publish(out_msg)

if __name__ == "__main__":
    JointCommandCalibrator()
    rospy.spin()