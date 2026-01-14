#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

# --- TOPICS ---
TOPIC_STATE_RAW       = "/joint_states"             
TOPIC_STATE_CALIB     = "/joint_states_calibrated"  

TOPIC_CMD_CALIB       = "/joint_commands_calibrated" 
TOPIC_CMD_RAW         = "/joint_commands"            

# --- JOINTS ---
JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

NUM_JOINTS = 6

# --- DIFFERENTIAL WRIST CONSTANTS (STRICTLY FROM VISUALIZER) ---
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0
ROLL_HOME_OFFSET = np.pi   

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

class CalibrationNode:
    def __init__(self):
        rospy.init_node("calibration_node")

        # Load offsets
        self.offsets = load_calibration(CALIB_PATH)

        # --------------------------------------------------------
        # Direction 1: State (Raw Motors -> Calibrated Pitch/Roll)
        # --------------------------------------------------------
        self.pub_state_calib = rospy.Publisher(TOPIC_STATE_CALIB, JointState, queue_size=10)
        self.sub_state_raw   = rospy.Subscriber(TOPIC_STATE_RAW, JointState, self.state_cb, queue_size=10)

        # --------------------------------------------------------
        # Direction 2: Command (Calibrated Pitch/Roll -> Raw Motors)
        # --------------------------------------------------------
        self.pub_cmd_raw   = rospy.Publisher(TOPIC_CMD_RAW, JointState, queue_size=10)
        self.sub_cmd_calib = rospy.Subscriber(TOPIC_CMD_CALIB, JointState, self.command_cb, queue_size=10)

        rospy.loginfo("==================================================")
        rospy.loginfo(" CALIBRATION NODE RUNNING (STRICT MATCH)")
        rospy.loginfo(f" Roll Offset: {ROLL_HOME_OFFSET:.4f} rad")
        rospy.loginfo(" Signs: Pitch(-1.0), Roll(-1.0)")
        rospy.loginfo("==================================================")

    # ============================================================
    # Callback: STATE (Raw -> World)
    # Logic: Exactly matches MujocoStateVisualizer
    # ============================================================
    def state_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q_out = np.zeros(NUM_JOINTS)

        # 1. Base Joints (1-4): Simple Offset
        for i in range(4):
            jname = JOINT_NAMES[i]
            if jname in name_to_idx:
                raw_val = msg.position[name_to_idx[jname]]
                off = self.offsets.get(jname, 0.0)
                q_out[i] = raw_val - off

        # 2. Wrist Joints (5-6): Differential Mapping
        if "joint_5" in name_to_idx and "joint_6" in name_to_idx:
            # A. Get Zeroed Motor Positions (Raw - Offset)
            raw_q5 = msg.position[name_to_idx["joint_5"]]
            raw_q6 = msg.position[name_to_idx["joint_6"]]
            
            q5_zeroed = raw_q5 - self.offsets.get("joint_5", 0.0)
            q6_zeroed = raw_q6 - self.offsets.get("joint_6", 0.0)

            # B. Apply Differential Formula (Strictly from visualizer)
            # Pitch = -0.5 * (q5 - q6)
            # Roll  = -0.5 * (q5 + q6) + pi
            wrist_pitch = WRIST_PITCH_SIGN * 0.5 * (q5_zeroed - q6_zeroed)
            wrist_roll  = WRIST_ROLL_SIGN  * 0.5 * (q5_zeroed + q6_zeroed) + ROLL_HOME_OFFSET

            q_out[4] = wrist_pitch
            q_out[5] = wrist_roll

        # Publish
        out_msg = JointState()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.name = JOINT_NAMES
        out_msg.position = q_out.tolist()
        self.pub_state_calib.publish(out_msg)

    # ============================================================
    # Callback: COMMAND (World -> Raw)
    # Logic: Inverse of MujocoStateVisualizer
    # ============================================================
    def command_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q_raw_out = np.zeros(NUM_JOINTS)

        # 1. Base Joints (1-4): Add Offset Back
        for i in range(4):
            jname = JOINT_NAMES[i]
            if jname in name_to_idx:
                world_val = msg.position[name_to_idx[jname]]
                off = self.offsets.get(jname, 0.0)
                q_raw_out[i] = world_val + off

        # 2. Wrist Joints (5-6): Inverse Differential Mapping
        if "joint_5" in name_to_idx and "joint_6" in name_to_idx:
            P = msg.position[name_to_idx["joint_5"]] # Desired Pitch
            R = msg.position[name_to_idx["joint_6"]] # Desired Roll

            # --- INVERSE DERIVATION ---
            # Forward Equations:
            # Pitch = S_p * 0.5 * (q5 - q6)
            # Roll  = S_r * 0.5 * (q5 + q6) + Roll_Off
            
            # Step 1: Isolate (q5 - q6) and (q5 + q6)
            # (q5 - q6) = 2 * Pitch / S_p
            # (q5 + q6) = 2 * (Roll - Roll_Off) / S_r
            
            diff_term = (2.0 * P) / WRIST_PITCH_SIGN
            sum_term  = (2.0 * (R - ROLL_HOME_OFFSET)) / WRIST_ROLL_SIGN

            # Step 2: Solve for q5, q6
            # (q5 + q6) + (q5 - q6) = 2*q5
            # (q5 + q6) - (q5 - q6) = 2*q6
            q5_zeroed = 0.5 * (sum_term + diff_term)
            q6_zeroed = 0.5 * (sum_term - diff_term)

            # Step 3: Add Calibration Offsets back
            q_raw_out[4] = q5_zeroed + self.offsets.get("joint_5", 0.0)
            q_raw_out[5] = q6_zeroed + self.offsets.get("joint_6", 0.0)

        # Publish
        out_msg = JointState()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.name = JOINT_NAMES
        out_msg.position = q_raw_out.tolist()
        self.pub_cmd_raw.publish(out_msg)

if __name__ == "__main__":
    CalibrationNode()
    rospy.spin()