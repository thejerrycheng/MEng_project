#!/usr/bin/env python3
import os
import sys
import rospy
import yaml
import torch
import numpy as np
import cv2
from collections import deque
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

# ------------------------------------------------------------
# Import Model
# ------------------------------------------------------------
# Ensure this script is run from the root folder (where 'models' is)
from models.transformer_model import ACT_RGB

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Update this path to your actual calibration file location
CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints/best_experiment_01.pth")

# Topics
CAMERA_TOPIC = "/camera/color/image_raw"
JOINT_INPUT_TOPIC = "/joint_states_calibrated"  # Input (Model Frame)
COMMAND_TOPIC = "/arm/command"                  # Output (HW Frame)

# Control Parameters
CONTROL_RATE = 15       # Hz (Inference loop rate)
MAX_DELTA_RAD = 0.15    # Safety: Max change per step (approx 8.5 deg)

# Joint Names
# 1. Names the MODEL expects (Calibrated/MuJoCo Frame)
CALIB_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "wrist_pitch", "wrist_roll"
]

# 2. Names the ROBOT HW expects (Raw Frame)
RAW_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

NUM_JOINTS = 6

# Differential Wrist Math Constants (Must match your calibrator)
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0
ROLL_HOME_OFFSET = np.pi

# Image Normalization (ImageNet Stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_calibration(path):
    if not os.path.exists(path):
        rospy.logwarn(f"Calibration file not found at {path}. Using 0 offsets.")
        return {}
    with open(path, "r") as f:
        calib = yaml.safe_load(f)
    return calib.get("joint_offsets", {})


class ACTDeploymentNode:
    def __init__(self):
        rospy.init_node("iris_act_deployment")

        # 1. Load Calibration (For Inverse Calculation)
        self.offsets = load_calibration(CALIB_PATH)

        # 2. Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Inference Device: {self.device}")

        # 3. Load Model
        # These params must match your training arguments!
        self.seq_len = 8
        self.future_steps = 15
        
        self.model = ACT_RGB(
            seq_len=self.seq_len,
            future_steps=self.future_steps,
            d_model=256,
            nhead=8,
            enc_layers=4,
            dec_layers=4,
            ff_dim=1024,
            dropout=0.0
        ).to(self.device)

        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            rospy.loginfo(f"Loaded ACT model from {MODEL_PATH}")
        else:
            rospy.logerr(f"CRITICAL: Model file not found at {MODEL_PATH}")
            sys.exit(1)

        # 4. Buffers
        self.rgb_buffer = deque(maxlen=self.seq_len)
        self.joint_buffer = deque(maxlen=self.seq_len) # Stores CALIBRATED state
        self.bridge = CvBridge()
        
        # Dummy Goal (XYZ)
        # In the future, you can update this via a ROS topic
        self.goal_xyz = torch.tensor([0.3, 0.0, 0.2], device=self.device)

        # 5. ROS Setup
        rospy.Subscriber(CAMERA_TOPIC, Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(JOINT_INPUT_TOPIC, JointState, self.joint_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher(COMMAND_TOPIC, JointState, queue_size=1)

        rospy.loginfo("Deployment Node Ready. Waiting for buffer fill...")

    # ------------------------------------------------------------
    # Inverse Calibration Logic
    # ------------------------------------------------------------
    def calibrate_to_raw(self, q_calib):
        """
        Converts a Model pose (Pitch/Roll) back to Robot HW commands (Left/Right Motors).
        """
        q_raw = np.zeros(6)

        # --- Base Joints (1-4) ---
        # Logic: Model = Raw - Offset  =>  Raw = Model + Offset
        for i in range(4):
            jname = RAW_JOINT_NAMES[i]
            off = self.offsets.get(jname, 0.0)
            q_raw[i] = q_calib[i] + off

        # --- Differential Wrist (5-6) ---
        # Logic: Solving the system of equations from the calibrator
        pitch = q_calib[4]
        roll  = q_calib[5]

        # In calibrator: pitch = SIGN * 0.5 * (q5 - q6)
        # Therefore: (q5 - q6) = pitch / (0.5 * SIGN)
        term_diff = pitch / (0.5 * WRIST_PITCH_SIGN)

        # In calibrator: roll = SIGN * 0.5 * (q5 + q6) + OFFSET
        # Therefore: (q5 + q6) = (roll - OFFSET) / (0.5 * SIGN)
        term_sum  = (roll - ROLL_HOME_OFFSET) / (0.5 * WRIST_ROLL_SIGN)

        # Solve for q5 and q6 (without offsets first)
        q5_val = 0.5 * (term_sum + term_diff)
        q6_val = 0.5 * (term_sum - term_diff)

        # Add hardware offsets back
        q_raw[4] = q5_val + self.offsets.get("joint_5", 0.0)
        q_raw[5] = q6_val + self.offsets.get("joint_6", 0.0)

        return q_raw

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def rgb_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            
            # Normalize to ImageNet Statistics
            img = img.astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()
            self.rgb_buffer.append(img_t)
        except Exception as e:
            rospy.logwarn_throttle(1, f"Image error: {e}")

    def joint_cb(self, msg):
        # Parses /joint_states_calibrated
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q = np.zeros(NUM_JOINTS, dtype=np.float32)

        missing = False
        for i, n in enumerate(CALIB_JOINT_NAMES):
            if n in name_to_idx:
                q[i] = msg.position[name_to_idx[n]]
            else:
                missing = True
        
        if not missing:
            self.joint_buffer.append(torch.from_numpy(q).float())

    def publish_command(self, q_raw):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = RAW_JOINT_NAMES
        msg.position = q_raw.tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(CONTROL_RATE)
        rospy.loginfo("Starting Inference Loop...")

        while not rospy.is_shutdown():
            if len(self.rgb_buffer) != self.seq_len or len(self.joint_buffer) != self.seq_len:
                rate.sleep()
                continue

            # 1. Prepare Inputs
            # Shape: (1, Seq, ...)
            rgb_seq = torch.stack(list(self.rgb_buffer)).unsqueeze(0).to(self.device)
            joint_seq = torch.stack(list(self.joint_buffer)).unsqueeze(0).to(self.device)
            goal_in = self.goal_xyz.unsqueeze(0).to(self.device)

            # 2. Inference
            with torch.no_grad():
                # Model outputs Delta q for future steps
                pred_delta = self.model(rgb_seq, joint_seq, goal_in)

            # 3. Process Prediction (Integration)
            # Take the first predicted step
            dq_next = pred_delta[0, 0, :].cpu().numpy()

            # Safety Clamp
            dq_next = np.clip(dq_next, -MAX_DELTA_RAD, MAX_DELTA_RAD)

            # Get current calibrated position (last in buffer)
            q_curr_calib = self.joint_buffer[-1].cpu().numpy()

            # Add Delta to Current => New Target (Calibrated Frame)
            q_target_calib = q_curr_calib + dq_next

            # 4. Inverse Calibration (Calibrated -> Raw)
            q_target_raw = self.calibrate_to_raw(q_target_calib)

            # 5. Publish to Robot
            self.publish_command(q_target_raw)

            rate.sleep()

if __name__ == "__main__":
    try:
        node = ACTDeploymentNode()
        node.run()
    except rospy.ROSInterruptException:
        pass