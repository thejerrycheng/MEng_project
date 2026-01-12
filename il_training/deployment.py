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

# Import your training modules
from models.transformer_model import ACT_RGB
from losses.loss import batch_fk  # not used directly, but handy for debug

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "outputs/models/best_model.pth")

CAMERA_TOPIC = "/camera/color/image_raw"
JOINT_TOPIC  = "/joint_states_calibrated"
COMMAND_TOPIC = "/arm/command"

CONTROL_RATE = 10  # Hz (policy rate, not motor rate)

JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","joint_5","joint_6"
]

NUM_JOINTS = 6

# ------------------------------------------------------------
class ACTDeploymentNode:
    def __init__(self):
        rospy.init_node("iris_act_deployment")

        # Load config
        self.cfg = yaml.safe_load(open(CONFIG_PATH))

        self.seq_len = self.cfg["seq_len"]
        self.future_steps = self.cfg["future_steps"]

        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = ACT_RGB(
            seq_len=self.cfg["seq_len"],
            future_steps=self.cfg["future_steps"],
            d_model=self.cfg["d_model"],
            nhead=self.cfg["nhead"],
            enc_layers=self.cfg["enc_layers"],
            dec_layers=self.cfg["dec_layers"],
            ff_dim=self.cfg["ff_dim"],
            dropout=self.cfg["dropout"]
        ).to(self.device)

        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        rospy.loginfo("Loaded ACT model from: %s", MODEL_PATH)

        # Buffers
        self.rgb_buffer = deque(maxlen=self.seq_len)
        self.joint_buffer = deque(maxlen=self.seq_len)

        self.bridge = CvBridge()
        self.latest_goal_xyz = None  # static goal per run if desired

        # ROS I/O
        rospy.Subscriber(CAMERA_TOPIC, Image, self.rgb_cb, queue_size=1)
        rospy.Subscriber(JOINT_TOPIC, JointState, self.joint_cb, queue_size=1)

        self.cmd_pub = rospy.Publisher(COMMAND_TOPIC, JointState, queue_size=1)

        rospy.loginfo("======================================")
        rospy.loginfo(" IRIS ACT Deployment Node Running")
        rospy.loginfo(" Subscribing RGB: %s", CAMERA_TOPIC)
        rospy.loginfo(" Subscribing Joints: %s", JOINT_TOPIC)
        rospy.loginfo(" Publishing Command: %s", COMMAND_TOPIC)
        rospy.loginfo(" Policy Rate: %d Hz", CONTROL_RATE)
        rospy.loginfo("======================================")

    # ------------------------------------------------------------
    def rgb_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128,128))
            img = torch.tensor(img).permute(2,0,1).float()/255.0
            self.rgb_buffer.append(img)
        except:
            pass

    # ------------------------------------------------------------
    def joint_cb(self, msg):
        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        q = np.zeros(NUM_JOINTS)

        for i,n in enumerate(JOINT_NAMES):
            if n in name_to_idx:
                q[i] = msg.position[name_to_idx[n]]

        self.joint_buffer.append(torch.tensor(q, dtype=torch.float32))

        # Goal = final joint of demonstration
        # For deployment we keep last joint in buffer as current goal.
        # (If you want fixed goal, load from file instead)
        if self.latest_goal_xyz is None:
            # dummy placeholder → not used by model at inference,
            # since model was trained goal-conditioned
            self.latest_goal_xyz = torch.zeros(3)

    # ------------------------------------------------------------
    def ready(self):
        return len(self.rgb_buffer) == self.seq_len and len(self.joint_buffer) == self.seq_len

    # ------------------------------------------------------------
    def publish_command(self, q_cmd):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_cmd.tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(CONTROL_RATE)

        while not rospy.is_shutdown():
            if not self.ready():
                rate.sleep()
                continue

            # Build batch tensors
            rgb_seq = torch.stack(list(self.rgb_buffer)).unsqueeze(0).to(self.device)
            joint_seq = torch.stack(list(self.joint_buffer)).unsqueeze(0).to(self.device)
            goal_xyz = torch.tensor(self.latest_goal_xyz).unsqueeze(0).to(self.device)

            # Forward policy
            with torch.no_grad():
                pred_delta = self.model(rgb_seq, joint_seq, goal_xyz)

            # Take first predicted step Δq
            dq = pred_delta[0,0,:].cpu().numpy()

            # Current joint
            q_curr = self.joint_buffer[-1].cpu().numpy()

            # Absolute command
            q_cmd = q_curr + dq

            # Publish
            self.publish_command(q_cmd)

            rate.sleep()


# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        node = ACTDeploymentNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
