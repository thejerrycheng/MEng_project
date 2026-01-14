#!/usr/bin/env python3
import os
import sys
import argparse
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

if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"MuJoCo XML not found: {XML_PATH}")

# ==================================================
# Configuration
# ==================================================
# The ROS message names we expect to see in the calibrated topic.
# We map these 1-to-1 to MuJoCo qpos[0] through qpos[5].
TARGET_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", 
    "joint_4", "joint_5", "joint_6"
]

NUM_JOINTS = 6

# ==================================================
class MujocoStateVisualizer:
    def __init__(self):
        # 1. Parse Arguments
        parser = argparse.ArgumentParser(description="Visualize Calibrated JointStates")
        parser.add_argument("--topic", type=str, default="/joint_states_calibrated", 
                            help="The ROS topic to visualize")
        
        args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
        self.topic_name = args.topic

        rospy.init_node("mujoco_state_visualizer")

        # 2. Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Buffers
        self.qpos = np.zeros(NUM_JOINTS)
        self.received_state = False

        # 3. Subscribe
        rospy.Subscriber(self.topic_name, JointState, self.joint_cb, queue_size=1)

        rospy.loginfo("==============================================")
        rospy.loginfo(f" Visualizing Topic: {self.topic_name}")
        rospy.loginfo(" Mode: Direct Mapping (No Calibration/Math)")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def joint_cb(self, msg):
        """
        Directly map ROS JointState names to MuJoCo qpos indices.
        Assumes input is already calibrated (World Frame).
        """
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        
        for i, target_name in enumerate(TARGET_JOINT_NAMES):
            if target_name in name_to_idx:
                # Direct assignment: qpos[i] = msg[joint_i]
                self.qpos[i] = msg.position[name_to_idx[target_name]]

        self.received_state = True

    # ------------------------------------------------
    def run(self):
        rate = rospy.Rate(60)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                if self.received_state:
                    # Update Simulation State
                    self.data.qpos[:NUM_JOINTS] = self.qpos
                    mujoco.mj_forward(self.model, self.data)

                viewer.sync()
                rate.sleep()

# ==================================================
if __name__ == "__main__":
    MujocoStateVisualizer().run()