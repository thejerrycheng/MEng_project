#!/usr/bin/env python3
import os
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
# Joint Naming
# ==================================================
# These must match the names published by your 
# calibrated topic and the order in your MuJoCo XML
MJ_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "wrist_pitch", "wrist_roll"
]

NUM_JOINTS = 6

# ==================================================
class MujocoStateVisualizer:
    def __init__(self):
        rospy.init_node("mujoco_state_visualizer")

        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Buffers
        self.qpos = np.zeros(NUM_JOINTS)
        self.received_state = False

        # ROS Subscriber
        # Now subscribing to the already calibrated topic
        rospy.Subscriber("/joint_states_calibrated", JointState, self.joint_cb, queue_size=1)

        rospy.loginfo("==============================================")
        rospy.loginfo(" MuJoCo Visualizer (Listening to Calibrated)  ")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def joint_cb(self, msg):
        """
        Since the incoming message is already calibrated and converted 
        to kinematic joints (wrist_pitch/roll), we just map 1:1.
        """
        # Create a lookup for the incoming message
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        # Iterate through the joints we expect in MuJoCo
        for i, target_joint in enumerate(MJ_JOINT_NAMES):
            if target_joint in name_to_idx:
                # Direct assignment
                val = msg.position[name_to_idx[target_joint]]
                self.qpos[i] = val
            else:
                # Optional: warn if a specific joint is missing from the stream
                # rospy.logwarn_throttle(1.0, f"Missing joint in stream: {target_joint}")
                pass

        self.received_state = True

    # ------------------------------------------------
    def run(self):
        # 60Hz visualization loop
        rate = rospy.Rate(60)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                if self.received_state:
                    # Update physics state
                    self.data.qpos[:NUM_JOINTS] = self.qpos
                    
                    # Forward kinematics (no step needed, just visualizing)
                    mujoco.mj_forward(self.model, self.data)

                # Sync viewer
                viewer.sync()
                rate.sleep()

# ==================================================
if __name__ == "__main__":
    MujocoStateVisualizer().run()