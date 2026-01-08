#!/usr/bin/env python3
import os
import yaml
import numpy as np
import rospy
import mujoco
import mujoco.viewer
import curses

from sensor_msgs.msg import JointState

# ==================================================
# Paths
# ==================================================
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

# ==================================================
# Config
# ==================================================
JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

NUM_JOINTS = 6
RATE_HZ = 200
DT = 1.0 / RATE_HZ

# Teleop velocity (rad/s)
VEL = np.array([0.6, 0.6, 0.6, 0.8, 1.0, 1.0])

# ==================================================
def load_calibration(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["joint_offsets"]

# ==================================================
class KeyboardJointTeleopROS:
    def __init__(self):
        rospy.init_node("keyboard_joint_teleop_ros")

        # ROS
        self.cmd_pub = rospy.Publisher("/arm/command", JointState, queue_size=1)
        self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_cb, queue_size=1)

        # Calibration
        self.offsets = load_calibration(CALIB_PATH)

        # Desired joint state
        self.q_des = None
        self.state_ready = False

        # Key state (velocity control)
        self.key_vel = np.zeros(NUM_JOINTS)

        # MuJoCo visualization
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Keyboard Joint Teleop (SMOOTH)")
        rospy.loginfo(" Starts from current robot pose")
        rospy.loginfo(" Controls: qawsed rftg y/h")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def state_cb(self, msg):
        """Latch current robot pose ONCE as starting point"""
        if self.state_ready:
            return

        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        self.q_des = np.zeros(NUM_JOINTS)

        for i, j in enumerate(JOINT_NAMES):
            self.q_des[i] = msg.position[name_to_idx[j]] - self.offsets.get(j, 0.0)

        self.state_ready = True
        rospy.loginfo("Teleop initialized from current joint state")

    # ------------------------------------------------
    def publish_command(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = self.q_des.tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------
    def handle_key(self, key):
        """Set velocity directions based on key state"""
        self.key_vel[:] = 0.0

        # Base joints
        if key == ord('q'): self.key_vel[0] = +1
        if key == ord('a'): self.key_vel[0] = -1

        if key == ord('w'): self.key_vel[1] = +1
        if key == ord('s'): self.key_vel[1] = -1

        if key == ord('e'): self.key_vel[2] = +1
        if key == ord('d'): self.key_vel[2] = -1

        if key == ord('r'): self.key_vel[3] = +1
        if key == ord('f'): self.key_vel[3] = -1

        # Wrist pitch (differential)
        if key == ord('t'):
            self.key_vel[4] = +1
            self.key_vel[5] = -1

        if key == ord('g'):
            self.key_vel[4] = -1
            self.key_vel[5] = +1

        # Wrist roll (differential)
        if key == ord('y'):
            self.key_vel[4] = +1
            self.key_vel[5] = +1

        if key == ord('h'):
            self.key_vel[4] = -1
            self.key_vel[5] = -1

    # ------------------------------------------------
    def integrate(self):
        """Integrate joint velocities"""
        self.q_des += self.key_vel * VEL * DT

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(
            0, 0,
            "Keyboard Teleop (smooth) — qawsed rftg y/h, x=exit"
        )
        stdscr.refresh()

        rate = rospy.Rate(RATE_HZ)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                if not self.state_ready:
                    rate.sleep()
                    continue

                key = stdscr.getch()
                if key == ord('x'):
                    break

                if key != -1:
                    self.handle_key(key)
                else:
                    self.key_vel[:] = 0.0

                # Integrate velocity → position
                self.integrate()

                # Publish to robot
                self.publish_command()

                # Visualize in MuJoCo
                self.data.qpos[:NUM_JOINTS] = self.q_des
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()

                rate.sleep()

# ==================================================
def main():
    node = KeyboardJointTeleopROS()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()
