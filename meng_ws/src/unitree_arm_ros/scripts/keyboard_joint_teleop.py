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
    "joint_1","joint_2","joint_3",
    "joint_4","joint_5","joint_6"
]

NUM_JOINTS = 6
RATE_HZ = 200
DT = 1.0 / RATE_HZ

# Jog velocities (rad/s)
VEL = np.array([0.6, 0.6, 0.6, 0.8, 1.0, 1.0])

# ==================================================
def load_calibration(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["joint_offsets"]

# ==================================================
class KeyboardJointTeleopROS:
    def __init__(self):
        rospy.init_node("keyboard_joint_teleop_ros")

        # ROS I/O
        self.cmd_pub   = rospy.Publisher("/arm/command", JointState, queue_size=1)
        self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_cb, queue_size=1)

        # Calibration offsets (for MuJoCo visualization only)
        self.offsets = load_calibration(CALIB_PATH)

        # Joint states
        self.q_des = None        # joint command sent to robot
        self.q_vis = None        # visualization joint pose
        self.state_ready = False

        # Key velocity state
        self.key_vel = np.zeros(NUM_JOINTS)

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Keyboard Joint Teleop (Robot + MuJoCo)")
        rospy.loginfo(" Starts from current robot pose")
        rospy.loginfo(" Controls: qawsed rftg y/h")
        rospy.loginfo(" x : exit")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def state_cb(self, msg):
        """
        Latch current robot joint state ONCE as starting point.
        """
        if self.state_ready:
            return

        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        self.q_des = np.zeros(NUM_JOINTS)
        self.q_vis = np.zeros(NUM_JOINTS)

        for i, j in enumerate(JOINT_NAMES):
            raw = msg.position[name_to_idx[j]]
            self.q_des[i] = raw                    # true robot joint command
            self.q_vis[i] = raw - self.offsets.get(j, 0.0)  # visualization pose

        self.state_ready = True
        rospy.loginfo("Teleop initialized from current robot pose")

        # ---- publish one initial lock command ----
        init_msg = JointState()
        init_msg.name = JOINT_NAMES
        init_msg.position = self.q_des.tolist()
        self.cmd_pub.publish(init_msg)
        rospy.sleep(0.1)

    # ------------------------------------------------
    def publish_command(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = self.q_des.tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------
    def handle_key(self, key):
        """Set velocity direction based on key"""
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
        """Integrate velocity to target position"""
        self.q_des += self.key_vel * VEL * DT

        # Update visualization pose
        for i, j in enumerate(JOINT_NAMES):
            self.q_vis[i] = self.q_des[i] - self.offsets.get(j, 0.0)

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(
            0, 0,
            "Keyboard Teleop — qawsed rftg y/h | x = exit"
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

                # Integrate velocity → joint command
                self.integrate()

                # Publish to real robot
                self.publish_command()

                # Update MuJoCo visualization
                self.data.qpos[:NUM_JOINTS] = self.q_vis
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()

                rate.sleep()

# ==================================================
def main():
    node = KeyboardJointTeleopROS()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()
