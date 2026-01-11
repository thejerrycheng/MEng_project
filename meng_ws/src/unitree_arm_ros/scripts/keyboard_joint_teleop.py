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
JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6

RATE_HZ = 200
DT = 1.0 / RATE_HZ

# Jog velocities (rad/s)
VEL = np.array([1, 1, 1, 1, 1, 1])

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
        rospy.Subscriber("/joint_states", JointState, self.state_cb, queue_size=1)

        # Calibration offsets (visualization only)
        self.offsets = load_calibration(CALIB_PATH)

        # Joint state
        self.q_des = None
        self.q_vis = None
        self.state_ready = False

        # Key velocity
        self.key_vel = np.zeros(NUM_JOINTS)

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        # Register shutdown hook
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Keyboard Joint Teleop (Safe Hold Enabled)")
        rospy.loginfo(" Ctrl+C or x → robot holds last position")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def state_cb(self, msg):
        if self.state_ready:
            return

        name_to_idx = {n:i for i,n in enumerate(msg.name)}

        self.q_des = np.zeros(NUM_JOINTS)
        self.q_vis = np.zeros(NUM_JOINTS)

        for i,j in enumerate(JOINT_NAMES):
            raw = msg.position[name_to_idx[j]]
            self.q_des[i] = raw
            self.q_vis[i] = raw - self.offsets.get(j,0.0)

        self.state_ready = True
        rospy.loginfo("Teleop initialized from current robot pose")

        # Publish one lock command
        self.publish_command()
        rospy.sleep(0.05)

    # ------------------------------------------------
    def publish_command(self):
        if self.q_des is None:
            return
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = self.q_des.tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------
    def handle_key(self, key):
        self.key_vel[:] = 0.0

        if key == ord('q'): self.key_vel[0] = +1
        if key == ord('a'): self.key_vel[0] = -1
        if key == ord('w'): self.key_vel[1] = +1
        if key == ord('s'): self.key_vel[1] = -1
        if key == ord('e'): self.key_vel[2] = +1
        if key == ord('d'): self.key_vel[2] = -1
        if key == ord('r'): self.key_vel[3] = +1
        if key == ord('f'): self.key_vel[3] = -1
        if key == ord('t'): self.key_vel[4] = +1; self.key_vel[5] = -1
        if key == ord('g'): self.key_vel[4] = -1; self.key_vel[5] = +1
        if key == ord('y'): self.key_vel[4] = +1; self.key_vel[5] = +1
        if key == ord('h'): self.key_vel[4] = -1; self.key_vel[5] = -1

    # ------------------------------------------------
    def integrate(self):
        self.q_des += self.key_vel * VEL * DT
        for i,j in enumerate(JOINT_NAMES):
            self.q_vis[i] = self.q_des[i] - self.offsets.get(j,0.0)

    # ------------------------------------------------
    def on_shutdown(self):
        """Hold last joint command on shutdown"""
        rospy.loginfo("Teleop shutting down → holding last robot position")
        for _ in range(5):
            self.publish_command()
            rospy.sleep(0.02)

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(0,0,"Keyboard Joint Teleop | qawsed rftg y/h | x exit")
        stdscr.refresh()

        rate = rospy.Rate(RATE_HZ)

        with mujoco.viewer.launch_passive(self.model,self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():

                if not self.state_ready:
                    rate.sleep()
                    continue

                key = stdscr.getch()
                if key == ord('x'):
                    rospy.signal_shutdown("User exit")
                    break

                if key != -1:
                    self.handle_key(key)
                else:
                    self.key_vel[:] = 0.0

                self.integrate()
                self.publish_command()

                # MuJoCo visualization
                self.data.qpos[:NUM_JOINTS] = self.q_vis
                mujoco.mj_forward(self.model,self.data)
                viewer.sync()

                rate.sleep()

# ==================================================
def main():
    node = KeyboardJointTeleopROS()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()
