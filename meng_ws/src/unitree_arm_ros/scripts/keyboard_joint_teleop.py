#!/usr/bin/env python3
import os
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

# ==================================================
# Config
# ==================================================
# Topics (Working in World Frame)
TOPIC_SUB = "/joint_states_calibrated"
TOPIC_PUB = "/joint_commands_calibrated"

JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6

RATE_HZ = 200
DT = 1.0 / RATE_HZ

# Jog velocities (rad/s)
VEL = np.array([2, 2, 2, 2.0, 1.0, 1.0])

# ==================================================
class KeyboardJointTeleopROS:
    def __init__(self):
        rospy.init_node("keyboard_joint_teleop_ros")

        # ROS I/O
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)

        # Joint state
        self.q_des = None
        self.state_ready = False

        # Key velocity
        self.key_vel = np.zeros(NUM_JOINTS)

        # MuJoCo
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"XML not found: {XML_PATH}")
            
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        # Register shutdown hook
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Forward Kinematics Teleop (Calibrated)")
        rospy.loginfo(" Controls: Pitch (t/g) | Roll (y/h)")
        rospy.loginfo(f" Listening: {TOPIC_SUB}")
        rospy.loginfo(f" Publishing: {TOPIC_PUB}")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def state_cb(self, msg):
        """Initialize from current World Frame pose"""
        if self.state_ready:
            return

        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        self.q_des = np.zeros(NUM_JOINTS)
        
        # Check if all joints are present
        for i, jname in enumerate(JOINT_NAMES):
            if jname not in name_to_idx:
                return # Wait for full state
            self.q_des[i] = msg.position[name_to_idx[jname]]

        self.state_ready = True
        rospy.loginfo(f"Initialized at: {np.round(self.q_des, 3)}")

        # Publish one lock command to hold position
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

        # Base Joints
        if key == ord('q'): self.key_vel[0] = +1
        if key == ord('a'): self.key_vel[0] = -1
        if key == ord('w'): self.key_vel[1] = +1
        if key == ord('s'): self.key_vel[1] = -1
        if key == ord('e'): self.key_vel[2] = +1
        if key == ord('d'): self.key_vel[2] = -1
        if key == ord('r'): self.key_vel[3] = +1
        if key == ord('f'): self.key_vel[3] = -1
        
        # Wrist Joints (Calibrated Pitch/Roll)
        # We control Pitch/Roll directly, Calibration Node handles the differential
        if key == ord('t'): self.key_vel[4] = +1 # Pitch Up
        if key == ord('g'): self.key_vel[4] = -1 # Pitch Down
        if key == ord('y'): self.key_vel[5] = +1 # Roll Left
        if key == ord('h'): self.key_vel[5] = -1 # Roll Right

    # ------------------------------------------------
    def integrate(self):
        # Update desired position
        self.q_des += self.key_vel * VEL * DT

    # ------------------------------------------------
    def on_shutdown(self):
        """Hold last joint command on shutdown"""
        rospy.loginfo("Teleop shutting down → holding last position")
        for _ in range(5):
            self.publish_command()
            rospy.sleep(0.02)

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(0,0,"FK Teleop | qawsed (Base) | rf (J4) | tg (Pitch) | yh (Roll) | x exit")
        stdscr.addstr(2,0,"Status: Waiting for robot state...")
        stdscr.refresh()

        rate = rospy.Rate(RATE_HZ)

        # Wait loop
        while not rospy.is_shutdown() and not self.state_ready:
            rate.sleep()

        stdscr.addstr(2,0,"Status: ACTIVE. Control Enabled.        ")
        stdscr.refresh()

        with mujoco.viewer.launch_passive(self.model,self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                
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
                # Since q_des is already in World Frame, we map it directly
                self.data.qpos[:NUM_JOINTS] = self.q_des
                mujoco.mj_forward(self.model,self.data)
                viewer.sync()

                rate.sleep()

# ==================================================
def main():
    node = KeyboardJointTeleopROS()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()