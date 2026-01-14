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
# Topics (All External Calibration)
TOPIC_SUB = "/joint_states_calibrated"
TOPIC_PUB = "/joint_commands_calibrated"

JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6

RATE_HZ = 200
DT = 1.0 / RATE_HZ

MOVE_SPEED = 0.5    # m/s
ROT_SPEED  = 0.50   # rad/s

JOINT_LIMITS_DEG = [(-170,170),(-170,170),(-150,150),(-180,180),(-100,100),(-360,360)]
EE_BODY_NAME = "ee_mount"

DAMPING = 1e-4
IK_STEP_SCALE = 0.5

# ==================================================
def clamp_qpos(qpos):
    limit_min = np.deg2rad([l[0] for l in JOINT_LIMITS_DEG])
    limit_max = np.deg2rad([l[1] for l in JOINT_LIMITS_DEG])
    return np.clip(qpos, limit_min, limit_max)

def mat2euler(mat):
    """Return Roll-Pitch-Yaw (XYZ convention)"""
    sy = np.sqrt(mat[0,0]**2 + mat[1,0]**2)
    if sy > 1e-6:
        roll  = np.arctan2(mat[2,1], mat[2,2])
        pitch = np.arctan2(-mat[2,0], sy)
        yaw   = np.arctan2(mat[1,0], mat[0,0])
        return np.array([roll, pitch, yaw])
    else:
        return np.array([0.0, np.arctan2(-mat[2,0], sy), 0.0])

# ==================================================
class MujocoIKTeleopROS:
    def __init__(self):
        rospy.init_node("mujoco_ik_teleop_ros")

        # 1. Setup Topics
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)

        # 2. Setup MuJoCo
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"XML not found: {XML_PATH}")
            
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        if self.ee_id < 0:
            raise RuntimeError(f"Body '{EE_BODY_NAME}' not found in XML")

        # 3. Initialization Flags & Buffers
        self.state_initialized = False
        
        # Current Target (will be set in state_cb)
        self.target_pos = np.zeros(3)
        self.target_rpy = np.zeros(3)

        # Velocity Inputs (from keyboard)
        self.cart_vel = np.zeros(3)
        self.rot_vel  = np.zeros(3)

        rospy.loginfo("==============================================")
        rospy.loginfo(" MuJoCo IK Teleop (Pure External Calib)")
        rospy.loginfo(f" Listening: {TOPIC_SUB}")
        rospy.loginfo(f" Publishing: {TOPIC_PUB}")
        rospy.loginfo(" Status: WAITING FOR ROBOT STATE...")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def state_cb(self, msg):
        """
        Only runs ONCE at startup.
        Sets the MuJoCo robot to the exact same pose as the real robot,
        then sets the IK Target to that pose so it doesn't move.
        """
        if self.state_initialized:
            return

        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        start_q = np.zeros(NUM_JOINTS)
        
        # Extract joint values safely
        for i, jname in enumerate(JOINT_NAMES):
            if jname not in name_to_idx:
                # If we don't have full state yet, wait.
                return 
            start_q[i] = msg.position[name_to_idx[jname]]

        # 1. Update Internal Sim to match Reality
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(start_q)
        mujoco.mj_forward(self.model, self.data)

        # 2. Set IK Target to CURRENT End-Effector Pose
        # This ensures error is 0.0 at t=0
        self.target_pos = self.data.body(self.ee_id).xpos.copy()
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3,3)
        self.target_rpy = mat2euler(curr_mat)

        # 3. Publish the initial hold command immediately
        # (This prevents the robot from going limp while we wait for the loop)
        self.publish_command()

        self.state_initialized = True
        rospy.loginfo(">>> Robot Sync Complete. Teleop Active.")
        rospy.loginfo(f">>> Initial EE Pos: {np.round(self.target_pos, 3)}")

    # ------------------------------------------------
    def publish_command(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        # Direct pass-through of the solved IK angles
        msg.position = self.data.qpos[:NUM_JOINTS].tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------
    def set_key_state(self, key):
        self.cart_vel[:] = 0.0
        self.rot_vel[:]  = 0.0

        if key == ord('q'): self.cart_vel[0] = +1
        if key == ord('a'): self.cart_vel[0] = -1
        if key == ord('w'): self.cart_vel[1] = +1
        if key == ord('s'): self.cart_vel[1] = -1
        if key == ord('e'): self.cart_vel[2] = +1
        if key == ord('d'): self.cart_vel[2] = -1

        if key == ord('r'): self.rot_vel[0] = +1
        if key == ord('f'): self.rot_vel[0] = -1
        if key == ord('t'): self.rot_vel[1] = +1
        if key == ord('g'): self.rot_vel[1] = -1
        if key == ord('y'): self.rot_vel[2] = +1
        if key == ord('h'): self.rot_vel[2] = -1

    # ------------------------------------------------
    def integrate_targets(self):
        self.target_pos += self.cart_vel * MOVE_SPEED * DT
        self.target_rpy += self.rot_vel  * ROT_SPEED * DT

    # ------------------------------------------------
    def ik_step(self):
        # 1. Target Orientation Error
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, self.target_rpy, 'xyz')
        
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        target_mat = mat.reshape(3,3)

        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3,3)

        pos_err = self.target_pos - curr_pos

        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        # 2. Jacobian
        jacp = np.zeros((3,self.model.nv))
        jacr = np.zeros((3,self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)

        J = np.vstack([jacp,jacr])[:,:NUM_JOINTS]
        err = np.concatenate([pos_err, rot_err])

        # 3. Solve (Damped Least Squares)
        dq = J.T @ np.linalg.solve(J@J.T + DAMPING*np.eye(6), err)

        # 4. Integrate
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(
            self.data.qpos[:NUM_JOINTS] + dq * IK_STEP_SCALE
        )

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(0,0,"MuJoCo IK Teleop | qawsed XYZ | rftgyh RPY | x exit")
        stdscr.addstr(2,0,"Status: Waiting for robot...")
        stdscr.refresh()

        rate = rospy.Rate(RATE_HZ)

        # Wait loop
        while not rospy.is_shutdown() and not self.state_initialized:
            rate.sleep()

        stdscr.addstr(2,0,"Status: CONNECTED. Control Active.    ")
        stdscr.refresh()

        # Control loop
        with mujoco.viewer.launch_passive(self.model,self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                key = stdscr.getch()
                if key == ord('x'):
                    break

                if key != -1:
                    self.set_key_state(key)
                else:
                    self.cart_vel[:] = 0.0
                    self.rot_vel[:]  = 0.0

                self.integrate_targets()
                self.ik_step()
                mujoco.mj_step(self.model,self.data)

                self.publish_command()

                viewer.sync()
                rate.sleep()

# ==================================================
def main():
    node = MujocoIKTeleopROS()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()