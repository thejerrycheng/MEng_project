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

if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"MuJoCo XML not found: {XML_PATH}")
if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(f"Calibration file not found: {CALIB_PATH}")

# ==================================================
# Config
# ==================================================
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
NUM_JOINTS = 6

RATE_HZ = 200
DT = 1.0 / RATE_HZ

# Cartesian teleop speeds (match your script)
MOVE_SPEED = 0.10  # m/s
ROT_SPEED  = 0.50  # rad/s

# Joint limits from your script (degrees)
JOINT_LIMITS_DEG = [(-170, 170), (-170, 170), (-150, 150), (-180, 180), (-100, 100), (-360, 360)]

# EE body name in MuJoCo XML
EE_BODY_NAME = "ee_mount"

# IK settings
DAMPING = 1e-4
IK_STEP_SCALE = 0.5  # same as your dq * 0.5

# ==================================================
def load_calibration(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["joint_offsets"]

def mat2euler(mat):
    """XYZ euler from 3x3 rot matrix (same logic as your script)."""
    sy = np.sqrt(mat[0,0]**2 + mat[1,0]**2)
    if sy > 1e-6:
        return np.array([
            np.arctan2(mat[2,1], mat[2,2]),
            np.arctan2(-mat[2,0], sy),
            np.arctan2(mat[1,0], mat[0,0]),
        ])
    return np.array([
        np.arctan2(-mat[1,2], mat[1,1]),
        np.arctan2(-mat[2,0], sy),
        0.0
    ])

def clamp_qpos(qpos):
    limit_min = np.deg2rad([l[0] for l in JOINT_LIMITS_DEG])
    limit_max = np.deg2rad([l[1] for l in JOINT_LIMITS_DEG])
    return np.clip(qpos, limit_min, limit_max)

# ==================================================
class MujocoIKTeleopROS:
    """
    IK teleop node:
    - Subscribes /joint_states (real robot) to set starting pose once
    - Runs MuJoCo IK to follow a keyboard-driven Cartesian target
    - Publishes /arm/command (JointState) with desired joint targets

    Keys:
      Position: q/a (x+/-), w/s (y+/-), e/d (z+/-)
      Rotation: r/f (roll +/-), t/g (pitch +/-), y/h (yaw +/-)
      Exit: x
    """
    def __init__(self):
        rospy.init_node("mujoco_ik_teleop_ros")

        # ROS pub/sub
        self.cmd_pub = rospy.Publisher("/arm/command", JointState, queue_size=1)
        self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_cb, queue_size=1)

        # Calibration offsets (what you used in joint teleop)
        self.offsets = load_calibration(CALIB_PATH)

        # Real robot starting pose (latched once)
        self.q_start = None
        self.state_ready = False

        # MuJoCo sim (IK solved in sim)
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        if self.ee_id < 0:
            raise RuntimeError(f"MuJoCo body '{EE_BODY_NAME}' not found in XML. Check iris.xml")

        # Targets (cartesian + euler)
        self.target_pos = np.zeros(3)
        self.target_euler = np.zeros(3)

        # Key velocity state (integrate like smooth joint teleop)
        self.cart_vel = np.zeros(3)   # x,y,z direction
        self.rot_vel  = np.zeros(3)   # roll,pitch,yaw direction

        # IK damping
        self.damping = DAMPING

        rospy.loginfo("==============================================")
        rospy.loginfo(" MuJoCo IK Teleop ROS Node (SMOOTH)")
        rospy.loginfo(" Publishes: /arm/command  |  Subscribes: /joint_states")
        rospy.loginfo(" Keys: qawsed (pos) | rftgyh (rot) | x=exit")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def state_cb(self, msg):
        """Latch current robot pose ONCE as starting pose for MuJoCo IK."""
        if self.state_ready:
            return

        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q = np.zeros(NUM_JOINTS)

        for i, j in enumerate(JOINT_NAMES):
            if j not in name_to_idx:
                rospy.logwarn_throttle(1.0, f"Waiting for joint '{j}' in /joint_states...")
                return
            q[i] = msg.position[name_to_idx[j]] - self.offsets.get(j, 0.0)

        self.q_start = q.copy()
        self.state_ready = True

        # Initialize MuJoCo qpos from real robot
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(self.q_start)
        mujoco.mj_forward(self.model, self.data)

        # Sync targets to current EE pose
        self.target_pos = self.data.body(self.ee_id).xpos.copy()
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        self.target_euler = mat2euler(curr_mat)

        rospy.loginfo("IK teleop initialized from current robot pose.")

    # ------------------------------------------------
    def publish_command(self, q_des):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_des.tolist()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------
    def set_key_state(self, key):
        """Set cart_vel / rot_vel directions based on one key event (like your smooth teleop)."""
        self.cart_vel[:] = 0.0
        self.rot_vel[:] = 0.0

        # Position
        if key == ord('q'): self.cart_vel[0] = +1
        if key == ord('a'): self.cart_vel[0] = -1
        if key == ord('w'): self.cart_vel[1] = +1
        if key == ord('s'): self.cart_vel[1] = -1
        if key == ord('e'): self.cart_vel[2] = +1
        if key == ord('d'): self.cart_vel[2] = -1

        # Rotation (roll, pitch, yaw)
        if key == ord('r'): self.rot_vel[0] = +1
        if key == ord('f'): self.rot_vel[0] = -1
        if key == ord('t'): self.rot_vel[1] = +1
        if key == ord('g'): self.rot_vel[1] = -1
        if key == ord('y'): self.rot_vel[2] = +1
        if key == ord('h'): self.rot_vel[2] = -1

    # ------------------------------------------------
    def integrate_targets(self):
        self.target_pos += self.cart_vel * MOVE_SPEED * DT
        self.target_euler += self.rot_vel * ROT_SPEED * DT

    # ------------------------------------------------
    def ik_step(self):
        """One damped least-squares IK step (same math as your script)."""
        target_quat = np.zeros(4)
        mujoco.mju_euler2Quat(target_quat, self.target_euler, 'xyz')
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)
        target_mat = target_mat.reshape(3, 3)

        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)

        pos_err = self.target_pos - curr_pos

        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)

        J = np.vstack([jacp, jacr])[:, :NUM_JOINTS]
        error = np.concatenate([pos_err, rot_err_vec])

        # dq = J^T (J J^T + λI)^-1 error
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), error)

        self.data.qpos[:NUM_JOINTS] = clamp_qpos(self.data.qpos[:NUM_JOINTS] + dq * IK_STEP_SCALE)

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(
            0, 0,
            "MuJoCo IK Teleop → ROS (/arm/command). qawsed pos | rftgyh rot | x exit"
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
                    self.set_key_state(key)
                else:
                    self.cart_vel[:] = 0.0
                    self.rot_vel[:] = 0.0

                # Integrate desired cartesian target
                self.integrate_targets()

                # Solve IK and update MuJoCo state
                self.ik_step()
                mujoco.mj_step(self.model, self.data)

                # Publish joint targets
                q_des = self.data.qpos[:NUM_JOINTS].copy()
                self.publish_command(q_des)

                viewer.sync()
                rate.sleep()

# ==================================================
def main():
    node = MujocoIKTeleopROS()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()
