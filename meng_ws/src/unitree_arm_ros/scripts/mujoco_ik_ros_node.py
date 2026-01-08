#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import rospy
import mujoco
import mujoco.viewer
import curses

from sensor_msgs.msg import JointState

# ==================================================
# Paths (ABSOLUTE, mesh-safe)
# ==================================================
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"MuJoCo XML not found: {XML_PATH}")

# ==================================================
# Robot config
# ==================================================
JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

START_Q_DEG = np.array([0, -45, -90, 0, 0, 0], dtype=np.float64)

JOINT_LIMITS_DEG = [
    (-170, 170),
    (-170, 170),
    (-150, 150),
    (-180, 180),
    (-100, 100),
    (-360, 360),
]

MOVE_SPEED = 0.50   # m/s
ROT_SPEED = 1.50    # rad/s
IK_DAMPING = 1e-4

# ==================================================
def mat2euler(mat):
    sy = np.sqrt(mat[0, 0]**2 + mat[1, 0]**2)
    if sy > 1e-6:
        return np.array([
            np.arctan2(mat[2, 1], mat[2, 2]),
            np.arctan2(-mat[2, 0], sy),
            np.arctan2(mat[1, 0], mat[0, 0])
        ])
    return np.array([
        np.arctan2(-mat[1, 2], mat[1, 1]),
        np.arctan2(-mat[2, 0], sy),
        0.0
    ])

# ==================================================
class MujocoIKKeyboardRosNode:
    def __init__(self):
        rospy.init_node("mujoco_ik_keyboard_node")

        self.joint_pub = rospy.Publisher(
            "/arm/command",
            JointState,
            queue_size=1
        )

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "ee_mount"
        )

        self.data.qpos[:6] = np.deg2rad(START_Q_DEG)
        mujoco.mj_forward(self.model, self.data)

        self.target_pos = self.data.body(self.ee_id).xpos.copy()
        self.target_euler = mat2euler(
            self.data.body(self.ee_id).xmat.reshape(3, 3)
        )

        rospy.loginfo("MuJoCo IK Keyboard ROS node ready")

    # ------------------------------------------------
    def process_key(self, key, dt):
        # Translation
        if key == ord('q'): self.target_pos[0] += MOVE_SPEED * dt
        if key == ord('a'): self.target_pos[0] -= MOVE_SPEED * dt
        if key == ord('w'): self.target_pos[1] += MOVE_SPEED * dt
        if key == ord('s'): self.target_pos[1] -= MOVE_SPEED * dt
        if key == ord('e'): self.target_pos[2] += MOVE_SPEED * dt
        if key == ord('d'): self.target_pos[2] -= MOVE_SPEED * dt

        # Rotation
        if key == ord('r'): self.target_euler[0] += ROT_SPEED * dt
        if key == ord('f'): self.target_euler[0] -= ROT_SPEED * dt
        if key == ord('t'): self.target_euler[1] += ROT_SPEED * dt
        if key == ord('g'): self.target_euler[1] -= ROT_SPEED * dt
        if key == ord('y'): self.target_euler[2] += ROT_SPEED * dt
        if key == ord('h'): self.target_euler[2] -= ROT_SPEED * dt

    # ------------------------------------------------
    def solve_ik_step(self):
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
        rot_err = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)

        J = np.vstack([jacp, jacr])[:, :6]
        err = np.concatenate([pos_err, rot_err])

        dq = J.T @ np.linalg.solve(
            J @ J.T + IK_DAMPING * np.eye(6),
            err
        )

        self.data.qpos[:6] += dq

        qmin = np.deg2rad([l[0] for l in JOINT_LIMITS_DEG])
        qmax = np.deg2rad([l[1] for l in JOINT_LIMITS_DEG])
        self.data.qpos[:6] = np.clip(self.data.qpos[:6], qmin, qmax)

        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------
    def publish_joint_command(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = self.data.qpos[:6].copy()
        self.joint_pub.publish(msg)

    # ------------------------------------------------
    def run(self, stdscr):
        stdscr.nodelay(True)
        stdscr.clear()
        stdscr.addstr(0, 0, "MuJoCo IK Teleop (qawsed rftgyh, x=exit)")
        stdscr.refresh()

        rate = rospy.Rate(1.0 / self.model.opt.timestep)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                key = stdscr.getch()
                if key == ord('x'):
                    break

                self.process_key(key, self.model.opt.timestep)
                self.solve_ik_step()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                self.publish_joint_command()
                rate.sleep()

# ==================================================
def main():
    node = MujocoIKKeyboardRosNode()
    curses.wrapper(node.run)

if __name__ == "__main__":
    main()
