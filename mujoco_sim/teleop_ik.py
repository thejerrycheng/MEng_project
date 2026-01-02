import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from pynput import keyboard
from collections import deque

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
START_Q = np.array([0, -45, -90, 0, 0, 0], dtype=np.float64) 

# Control Sensitivities
MOVE_SPEED = 0.1  # m/s
ROT_SPEED = 0.5   # rad/s

# --- Global Control State ---
joint_limits = [(-170, 170), (-170, 170), (-150, 150), (-180, 180), (-100, 100), (-360, 360)]
target_pos_global = np.zeros(3)
target_euler_global = np.zeros(3) # [Roll, Pitch, Yaw]
active_keys = set()

def on_press(key):
    try: active_keys.add(key.char.lower())
    except AttributeError: pass

def on_release(key):
    try:
        k = key.char.lower()
        if k in active_keys: active_keys.remove(k)
    except (AttributeError, KeyError): pass

def mat2euler(mat):
    """Helper to extract XYZ Euler angles from a 3x3 rotation matrix."""
    sy = np.sqrt(mat[0,0]**2 + mat[1,0]**2)
    if sy > 1e-6:
        return np.array([np.arctan2(mat[2,1], mat[2,2]), np.arctan2(-mat[2,0], sy), np.arctan2(mat[1,0], mat[0,0])])
    return np.array([np.arctan2(-mat[1,2], mat[1,1]), np.arctan2(-mat[2,0], sy), 0])

class IKController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        # 1. Initialize Robot
        self.data.qpos[:6] = np.deg2rad(START_Q)
        mujoco.mj_forward(model, data)
        
        # 2. Sync Global Targets to current EE pose
        global target_pos_global, target_euler_global
        target_pos_global = self.data.body(self.ee_id).xpos.copy()
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        target_euler_global = mat2euler(curr_mat)
        
        self.damping = 1e-4

    def solve_step(self):
        target_quat = np.zeros(4)
        mujoco.mju_euler2Quat(target_quat, target_euler_global, 'xyz')
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)
        target_mat = target_mat.reshape(3, 3)

        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        pos_err = target_pos_global - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        error = np.concatenate([pos_err, rot_err_vec])
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), error)
        
        self.data.qpos[:6] += dq * 0.5 
        limit_min = np.deg2rad([l[0] for l in joint_limits])
        limit_max = np.deg2rad([l[1] for l in joint_limits])
        self.data.qpos[:6] = np.clip(self.data.qpos[:6], limit_min, limit_max)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = IKController(model, data)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Clear terminal
    sys.stdout.write("\033[2J\033[H")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            # Process Inputs
            if 'q' in active_keys: target_pos_global[0] += MOVE_SPEED * dt
            if 'a' in active_keys: target_pos_global[0] -= MOVE_SPEED * dt
            if 'w' in active_keys: target_pos_global[1] += MOVE_SPEED * dt
            if 's' in active_keys: target_pos_global[1] -= MOVE_SPEED * dt
            if 'e' in active_keys: target_pos_global[2] += MOVE_SPEED * dt
            if 'd' in active_keys: target_pos_global[2] -= MOVE_SPEED * dt
            
            if 'r' in active_keys: target_euler_global[0] += ROT_SPEED * dt
            if 'f' in active_keys: target_euler_global[0] -= ROT_SPEED * dt
            if 't' in active_keys: target_euler_global[1] += ROT_SPEED * dt
            if 'g' in active_keys: target_euler_global[1] -= ROT_SPEED * dt
            if 'y' in active_keys: target_euler_global[2] += ROT_SPEED * dt
            if 'h' in active_keys: target_euler_global[2] -= ROT_SPEED * dt

            controller.solve_step()
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- Multi-Row Detailed Dashboard ---
            if int(data.time * 100) % 10 == 0:
                sys.stdout.write("\033[H")
                actual_pos = data.body(controller.ee_id).xpos
                actual_mat = data.body(controller.ee_id).xmat.reshape(3, 3)
                actual_euler = mat2euler(actual_mat)
                
                out = [
                    "====================== IRIS 6-DOF TELEOP LOG ======================",
                    f" Sim Time: {data.time:6.2f}s | Active Keys: {list(active_keys)}",
                    "-" * 67,
                    f" {'METRIC':<10} | {'X / ROLL':^16} | {'Y / PITCH':^16} | {'Z / YAW':^16}",
                    "-" * 67,
                    " [ CARTESIAN POSITION (meters) ]",
                    f" {'Desired':<10} | {target_pos_global[0]:14.3f} | {target_pos_global[1]:14.3f} | {target_pos_global[2]:14.3f}",
                    f" {'Actual':<10} | {actual_pos[0]:14.3f} | {actual_pos[1]:14.3f} | {actual_pos[2]:14.3f}",
                    "-" * 67,
                    " [ ORIENTATION (radians) ]",
                    f" {'Desired':<10} | {target_euler_global[0]:14.3f} | {target_euler_global[1]:14.3f} | {target_euler_global[2]:14.3f}",
                    f" {'Actual':<10} | {actual_euler[0]:14.3f} | {actual_euler[1]:14.3f} | {actual_euler[2]:14.3f}",
                    "-" * 67,
                    f" L2 Dist Error: {np.linalg.norm(actual_pos - target_pos_global):.6f} m",
                    " Controls: QAWSED (Pos) | RFTGYH (Rot) | Focus Sim Window",
                    "==================================================================="
                ]
                sys.stdout.write("\n".join(out) + "\n")
                sys.stdout.flush()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()