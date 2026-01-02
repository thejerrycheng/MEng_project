import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from collections import deque

# --- Configuration ---
XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "iris_with_axis.xml")
START_Q = np.array([0, 60, 90, 0, 0, 0], dtype=np.float64) # Degrees
STEP_SIZE_POS = 0.002  # m per step
STEP_SIZE_ROT = 0.01   # rad per step

class IRISKinematics:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        # Initial Target Sync
        self.data.qpos[:6] = np.deg2rad(START_Q)
        mujoco.mj_forward(model, data)
        self.target_pos = self.data.body(self.ee_id).xpos.copy()
        
        # Extract initial rotation matrix
        self.target_mat = self.data.body(self.ee_id).xmat.reshape(3, 3).copy()
        self.damping = 1e-4

    def solve_ik(self):
        """Damped Least Squares IK integration."""
        # 1. Current State
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # 2. Compute Errors
        pos_err = self.target_pos - curr_pos
        
        # Orientation error via Axis-Angle
        rot_err_mat = self.target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        error = np.concatenate([pos_err, rot_err_vec])

        # 3. Jacobian Calculation
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        # 4. Damped Least Squares: dq = J^T * inv(J*J^T + lambda^2 * I) * err
        n = J.shape[0]
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(n), error)
        
        # 5. Integrate and Clip to joint limits
        q = self.data.qpos[:6].copy()
        q += dq * 0.1 # Integration gain
        self.data.qpos[:6] = np.clip(q, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

# --- Global Input State ---
active_keys = set()

def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: XML not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISKinematics(model, data)
    
    torque_history = deque(maxlen=10)
    sys.stdout.write("\033[2J\033[H") # Clear terminal

    # Passive viewer handles its own event loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # --- Native Keyboard Callback ---
        def key_callback(keycode):
            try:
                char = chr(keycode).lower()
                active_keys.add(char)
            except: pass
        
        viewer.key_callback = key_callback

        while viewer.is_running():
            step_start = time.time()

            # 1. Process Key Commands (Multi-key supported)
            # Position
            if 'q' in active_keys: kin.target_pos[0] += STEP_SIZE_POS
            if 'a' in active_keys: kin.target_pos[0] -= STEP_SIZE_POS
            if 'w' in active_keys: kin.target_pos[1] += STEP_SIZE_POS
            if 's' in active_keys: kin.target_pos[1] -= STEP_SIZE_POS
            if 'e' in active_keys: kin.target_pos[2] += STEP_SIZE_POS
            if 'd' in active_keys: kin.target_pos[2] -= STEP_SIZE_POS
            
            # Simulated key release for passive viewer callback logic
            active_keys.clear() 

            # 2. IK and Physics
            kin.solve_ik()
            mujoco.mj_step(model, data)
            viewer.sync()

            # 3. Dashboard
            if int(data.time * 100) % 10 == 0:
                curr_q_deg = np.rad2deg(data.qpos[:6])
                sim_pos = data.body(kin.ee_id).xpos
                
                sys.stdout.write("\033[H")
                out = [
                    "=================== IRIS IK TELEOP DASHBOARD ===================",
                    f" Time: {data.time:6.2f}s | Focus the Sim Window to move",
                    "-" * 64,
                    f"{'COORD':<8} | {'X (m)':^12} | {'Y (m)':^12} | {'Z (m)':^12}",
                    "-" * 64,
                    f"{'Target':<8} | {kin.target_pos[0]:10.3f}   | {kin.target_pos[1]:10.3f}   | {kin.target_pos[2]:10.3f}",
                    f"{'Actual':<8} | {sim_pos[0]:10.3f}   | {sim_pos[1]:10.3f}   | {sim_pos[2]:10.3f}",
                    "-" * 64,
                    f" L2 Tracking Error: {np.linalg.norm(sim_pos - kin.target_pos):.6f}",
                    " Controls: [QAWSED] for Cartesian XYZ | [R] to Reset Pose",
                    "================================================================"
                ]
                sys.stdout.write("\n".join(out) + "\n")
                sys.stdout.flush()

            # Real-time sync
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()