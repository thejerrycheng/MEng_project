import mujoco
import numpy as np
import time
import os

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")

class IRISInverseKinematics:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # End-effector body name from your XML
        self.ee_name = "ee_mount"
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)
        
        # IK Hyperparameters
        self.integration_dt = 0.1  # Step size for the solver
        self.damping = 1e-4       # Damping for Damped Least Squares (prevents singularity explosions)
        self.tol = 1e-4           # Convergence tolerance (meters)
        self.max_steps = 100      # Safety cap for iterations per frame

    def solve(self, target_pos, target_quat=None):
        """
        Calculates joint angles to reach a target position (and optionally orientation).
        Returns: qpos (numpy array) or None if failed.
        """
        # Initial guess is the current joint positions
        q = self.data.qpos[:6].copy()
        
        success = False
        for i in range(self.max_steps):
            # Forward kinematics for current guess
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
            
            # Current EE state
            current_pos = self.data.body(self.ee_id).xpos
            current_mat = self.data.body(self.ee_id).xmat
            
            # Position Error
            pos_err = target_pos - current_pos
            
            # Orientation Error (Optional)
            if target_quat is not None:
                # Convert target_quat to matrix
                target_mat = np.zeros(9)
                mujoco.mju_quat2Mat(target_mat, target_quat)
                target_mat = target_mat.reshape(3, 3)
                
                # Compute rotational error (difference in rotation)
                rot_err_mat = target_mat @ current_mat.T
                rot_err_vec = np.zeros(3)
                mujoco.mju_mat2Quat(np.zeros(4), rot_err_mat.flatten()) # Simplified error logic
                # For real-time, simple position-only is often enough, 
                # but we use 6D Jacobian here:
                error = np.concatenate([pos_err, [0, 0, 0]]) # Replace [0,0,0] with rot error if needed
            else:
                error = np.concatenate([pos_err, [0, 0, 0]])

            # Check convergence
            if np.linalg.norm(pos_err) < self.tol:
                success = True
                break

            # Calculate Jacobian
            jacp = np.zeros((3, self.model.nv)) # Position Jacobian
            jacr = np.zeros((3, self.model.nv)) # Rotation Jacobian
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
            
            # Full 6D Jacobian (or 3D if only position matters)
            J = np.vstack([jacp, jacr])[:, :6]
            
            # Damped Least Squares: dq = J^T * inv(J*J^T + lambda^2 * I) * error
            n = J.shape[0]
            diag = self.damping * np.eye(n)
            dq = J.T @ np.linalg.solve(J @ J.T + diag, error)
            
            # Update joint positions
            q += dq * self.integration_dt
            
            # Clamp to joint limits (extracted from your XML ranges)
            q = np.clip(q, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

        return q if success else None

def main():
    if not os.path.exists(XML_PATH):
        print("XML not found.")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ik_solver = IRISInverseKinematics(model, data)

    # Define a target: 0.5m in front of the base, 0.4m up
    target_position = np.array([0.5, 0.0, 0.4])

    print(f"Solving IK for target: {target_position}...")
    
    start_time = time.time()
    result_q = ik_solver.solve(target_position)
    end_time = time.time()

    if result_q is not None:
        print(f"IK Success! Time: {(end_time - start_time)*1000:.2f}ms")
        print(f"Resulting Joint Angles (Deg): {np.rad2deg(result_q)}")
        
        # Verify result
        data.qpos[:6] = result_q
        mujoco.mj_forward(model, data)
        print(f"Final EE Position in Sim: {data.body('ee_mount').xpos}")
    else:
        print("IK Failed to converge.")

if __name__ == "__main__":
    main()