import mujoco
import numpy as np
import os
from scipy.optimize import least_squares

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris_with_axis.xml")

def get_dh_matrix(q, a, alpha, d, theta_off):
    """Standard DH Transformation Matrix."""
    theta = q + theta_off
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def forward_kinematics(q_vec, params):
    """Calculates EE position based on a flat array of DH parameters."""
    T = np.eye(4)
    # params is organized as [a1, al1, d1, th1, a2, al2, d2, th2, ...]
    for i in range(6):
        idx = i * 4
        T = T @ get_dh_matrix(q_vec[i], params[idx], params[idx+1], params[idx+2], params[idx+3])
    return T[:3, 3]

def residuals(params, q_samples, pos_samples):
    """Difference between analytical FK and simulation samples."""
    res = []
    for q, pos_true in zip(q_samples, pos_samples):
        pos_pred = forward_kinematics(q, params)
        res.extend(pos_pred - pos_true)
    return np.array(res)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("Gathering 100 random samples from simulation...")
    q_samples = []
    pos_samples = []
    
    for _ in range(100):
        # Sample random joint angles within limits
        q = np.random.uniform(-1.5, 1.5, 6)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        
        q_samples.append(q.copy())
        pos_samples.append(data.body('wrist2').xpos.copy())

    # Initial Guess [a, alpha, d, theta_off] * 6 joints
    # Starting with zeros but using known heights for 'd' helps convergence
    initial_guess = np.zeros(24)
    initial_guess[2] = 0.25  # d1 guess
    initial_guess[6] = 1.57  # alpha2 guess (90 deg)

    print("Running Least Squares Optimization (Levenberg-Marquardt)...")
    result = least_squares(residuals, initial_guess, args=(q_samples, pos_samples), method='lm')

    final_params = result.x.reshape(6, 4)
    
    print("\n" + "="*75)
    print(f"{'IDENTIFIED ROBUST DH TABLE':^75}")
    print("="*75)
    print(f"{'Joint':<8} | {'a (m)':^12} | {'alpha (deg)':^12} | {'d (m)':^12} | {'th_off (deg)':^12}")
    print("-" * 75)
    
    for i in range(6):
        p = final_params[i]
        print(f" J{i+1:<7} | {p[0]:12.4f} | {np.rad2deg(p[1]):12.2f} | {p[2]:12.4f} | {np.rad2deg(p[3]):12.2f}")
    
    print("="*75)
    print(f"Optimization Success: {result.success}")
    print(f"Final Cost (Residual Sum): {result.cost:.6e}")

if __name__ == "__main__":
    main()