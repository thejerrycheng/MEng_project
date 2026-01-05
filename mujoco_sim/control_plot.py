import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
START_Q = np.deg2rad([0, -45, -90, 0, 0, 0])

# Realistic Simulation Parameters
T_END = 5.0          
DT = 0.002           
MOVE_SPEED = 0.1     # 0.1 meters per second - realistic for 6-DOF arms
KP = 400.0           # Tuned to avoid torque saturation
KD = 40.0            

# Path Definition
POS_A = np.array([0.4, -0.2, 0.3])
POS_B = np.array([0.4,  0.2, 0.6])

def get_cartesian_traj(t, t_end):
    """Quintic scaling for smooth acceleration."""
    s = (10 * (t/t_end)**3 - 15 * (t/t_end)**4 + 6 * (t/t_end)**5)
    s = np.clip(s, 0, 1)
    return POS_A + s * (POS_B - POS_A)

def run_simulation(controller_type="ID"):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
    
    data.qpos[:6] = START_Q
    mujoco.mj_forward(model, data)
    
    time_log, error_log, pos_history = [], [], []
    steps = int(T_END / DT)
    
    current_q_ref = START_Q.copy()

    print(f"  > Simulating {controller_type} at {MOVE_SPEED} m/s...")
    for i in range(steps):
        t = i * DT
        target_pos_cart = get_cartesian_traj(t, T_END)
        
        # 1. IK Step to find q_desired and dq_desired
        prev_q_ref = current_q_ref.copy()
        
        # Temporary data for IK
        data_ik = mujoco.MjData(model)
        data_ik.qpos[:6] = current_q_ref
        mujoco.mj_forward(model, data_ik)
        
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data_ik, jacp, None, ee_id)
        J = jacp[:, :6]
        
        # Update q_ref
        cart_err = target_pos_cart - data_ik.body(ee_id).xpos
        q_update = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), cart_err)
        current_q_ref += q_update
        
        # Calculate Target Velocity (Feedforward)
        current_dq_ref = (current_q_ref - prev_q_ref) / DT

        # 2. Control Logic
        e = current_q_ref - data.qpos[:6]
        de = current_dq_ref - data.qvel[:6]
        
        if controller_type == "ID":
            # Correct ID law: tau = M(q)(KP*e + KD*de) + bias
            data.qacc[:6] = KP * e + KD * de
            mujoco.mj_inverse(model, data)
            tau = data.qfrc_inverse[:6].copy()
        else:
            # Standard PD (Raw torque application)
            tau = KP * e + KD * de
            
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        
        # Logging
        actual_pos = data.body(ee_id).xpos.copy()
        pos_history.append(actual_pos)
        error_log.append(np.linalg.norm(target_pos_cart - actual_pos))
        time_log.append(t)
        
        if i % (steps // 10) == 0:
            sys.stdout.write(f"\r    Progress: {(i/steps)*100:3.0f}%")
            sys.stdout.flush()
            
    print("\r    Progress: 100% | Done.")
    return time_log, error_log, np.array(pos_history)

# --- Execute and Generate Plots ---
id_time, id_err, id_pos = run_simulation("ID")
pd_time, pd_err, pd_pos = run_simulation("PD")

plt.figure(figsize=(10, 5))
plt.plot(id_time, np.array(id_err)*1000, 'b', label='Inverse Dynamics')
plt.plot(pd_time, np.array(pd_err)*1000, 'r--', label='Standard PD')
plt.title(f'Tracking Error Comparison at {MOVE_SPEED} m/s')
plt.ylabel("Error (mm)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CURR_DIR, "debugged_error.png"))