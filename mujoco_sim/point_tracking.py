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

# Simulation Parameters
T_END = 5.0          
DT = 0.002           
KP = 600.0           # Proportional gain for joint space
KD = 50.0            # Derivative gain for joint space

# Path Definition (Point A to Point B)
POS_A = np.array([0.4, -0.2, 0.3])
POS_B = np.array([0.4,  0.2, 0.6])

def get_cartesian_target(t, t_end):
    """Smooth path interpolation using quintic scaling."""
    s = (10 * (t/t_end)**3 - 15 * (t/t_end)**4 + 6 * (t/t_end)**5)
    s = np.clip(s, 0, 1)
    return POS_A + s * (POS_B - POS_A)

def run_simulation(controller_type="ID"):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
    
    # Initialize Robot Position
    data.qpos[:6] = START_Q
    mujoco.mj_forward(model, data)
    
    time_log, error_log = [], []
    steps = int(T_END / DT)
    
    # We maintain a separate IK reference state that moves perfectly along the path
    q_ref = START_Q.copy()

    print(f"  > Simulating {controller_type} Controller...")
    for i in range(steps):
        t = i * DT
        target_pos_cart = get_cartesian_target(t, t_end=T_END)
        
        # --- 1. PREDICTIVE IK STEP ---
        # Find q_ref that puts the EE exactly on the target_pos_cart
        # We use a temporary data object to solve IK without affecting the simulation
        ik_data = mujoco.MjData(model)
        ik_data.qpos[:6] = q_ref
        mujoco.mj_forward(model, ik_data)
        
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, ik_data, jacp, None, ee_id)
        J = jacp[:, :6]
        
        # Calculate Cartesian velocity needed to reach target in one DT
        # cart_vel = (target - current_ee_at_q_ref) / DT
        dx = (target_pos_cart - ik_data.body(ee_id).xpos) / DT
        
        # Joint velocity needed: dq = J_inv * dx
        dq_ref = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), dx)
        
        # Update q_ref for the next step
        q_ref += dq_ref * DT

        # --- 2. CONTROL CALCULATION ---
        # Error between where the robot IS and where the IK says it SHOULD BE
        q_err = q_ref - data.qpos[:6]
        dq_err = dq_ref - data.qvel[:6]
        
        if controller_type == "ID":
            # Computed Torque Control: tau = M(q) * (KP*e + KD*de) + bias(q, v)
            # Setting qacc triggers mj_inverse to calculate necessary torques
            data.qacc[:6] = KP * q_err + KD * dq_err
            mujoco.mj_inverse(model, data)
            tau = data.qfrc_inverse[:6].copy()
        else:
            # Standard PD: No model awareness, just raw gain application
            tau = KP * q_err + KD * dq_err

        # --- 3. STEP PHYSICS ---
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        
        # --- 4. LOGGING END-EFFECTOR ERROR ---
        actual_ee_pos = data.body(ee_id).xpos
        ee_error = np.linalg.norm(target_pos_cart - actual_ee_pos)
        
        time_log.append(t)
        error_log.append(ee_error)
        
        if i % (steps // 10) == 0:
            sys.stdout.write(f"\r    Progress: {(i/steps)*100:3.0f}%")
            sys.stdout.flush()
            
    print("\r    Progress: 100% | Done.")
    return time_log, error_log

# --- Main Execution and Plotting ---
print("="*60)
print("REVISED TRACKING ERROR ANALYSIS (IK-BASED REFERENCE)")
print("="*60)

id_time, id_err = run_simulation("ID")
pd_time, pd_err = run_simulation("PD")

plt.figure(figsize=(12, 6))
# Convert to mm for clarity
plt.plot(id_time, np.array(id_err) * 1000, label='Inverse Dynamics (Computed Torque)', color='blue', linewidth=2)
plt.plot(pd_time, np.array(pd_err) * 1000, label='Standard PD Control', color='red', linestyle='--', linewidth=1.5)

plt.title('End-Effector Path Tracking Error Over Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Euclidean Error (mm)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = os.path.join(CURR_DIR, "high_precision_error_plot.png")
plt.savefig(output_path)

print("-" * 60)
print(f"Max Error (ID): {np.max(id_err)*1000:.4f} mm")
print(f"Max Error (PD): {np.max(pd_err)*1000:.4f} mm")
print(f"Comparison plot saved to: {output_path}")
print("=" * 60)