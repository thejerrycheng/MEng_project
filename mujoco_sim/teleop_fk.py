import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from pynput import keyboard

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")

# Home position (Degrees)
HOME_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

class IRISKinematics:
    def __init__(self):
        # DH Parameters [theta_offset, d, a, alpha]
        self.params = [
            {'th_off': 0,   'd': 2.5,  'a': 0.22, 'alpha': 90},  # J1
            {'th_off': 0,  'd': 0.0,  'a': 3.0,  'alpha': 0},   # J2
            {'th_off': 0,   'd': 0.0,  'a': 0.2,  'alpha': 0},  # J3
            {'th_off': 0,   'd': 3.16, 'a': 0.0,  'alpha': -90}, # J4
            {'th_off': 0,   'd': 0.0,  'a': 0.0,  'alpha': 0},  # J5
            {'th_off': 0,   'd': 0.43, 'a': 0.0,  'alpha': 0}    # J6
        ]

    def dh_matrix(self, theta, d, a, alpha):
        th, al = np.deg2rad(theta), np.deg2rad(alpha)
        return np.array([
            [np.cos(th), -np.sin(th)*np.cos(al),  np.sin(th)*np.sin(al), a*np.cos(th)],
            [np.sin(th),  np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
            [0,           np.sin(al),             np.cos(al),            d],
            [0,           0,                      0,                     1]
        ])

    def calculate_fk(self, q):
        T = np.eye(4)
        for i in range(6):
            theta = q[i] + self.params[i]['th_off']
            T = T @ self.dh_matrix(theta, self.params[i]['d'], self.params[i]['a'], self.params[i]['alpha'])
        return T[:3, 3]

# --- Keyboard Handling ---
active_keys = set()
def on_press(key):
    try: active_keys.add(key.char)
    except AttributeError: pass
def on_release(key):
    try: active_keys.remove(key.char)
    except (AttributeError, KeyError): pass

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISKinematics()

    control_map = {
        'q': (0, 1), 'a': (0, -1), 'w': (1, 1), 's': (1, -1),
        'e': (2, 1), 'd': (2, -1), 'r': (3, 1), 'f': (3, -1),
        't': (4, 1), 'g': (4, -1), 'y': (5, 1), 'h': (5, -1)
    }

    # Initialize state and target to Home
    target_q = HOME_POSITION.copy()
    data.qpos[:6] = np.deg2rad(HOME_POSITION)
    
    # --- Higher P Values for Shoulder and Elbow ---
    # Indices:    [J1,    J2,     J3,     J4,    J5,    J6]
    kp = np.array([800.0, 1500.0, 1200.0, 400.0, 300.0, 200.0])
    kd = np.array([80.0,  150.0,  120.0,  40.0,  30.0,  20.0])

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("IRIS Controller Started. Tracking Joint States and Torques...")
        
        while viewer.is_running():
            step_start = time.time()

            # 1. Update target position from keys
            for key, (idx, direction) in control_map.items():
                if key in active_keys:
                    target_q[idx] += direction * 0.5 

            # 2. Get current state in Degrees
            current_q_deg = np.rad2deg(data.qpos[:6])
            current_dq_deg = np.rad2deg(data.qvel[:6])
            
            # 3. PID Control (Torque Calculation)
            # data.ctrl represents the raw input to the <motor> tags
            applied_torque = kp * (target_q - current_q_deg) - kd * current_dq_deg
            data.ctrl[:6] = applied_torque

            # 4. Simulation Step
            mujoco.mj_step(model, data)
            viewer.sync()

            # 5. Dashboard Logging (Cleaned for readability)
            if int(data.time * 100) % 20 == 0:
                sim_pos = data.body('wrist2').xpos 
                analytical_pos = kin.calculate_fk(current_q_deg)
                
                os.system('clear')
                print(f"--- IRIS REAL-TIME STATUS DASHBOARD ---")
                print(f"Controls: [QAWSEDRFTGYH] | Time: {data.time:6.2f}s")
                print("-" * 55)
                
                # Joint State Table
                print(f"{'Joint':<8} | {'Target (°)':<12} | {'Actual (°)':<12} | {'Torque (Nm)':<12}")
                print("-" * 55)
                for i in range(6):
                    print(f"J{i+1:<7} | {target_q[i]:10.1f}   | {current_q_deg[i]:10.1f}   | {data.actuator_force[i]:10.2f}")
                
                print("-" * 55)
                # EE Position Table
                print(f"EE Source   | {'X':^10} | {'Y':^10} | {'Z':^10}")
                print(f"ANALYTICAL  | {analytical_pos[0]:10.3f} | {analytical_pos[1]:10.3f} | {analytical_pos[2]:10.3f}")
                print(f"SIMULATION  | {sim_pos[0]:10.3f} | {sim_pos[1]:10.3f} | {sim_pos[2]:10.3f}")
                print(f"L2 ERROR    | {np.linalg.norm(sim_pos - analytical_pos):.6f}")

            # Real-time sync
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()