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
XML_PATH = os.path.join(CURR_DIR, "assets", "iris_with_axis.xml")
HOME_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

class IRISKinematics:
    def __init__(self):
        # DH Parameters [theta_offset, d, a, alpha] in Meters
        self.params = [
            {'th_off': 0,   'd': 0.2487,   'a': 0.0218,   'alpha': 90},  # J1
            {'th_off': 90,  'd': 0.0,      'a': 0.299774, 'alpha': 0},   # J2
            {'th_off': 0,   'd': 0.0,      'a': 0.02,     'alpha': 90},  # J3
            {'th_off': 0,   'd': 0.315,    'a': 0.0,      'alpha': -90}, # J4
            {'th_off': 0,   'd': 0.0,      'a': 0.0,      'alpha': 90},  # J5
            {'th_off': 0,   'd': 0.042824, 'a': 0.0,      'alpha': 0}    # J6
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

# --- Global Control State ---
target_q = HOME_POSITION.copy()
joint_limits = [(-170, 170), (-170, 170), (-150, 150), (-180, 180), (-100, 100), (-360, 360)]
torque_history = deque(maxlen=10)
active_keys = set()

def on_press(key):
    try: active_keys.add(key.char)
    except AttributeError: pass

def on_release(key):
    try: active_keys.remove(key.char)
    except (AttributeError, KeyError): pass

def main():
    global target_q
    if not os.path.exists(XML_PATH):
        print(f"Error: XML not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISKinematics()

    # Initialization
    data.qpos[:6] = np.deg2rad(HOME_POSITION)
    
    # Realistic Unitree-style gains (Low Kp, High Damping)
    kp = np.array([80.0, 120.0, 100.0, 40.0, 30.0, 20.0])
    kd = np.array([3.5,  5.5,   4.5,   1.8,  1.2,  1.0])

    lpf_alpha = 0.25 
    prev_torque = np.zeros(6)
    move_speed = 0.1 # Degrees per simulation step while key is held

    control_map = {
        'q': (0, 1), 'a': (0, -1), 'w': (1, 1), 's': (1, -1),
        'e': (2, 1), 'd': (2, -1), 'r': (3, 1), 'f': (3, -1),
        't': (4, 1), 'g': (4, -1), 'y': (5, 1), 'h': (5, -1)
    }

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    os.system('clear' if os.name == 'posix' else 'cls')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. Multi-Key Processing
            # Iterate through the map; if key is in active_keys, increment target_q
            for key, (idx, direction) in control_map.items():
                if key in active_keys:
                    new_val = target_q[idx] + direction * move_speed
                    target_q[idx] = np.clip(new_val, joint_limits[idx][0], joint_limits[idx][1])

            # 2. Physics States
            current_q_deg = np.rad2deg(data.qpos[:6])
            current_v_deg = np.rad2deg(data.qvel[:6])

            # 3. Gravity Compensation
            mujoco.mj_forward(model, data)
            gravity_comp = data.qfrc_passive[:6]

            # 4. Control Calculation (PD + Gravity Comp)
            raw_torque = (kp * (target_q - current_q_deg)) - (kd * current_v_deg) + gravity_comp
            
            # 5. Low Pass Filter (LPF) for Torque
            filtered_torque = lpf_alpha * raw_torque + (1 - lpf_alpha) * prev_torque
            data.ctrl[:6] = filtered_torque
            prev_torque = filtered_torque

            # 6. Logging History
            torque_history.append(data.actuator_force[:6].copy())

            mujoco.mj_step(model, data)
            viewer.sync()

            # 7. Flicker-Free Dashboard
            if int(data.time * 100) % 10 == 0:
                avg_torque = np.mean(torque_history, axis=0)
                sim_pos = data.body('wrist2').xpos 
                analytical_pos = kin.calculate_fk(current_q_deg)
                
                sys.stdout.write("\033[H")
                out = [
                    "=================== IRIS MULTI-KEY DASHBOARD ===================",
                    f" Time: {data.time:6.2f}s | Mode: Unitree Realism | LPF: {lpf_alpha}",
                    "-" * 68,
                    f"{'Joint':<8} | {'Target (°)':^12} | {'Actual (°)':^12} | {'Avg Trq(Nm)':^12}",
                    "-" * 68
                ]
                for i in range(6):
                    out.append(f" J{i+1:<7} | {target_q[i]:10.1f}   | {current_q_deg[i]:10.1f}   | {avg_torque[i]:10.2f}")
                
                out.extend([
                    "-" * 68,
                    f"{'EE POS':<8} | {'X (m)':^12} | {'Y (m)':^12} | {'Z (m)':^12}",
                    f"{'Sim':<8} | {sim_pos[0]:10.3f}   | {sim_pos[1]:10.3f}   | {sim_pos[2]:10.3f}",
                    f"{'DH':<8} | {analytical_pos[0]:10.3f}   | {analytical_pos[1]:10.3f}   | {analytical_pos[2]:10.3f}",
                    "-" * 68,
                    f" L2 Error: {np.linalg.norm(sim_pos - analytical_pos):.6f}",
                    "================================================================"
                ])
                sys.stdout.write("\n".join(out) + "\n")
                sys.stdout.flush()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()