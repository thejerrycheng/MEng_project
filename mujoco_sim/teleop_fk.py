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

# Moving Average Buffer for Log (3 timesteps)
torque_history = deque(maxlen=3)

def on_press(key):
    global target_q
    try:
        char = key.char
        control_map = {
            'q': (0, 1), 'a': (0, -1), 'w': (1, 1), 's': (1, -1),
            'e': (2, 1), 'd': (2, -1), 'r': (3, 1), 'f': (3, -1),
            't': (4, 1), 'g': (4, -1), 'y': (5, 1), 'h': (5, -1)
        }
        if char in control_map:
            idx, direction = control_map[char]
            target_q[idx] = np.clip(target_q[idx] + direction * 1.0, joint_limits[idx][0], joint_limits[idx][1])
    except AttributeError: pass

def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: XML not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISKinematics()

    data.qpos[:6] = np.deg2rad(HOME_POSITION)
    
    # PID Gains
    kp = np.array([1200.0, 5000.0, 3000.0, 600.0, 400.0, 300.0])
    kd = np.array([120.0,  500.0,  200.0,  60.0,  40.0,  30.0])

    # Low Pass Filter parameter (0.0 to 1.0)
    # Lower is smoother but adds more lag
    lpf_alpha = 0.2
    prev_torque = np.zeros(6)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    os.system('clear' if os.name == 'posix' else 'cls')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. Calculate raw PID Torque
            current_q_deg = np.rad2deg(data.qpos[:6])
            raw_torque = kp * (target_q - current_q_deg) - kd * np.rad2deg(data.qvel[:6])

            # 2. Apply Low-Pass Filter
            # Formula: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
            filtered_torque = lpf_alpha * raw_torque + (1 - lpf_alpha) * prev_torque
            data.ctrl[:6] = filtered_torque
            prev_torque = filtered_torque

            # 3. Store in history for moving average log
            # actuator_force[0:6] captures the final torque applied in simulation
            torque_history.append(data.actuator_force[:6].copy())

            mujoco.mj_step(model, data)
            viewer.sync()

            if int(data.time * 100) % 10 == 0:
                avg_torque = np.mean(torque_history, axis=0)
                sim_pos = data.body('wrist2').xpos 
                analytical_pos = kin.calculate_fk(current_q_deg)
                
                sys.stdout.write("\033[H")
                out = [
                    "=================== IRIS ROBOT DASHBOARD ===================",
                    f" Time: {data.time:6.2f}s | LPF Alpha: {lpf_alpha} | Torque: 10-step Avg",
                    "-" * 65,
                    f"{'Joint':<8} | {'Target (°)':^12} | {'Actual (°)':^12} | {'Avg Trq(Nm)':^12}",
                    "-" * 65
                ]
                for i in range(6):
                    out.append(f" J{i+1:<7} | {target_q[i]:10.1f}   | {current_q_deg[i]:10.1f}   | {avg_torque[i]:10.2f}")
                
                out.extend([
                    "-" * 65,
                    f"{'EE POS':<8} | {'X':^12} | {'Y':^12} | {'Z':^12}",
                    f"{'Sim':<8} | {sim_pos[0]:10.3f}   | {sim_pos[1]:10.3f}   | {sim_pos[2]:10.3f}",
                    f"{'DH':<8} | {analytical_pos[0]:10.3f}   | {analytical_pos[1]:10.3f}   | {analytical_pos[2]:10.3f}",
                    "-" * 65,
                    f" L2 Error: {np.linalg.norm(sim_pos - analytical_pos):.6f}",
                    "============================================================"
                ])
                sys.stdout.write("\n".join(out) + "\n")
                sys.stdout.flush()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()