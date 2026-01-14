#!/usr/bin/env python
# ---------------------------------------------------------
# CRITICAL FIX: Set backend to TkAgg
# ---------------------------------------------------------
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    matplotlib.use('Agg')

import time
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# ----------------------------
# SDK Setup
# ----------------------------
sys.path.append('../lib')
try:
    from unitree_actuator_sdk import *
except ImportError:
    print("Error: unitree_actuator_sdk not found.")
    sys.exit(1)

# ----------------------------
# Config & Differential Constants
# ----------------------------
SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
MOTOR_TYPE = MotorType.GO_M8010_6
CALIB_FILE = "calibration_data.csv"

# --- WRIST MATH CONSTANTS (Match ROS Node) ---
WRIST_PITCH_SIGN = -1.0
WRIST_ROLL_SIGN  = -1.0
ROLL_HOME_OFFSET = np.pi

# ----------------------------
# Hardware Init
# ----------------------------
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()
GEAR = queryGearRatio(MOTOR_TYPE)

# ----------------------------
# Helpers
# ----------------------------
def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print("Calib file not found.")
        return np.zeros(6)
    offsets = {}
    with open(CALIB_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            offsets[int(row["motor_id"])] = float(row["zero_offset_rad"])
    return np.array([offsets[i] for i in MOTOR_IDS])

def read_motor_raw(motor_id):
    """Returns Raw Motor Angle (Output Shaft)"""
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    
    # Passive Keep-Alive
    raw = data.q
    cmd.q = raw
    cmd.dq, cmd.kp, cmd.kd, cmd.tau = 0.0, 0.0, 0.0, 0.0
    serial.sendRecv(cmd, data)
    
    return data.q / GEAR

def get_robot_state(offsets):
    """
    Reads 6 raw motors -> Converts to 6 Joint Angles using Differential Logic
    """
    # 1. Read Raw
    raw = []
    for mid in MOTOR_IDS:
        # Retry loop
        for _ in range(3):
            val = read_motor_raw(mid)
            if abs(val) < 100: 
                raw.append(val)
                break
            time.sleep(0.002)
        else:
            return None # Error
    
    raw = np.array(raw)
    
    # 2. Apply Zero Offsets
    zeroed = raw - offsets
    
    # 3. Map to Joint Angles (q_out)
    q_out = np.zeros(6)
    
    # Joints 0-3: Direct Mapping
    q_out[0:4] = zeroed[0:4]
    
    # Joints 4-5: Differential Wrist Mapping
    # Logic matches ROS CalibrationNode state_cb
    # Pitch (J4) = -0.5 * (M5 - M6)
    # Roll  (J5) = -0.5 * (M5 + M6) + PI
    
    m5 = zeroed[4]
    m6 = zeroed[5]
    
    pitch = WRIST_PITCH_SIGN * 0.5 * (m5 - m6)
    roll  = WRIST_ROLL_SIGN  * 0.5 * (m5 + m6) + ROLL_HOME_OFFSET
    
    q_out[4] = pitch
    q_out[5] = roll
    
    return q_out

# ----------------------------
# Kinematics & Plotting
# ----------------------------
class IRISKinematics:
    def __init__(self):
        self.link_configs = [
            {'pos': [0, 0, 0.2487],        'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            {'pos': [0.0218, 0, 0.059],    'euler': [0, 90, 180], 'axis': [0, 0, 1]}, 
            {'pos': [0.299774, 0, -0.0218],'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            {'pos': [0.02, 0, 0],          'euler': [0, 90, 0],   'axis': [0, 0, 1]}, 
            {'pos': [0, 0, 0.315],         'euler': [0, -90, 0],  'axis': [0, 0, 1]}, 
            {'pos': [0.042824, 0, 0],      'euler': [0, 90, 0],   'axis': [0, 0, 1]}  
        ]

    def compute_positions(self, q):
        points = [[0,0,0]]
        T = np.eye(4)
        for i, cfg in enumerate(self.link_configs):
            # Static
            Ts = np.eye(4); Ts[:3,3] = cfg['pos']
            Ts[:3,:3] = R.from_euler('xyz', cfg['euler'], degrees=True).as_matrix()
            # Joint
            rv = np.array(cfg['axis']) * q[i]
            Tj = np.eye(4); Tj[:3,:3] = R.from_rotvec(rv).as_matrix()
            
            T = T @ Ts @ Tj
            points.append(T[:3, 3].tolist())
        return np.array(points)

def update_plot(line, points):
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])

# ----------------------------
# Main
# ----------------------------
def main():
    print("==========================================")
    print(" DIFFERENTIAL WRIST FK VISUALIZER")
    print("==========================================")
    
    offsets = load_calibration()
    kin = IRISKinematics()
    
    # Init Plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 0.8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    line, = ax.plot([],[],[], 'o-', lw=3)

    print("Monitoring... (Ctrl+C to stop)")
    try:
        while True:
            # 1. Get State (With Diff Logic)
            q_joint = get_robot_state(offsets)
            
            if q_joint is not None:
                # 2. Compute FK
                pts = kin.compute_positions(q_joint)
                ee = pts[-1]
                
                # 3. Update
                update_plot(line, pts)
                ax.set_title(f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}]")
                
                status = f"J4(P):{q_joint[4]:.2f} | J5(R):{q_joint[5]:.2f}"
                print(f"\r{status} | EE_Z: {ee[2]:.3f}", end='', flush=True)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nDone.")

if __name__ == "__main__":
    main()