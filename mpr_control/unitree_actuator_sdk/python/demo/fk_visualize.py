#!/usr/bin/env python
# ---------------------------------------------------------
# CRITICAL FIX: Set backend to TkAgg for live plotting
# ---------------------------------------------------------
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    print("Warning: TkAgg not found. Switching to 'Agg' (No GUI).")
    matplotlib.use('Agg')

import time
import math
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
    print("Error: unitree_actuator_sdk not found. Check library path.")
    sys.exit(1)

# ----------------------------
# Configuration
# ----------------------------
SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
MOTOR_TYPE = MotorType.GO_M8010_6
CALIB_FILE = "calibration_data.csv"

# ----------------------------
# Hardware Globals
# ----------------------------
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()
GEAR = queryGearRatio(MOTOR_TYPE)

# ----------------------------
# Kinematics Class (CORRECTED)
# ----------------------------
class IRISKinematics:
    def __init__(self):
        # Adjusted parameters to match Physical Reality based on your feedback
        self.link_configs = [
            # Joint 1: Base (FIX: Axis inverted to -1 to match direction)
            {'pos': [0, 0, 0.2487],        'euler': [0, 0, 0],    'axis': [0, 0, -1]}, 
            
            # Joint 2: Shoulder (FIX: Euler changed 180->0 to point UP)
            {'pos': [0.0218, 0, 0.059],    'euler': [0, -90, -180],   'axis': [0, 0, 1]}, 
            
            # Joint 3: Elbow
            {'pos': [0.299774, 0, -0.0218],'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            
            # Joint 4: Arm Link 2
            {'pos': [0.02, 0, 0],          'euler': [0, 90, 0],   'axis': [0, 0, 1]}, 
            
            # Joint 5: Wrist 1
            {'pos': [0, 0, 0.315],         'euler': [0, -90, 0],  'axis': [0, 0, 1]}, 
            
            # Joint 6: Wrist 2
            {'pos': [0.042824, 0, 0],      'euler': [0, 90, 0],   'axis': [0, 0, 1]}  
        ]

    def get_transform_matrix(self, config, q_val):
        T_static = np.eye(4)
        T_static[:3, 3] = config['pos']
        # Intrinsic xyz matches standard robotics conventions usually
        r_static = R.from_euler('xyz', config['euler'], degrees=True).as_matrix()
        T_static[:3, :3] = r_static
        
        r_vec = np.array(config['axis']) * q_val
        r_joint = R.from_rotvec(r_vec).as_matrix()
        T_joint = np.eye(4)
        T_joint[:3, :3] = r_joint
        
        return T_static @ T_joint

    def compute_chain_positions(self, joint_angles):
        """Returns list of 3D positions [Base, J1, ... EE]"""
        positions = [[0,0,0]] 
        T_accum = np.eye(4)
        
        for i, cfg in enumerate(self.link_configs):
            q = joint_angles[i]
            T_local = self.get_transform_matrix(cfg, q)
            T_accum = T_accum @ T_local
            positions.append(T_accum[:3, 3].tolist())
            
        return np.array(positions)

# ----------------------------
# Hardware Functions
# ----------------------------
def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f"Error: {CALIB_FILE} not found. Returning zeros.")
        return np.zeros(6)
    offsets = {}
    with open(CALIB_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            offsets[int(row["motor_id"])] = float(row["zero_offset_rad"])
    return np.array([offsets[i] for i in MOTOR_IDS])

def read_motor_passive(motor_id):
    """Returns Joint Angle (radians)"""
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    
    # Keep alive
    cmd.q, cmd.dq, cmd.kp, cmd.kd, cmd.tau = data.q, 0.0, 0.0, 0.0, 0.0
    serial.sendRecv(cmd, data)
    
    return data.q / GEAR

def read_all_motors_logical(offsets):
    raw_qs = []
    for mid in MOTOR_IDS:
        for _ in range(3):
            val = read_motor_passive(mid)
            if abs(val) < 100.0: 
                raw_qs.append(val)
                break
            time.sleep(0.002)
        else:
            print(f"Error reading motor {mid}")
            return np.zeros(6)
    return np.array(raw_qs) - offsets

# ----------------------------
# Visualization
# ----------------------------
def init_plot():
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Real-Time Robot Pose (Corrected)')
    
    # Limits (meters)
    limit = 0.8
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, limit*1.5)
    
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    
    # Robot "Bones"
    line, = ax.plot([], [], [], 'o-', linewidth=5, markersize=8, color='blue')
    
    # Ground Plane
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 2), np.linspace(-0.5, 0.5, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='gray')
    
    return fig, ax, line

def update_plot(line, positions):
    line.set_data(positions[:, 0], positions[:, 1])
    line.set_3d_properties(positions[:, 2])

# ----------------------------
# Main Loop
# ----------------------------
def main():
    print("==========================================")
    print(" FK VISUALIZER (FIXED) ")
    print("==========================================")

    offsets = load_calibration()
    print(f"Calibration Loaded: {[round(x,2) for x in offsets]}")
    
    kinematics = IRISKinematics()
    fig, ax, line = init_plot()
    
    print("\nStarting Loop... (Ctrl+C to stop)")
    
    try:
        while True:
            # 1. Read Hardware
            q_logical = read_all_motors_logical(offsets)
            
            # 2. Compute FK
            points_3d = kinematics.compute_chain_positions(q_logical)
            ee = points_3d[-1]
            
            # 3. Update Plot
            update_plot(line, points_3d)
            
            # Info
            status = f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}]"
            ax.set_title(f"Pose | {status}")
            
            sys.stdout.write(f"\r{status} | J1:{q_logical[0]:.2f} J2:{q_logical[1]:.2f}")
            sys.stdout.flush()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopped.")
        plt.close()
        sys.exit(0)

if __name__ == "__main__":
    main()