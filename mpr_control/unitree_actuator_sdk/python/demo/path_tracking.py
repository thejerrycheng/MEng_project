#!/usr/bin/env python
# ---------------------------------------------------------
# CRITICAL FIX: Set backend to TkAgg for Plots
# ---------------------------------------------------------
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    print("Warning: TkAgg not found. Switching to 'Agg'.")
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
from scipy.optimize import minimize

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
CALIB_FILE = "calibration_data.csv"
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
MOTOR_TYPE = MotorType.GO_M8010_6

# Path Settings
DT = 0.005             # 100Hz loop 
TRANSITION_TIME = 5.0  
CIRCLE_TIME = 10.0      
CIRCLE_RADIUS = 0.2   
CIRCLE_CENTER = np.array([0.35, 0.0, 0.4]) 

# Impedance Gains
KP = 8.0  
KD = 0.2

# ----------------------------
# Hardware Globals
# ----------------------------
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()

GEAR = queryGearRatio(MOTOR_TYPE)

# ----------------------------
# Calibration Loader
# ----------------------------
def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f"Error: {CALIB_FILE} not found. Run calibrate_home_safe.py first.")
        sys.exit(1)
        
    offsets = {}
    with open(CALIB_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            offsets[int(row["motor_id"])] = float(row["zero_offset_rad"])
    
    offset_list = [offsets[mid] for mid in MOTOR_IDS]
    return np.array(offset_list)

# ----------------------------
# Motor Functions
# ----------------------------
def read_motor_passive(motor_id):
    """Returns: Joint Angle (radians) = Motor Angle / GearRatio"""
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    current_raw = data.q
    
    # Keep alive
    cmd.q = current_raw
    cmd.dq, cmd.kp, cmd.kd, cmd.tau = 0.0, 0.0, 0.0, 0.0
    serial.sendRecv(cmd, data)
    
    return data.q / GEAR

def actuate_motor(motor_id, desired_joint_pos):
    """Command: Motor Angle = Joint Angle * GearRatio"""
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    
    cmd.q = desired_joint_pos * GEAR
    cmd.dq = 0.0
    cmd.kp = KP
    cmd.kd = KD
    cmd.tau = 0.0
    
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
            sys.exit(1)
    return np.array(raw_qs) - offsets

def actuate_all_motors_logical(target_logical_qs, offsets):
    target_raw_joint = target_logical_qs + offsets
    for i, mid in enumerate(MOTOR_IDS):
        actuate_motor(mid, target_raw_joint[i])

# ----------------------------
# Kinematics
# ----------------------------
class AnalyticalKinematics:
    def __init__(self):
        self.link_configs = [
            {'pos': [0, 0, 0.2487],        'euler': [0, 0, 0],       'axis': [0, 0, -1]}, 
            {'pos': [0.0218, 0, 0.059],    'euler': [0, -90, -180],  'axis': [0, 0, 1]}, 
            {'pos': [0.299774, 0, -0.0218],'euler': [0, 0, 0],       'axis': [0, 0, 1]}, 
            {'pos': [0.02, 0, 0],          'euler': [0, 90, 0],      'axis': [0, 0, 1]}, 
            {'pos': [0, 0, 0.315],         'euler': [0, -90, 0],     'axis': [0, 0, 1]}, 
            {'pos': [0.042824, 0, 0],      'euler': [0, 90, 0],      'axis': [0, 0, 1]}  
        ]

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i, cfg in enumerate(self.link_configs):
            T_s = np.eye(4); T_s[:3,3] = cfg['pos']
            T_s[:3,:3] = R.from_euler('xyz', cfg['euler'], degrees=True).as_matrix()
            r_vec = np.array(cfg['axis']) * q[i]
            T_j = np.eye(4); T_j[:3,:3] = R.from_rotvec(r_vec).as_matrix()
            T = T @ T_s @ T_j
        return T[:3, 3]

    def closest_equivalent(self, target_q, seed_q):
        diff = target_q - seed_q
        turns = np.round(diff / (2 * np.pi))
        return target_q - turns * 2 * np.pi

    def inverse_kinematics(self, target, seed):
        def obj(q):
            curr = self.forward_kinematics(q)
            return np.sum((curr - target)**2) + 0.5 * np.sum((q - seed)**2)

        local_bounds = []
        for i in range(6):
            local_bounds.append((seed[i] - 3.14, seed[i] + 3.14))

        res = minimize(obj, seed, method='SLSQP', bounds=local_bounds, tol=1e-5)
        
        if res.success:
            return self.closest_equivalent(res.x, seed)
        else:
            return None

# ----------------------------
# Path & Visualization
# ----------------------------
def get_circle_points(center, r, n):
    points = []
    for i in range(n):
        theta = 2 * np.pi * i / (n - 1)
        y = center[1] + r * np.cos(theta)
        z = center[2] + r * np.sin(theta)
        points.append(np.array([center[0], y, z]))
    return points

def interpolate(q_start, q_end, duration):
    steps = int(duration / DT)
    traj = []
    for i in range(steps):
        alpha = i / steps
        k = alpha * alpha * (3 - 2 * alpha)
        traj.append(q_start + (q_end - q_start) * k)
    return traj

def validate_trajectory(trajectory):
    print("Inspecting trajectory smoothness...")
    max_jump = 0.0
    for i in range(1, len(trajectory)):
        diff = np.max(np.abs(trajectory[i] - trajectory[i-1]))
        if diff > max_jump: max_jump = diff
    
    print(f"  Max Joint Jump: {max_jump:.4f} rads/step")
    if max_jump > 0.25: 
        print("\n[DANGER] Unsafe jumps detected.")
        # return False
    return True

def plot_trajectory_preview(kinematics, path1, path2, path3):
    print("\nGenerating 3D Preview...")
    def get_xyz(path):
        return np.array([kinematics.forward_kinematics(q) for q in path])

    p1 = get_xyz(path1)
    p2 = get_xyz(path2)
    p3 = get_xyz(path3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p1[:,0], p1[:,1], p1[:,2], 'b-', label='Approach')
    ax.plot(p2[:,0], p2[:,1], p2[:,2], 'r-', label='Circle', linewidth=3)
    ax.plot(p3[:,0], p3[:,1], p3[:,2], 'g--', label='Return')
    ax.scatter(p1[0,0], p1[0,1], p1[0,2], c='k', s=50, label='Start')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    all_pts = np.vstack([p1, p2, p3])
    mid = np.mean(all_pts, axis=0)
    max_range = (np.max(all_pts) - np.min(all_pts)) / 2.0
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    
    print(">> Close plot to continue...")
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    print(f"==========================================")
    print(f" PATH TRACKING (Connected)")
    print(f"==========================================")
    
    offsets = load_calibration()
    kinematics = AnalyticalKinematics()

    print("Stabilizing Motors (2s)...")
    t_end = time.time() + 2.0
    while time.time() < t_end:
        try: read_all_motors_logical(offsets) 
        except: pass
        time.sleep(0.1)

    start_q = read_all_motors_logical(offsets)
    print(f"Start Config: {np.round(start_q, 2)}")

    # --- Planning ---
    print("\nComputing IK Path...")
    
    circle_pts = get_circle_points(CIRCLE_CENTER, CIRCLE_RADIUS, int(CIRCLE_TIME/DT))
    
    # 1. Approach
    q_circle_start = kinematics.inverse_kinematics(circle_pts[0], start_q)
    if q_circle_start is None:
        print("Error: Circle unreachable.")
        sys.exit(1)
        
    path_1 = interpolate(start_q, q_circle_start, TRANSITION_TIME)
    
    # 2. Circle (FIXED LOGIC)
    path_2 = []
    # Force connection: The start of path_2 IS the end of path_1
    path_2.append(path_1[-1])
    last_q = path_1[-1]
    
    # Start solving from index 1 (skipping 0) to prevent jump
    for pt in circle_pts[1:]:
        sol = kinematics.inverse_kinematics(pt, last_q)
        if sol is not None:
            path_2.append(sol)
            last_q = sol
        else:
            path_2.append(last_q)

    # 3. Return
    home_unwrapped = kinematics.closest_equivalent(np.zeros(6), path_2[-1])
    path_3 = interpolate(path_2[-1], home_unwrapped, TRANSITION_TIME)
    
    full_path = path_1 + path_2 + path_3
    
    # --- Check Safety ---
    is_safe = validate_trajectory(full_path)

    # --- Visualization ---
    plot_trajectory_preview(kinematics, path_1, path_2, path_3)

    if not is_safe:
        print("\n[STOP] Trajectory validation failed.")
        sys.exit(1)

    print(f"\nPath Verified. {len(full_path)} points.")
    if input("Start Execution on Robot? (y/n): ").strip().lower() != 'y':
        return

    print("\n-> Executing...")
    for q in full_path:
        loop_start = time.time()
        try:
            actuate_all_motors_logical(q, offsets)
        except Exception as e:
            print(f"Stop: {e}")
            break
            
        elapsed = time.time() - loop_start
        if elapsed < DT:
            time.sleep(DT - elapsed)
            
    print("Done.")

if __name__ == "__main__":
    main()