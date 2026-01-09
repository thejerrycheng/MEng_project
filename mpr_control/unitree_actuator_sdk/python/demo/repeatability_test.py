#!/usr/bin/env python
import time
import sys
import csv
import argparse
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import os

# ----------------------------
# Configuration & Setup
# ----------------------------
sys.path.append('../lib')
try:
    from unitree_actuator_sdk import *
except ImportError:
    print("Error: 'unitree_actuator_sdk' not found. Check library path.")
    sys.exit(1)

SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_IDS = [0, 1, 2, 3, 4, 5]

# Impedance settings for active control
IMPEDANCE_SETTINGS = {
    0: {"kp": 1.5, "kd": 0.05},
    1: {"kp": 2.5, "kd": 0.15},
    2: {"kp": 2.5, "kd": 0.15},
    3: {"kp": 1.0, "kd": 0.025},
    4: {"kp": 1.0, "kd": 0.125},
    5: {"kp": 1.0, "kd": 0.125},
}

# ----------------------------
# IRIS Kinematics Class
# ----------------------------
class IRISKinematics:
    def __init__(self):
        # Link definitions strictly extracted from XML
        self.link_configs = [
            {'pos': [0, 0, 0.2487],        'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            {'pos': [0.0218, 0, 0.059],    'euler': [0, 90, 180], 'axis': [0, 0, 1]}, 
            {'pos': [0.299774, 0, -0.0218],'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            {'pos': [0.02, 0, 0],          'euler': [0, 90, 0],   'axis': [0, 0, 1]}, 
            {'pos': [0, 0, 0.315],         'euler': [0, -90, 0],  'axis': [0, 0, 1]}, 
            {'pos': [0.042824, 0, 0],      'euler': [0, 90, 0],   'axis': [0, 0, 1]}  
        ]

    def get_local_transform(self, config, q_rad):
        T = np.eye(4)
        T[:3, 3] = config['pos']
        
        # Fixed Rotation
        R_fixed = np.eye(3)
        if any(config['euler']):
            quat_e = np.zeros(4)
            mujoco.mju_euler2Quat(quat_e, np.deg2rad(config['euler']), 'xyz')
            res_e = np.zeros(9)
            mujoco.mju_quat2Mat(res_e, quat_e)
            R_fixed = res_e.reshape(3, 3)
            
        # Joint Rotation
        quat_j = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat_j, np.array(config['axis']), q_rad)
        res_j = np.zeros(9)
        mujoco.mju_quat2Mat(res_j, quat_j)
        R_joint = res_j.reshape(3, 3)
        
        T[:3, :3] = R_fixed @ R_joint
        return T

    def forward_kinematics(self, q_rad):
        """Calculates the global (x, y, z) of the end-effector."""
        T_accum = np.eye(4)
        for i in range(len(self.link_configs)):
            # Ensure q_rad has enough elements; pad if necessary or truncate
            angle = q_rad[i] if i < len(q_rad) else 0.0
            T_local = self.get_local_transform(self.link_configs[i], angle)
            T_accum = T_accum @ T_local
        return T_accum[:3, 3]

# ----------------------------
# Hardware Interface
# ----------------------------
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()
kinematics = IRISKinematics()

def read_motor(motor_id):
    """Read motor state passively (0 gains)."""
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    
    # Send 0 command to keep it passive/safe
    cmd.q = data.q
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

def read_all_motors():
    return [read_motor(mid) for mid in MOTOR_IDS]

def actuate_motor(motor_id, target_q):
    """Command motor with active impedance."""
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = target_q
    cmd.dq = 0.0
    cmd.kp = IMPEDANCE_SETTINGS[motor_id]["kp"]
    cmd.kd = IMPEDANCE_SETTINGS[motor_id]["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)

# ----------------------------
# Motion Control
# ----------------------------
def move_to_start_smooth(target_pos, duration=3.0):
    """Smooth sine-wave interpolation to start position to prevent jerks."""
    print(f"Moving to start configuration ({duration}s)...")
    start_pos = read_all_motors()
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > duration:
            break
        
        # Smoothstep (S-curve) interpolation
        alpha = elapsed / duration
        alpha_smooth = 0.5 * (1 - math.cos(math.pi * alpha))
        
        for i, mid in enumerate(MOTOR_IDS):
            current_cmd = start_pos[i] + (target_pos[i] - start_pos[i]) * alpha_smooth
            actuate_motor(mid, current_cmd)
        time.sleep(0.01) # 100Hz control loop

    # Final hold to ensure exact start
    for i, mid in enumerate(MOTOR_IDS):
        actuate_motor(mid, target_pos[i])

def playback_trajectory(trajectory):
    """Replays recorded trajectory data."""
    if not trajectory:
        return
    
    t0 = trajectory[0][0] # First timestamp
    start_real = time.time()
    
    for row in trajectory:
        target_time = row[0] - t0
        target_joints = row[1:]
        
        # Wait for the correct time slot
        while (time.time() - start_real) < target_time:
            time.sleep(0.001)
            
        for i, mid in enumerate(MOTOR_IDS):
            actuate_motor(mid, target_joints[i])

# ----------------------------
# Visualization & Analysis
# ----------------------------
def visualize_results(points):
    """Generates 3D scatter plot of end-effector positions."""
    if not points:
        print("No data points to plot.")
        return

    pts = np.array(points) * 1000.0 # Convert m to mm for easier reading
    
    # Calculate Statistics
    mean_pos = np.mean(pts, axis=0)
    distances = np.linalg.norm(pts - mean_pos, axis=1)
    repeatability = np.mean(distances) # Mean radius deviation
    max_dev = np.max(distances)

    print("\n" + "="*40)
    print("      STATISTICAL ANALYSIS (mm)")
    print("="*40)
    print(f" Samples:       {len(pts)}")
    print(f" Mean Position: X={mean_pos[0]:.2f}, Y={mean_pos[1]:.2f}, Z={mean_pos[2]:.2f}")
    print(f" Repeatability: {repeatability:.4f} mm (avg deviation)")
    print(f" Max Deviation: {max_dev:.4f} mm")
    print("="*40)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='r', marker='o', s=60, label='Endpoints')
    ax.scatter(mean_pos[0], mean_pos[1], mean_pos[2], c='b', marker='x', s=100, label='Mean')
    
    # Labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f"End-Effector Repeatability\n(Avg Dev: {repeatability:.3f}mm)")
    ax.legend()
    plt.show()

# ----------------------------
# Main Routine
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="IRIS Robot Repeatability Test")
    parser.add_argument('--demo', type=str, required=True, help="Path to demonstration CSV file")
    parser.add_argument('--num', type=int, default=10, help="Number of repetitions (default: 10)")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading demonstration: {args.demo}")
    traj_data = []
    try:
        with open(args.demo, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                # Format: [timestamp, m0, m1, m2, m3, m4, m5]
                traj_data.append([float(x) for x in row])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if not traj_data:
        print("CSV file is empty.")
        sys.exit(1)

    start_pose = traj_data[0][1:]
    end_pose = traj_data[-1][1:]
    
    measured_positions = []

    print(f"\nStarting Test: {args.num} iterations")
    print("-----------------------------------")

    try:
        # Initialize motors first
        print("Initializing motors...")
        read_all_motors()

        for i in range(args.num):
            print(f"\n[Iteration {i+1}/{args.num}]")
            
            # A. Move to Start
            move_to_start_smooth(start_pose, duration=4.0)
            time.sleep(0.5) # Short settle at start

            # B. Playback
            print("  Replaying trajectory...")
            playback_trajectory(traj_data)

            # C. Hold & Measure
            print("  Holding target for 5 seconds...")
            hold_start = time.time()
            while (time.time() - hold_start) < 5.0:
                # Actively hold the final pose
                for idx, mid in enumerate(MOTOR_IDS):
                    actuate_motor(mid, end_pose[idx])
                time.sleep(0.01)
            
            # D. Record
            current_q = read_all_motors()
            ee_pos = kinematics.forward_kinematics(current_q)
            measured_positions.append(ee_pos)
            print(f"  -> Recorded EE: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

        # Save Raw Results
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", f"repeatability_{int(time.time())}.csv")
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_m", "y_m", "z_m"])
            writer.writerows(measured_positions)
        print(f"\nRaw data saved to: {save_path}")

        # Visualize
        visualize_results(measured_positions)

    except KeyboardInterrupt:
        print("\nTest interrupted by user. Safely exiting.")

if __name__ == "__main__":
    main()