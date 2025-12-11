#!/usr/bin/env python
import time
import math
import sys
import argparse
import csv

sys.path.append('../lib')
from unitree_actuator_sdk import *

# ----------------------------
# Robot and Communication Setup
# ----------------------------
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Get gear ratio for position conversion
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)

# Motor IDs for the 6 joints
motor_ids = [0, 1, 2, 3, 4, 5]

# Joint names for display
joint_names = [
    "Base (J0)",
    "Shoulder (J1)", 
    "Elbow (J2)",
    "Wrist Roll (J3)",
    "Wrist Pitch (J4)",
    "Wrist Yaw (J5)"
]

# ----------------------------
# DH Parameters (Standard DH Convention)
# Frame assignments:
#   Frame 0: Base (ground)
#   Frame 1: Shoulder joint (after 238.1mm base height)
#   Frame 2: Elbow joint (after 208.923mm link)
#   Frame 3: Wrist center (after 258.5mm forearm)
#   Frames 4,5,6: Spherical wrist (3 axes intersecting)
#   End-effector: 70mm from wrist center
# ----------------------------
dh_params = {
    'a':     [0,        208.923,  258.5,   0,         0,         0],      # Link lengths (mm)
    'd':     [238.1,    0,        0,       0,         0,         70],     # Offset along Z (mm)
    'alpha': [math.pi/2, 0,       0,       math.pi/2, math.pi/2, math.pi/2]  # Link twists (rad)
}

# DH Table Breakdown:
# Joint 0 (Base): Rotates about vertical Z0
#   - d0 = 238.1mm: Base height to shoulder
#   - alpha0 = π/2: Z0 perpendicular to Z1
#
# Joint 1 (Shoulder): Pitch motion
#   - a1 = 208.923mm: Shoulder to elbow distance
#   - alpha1 = 0: Parallel rotation axes
#
# Joint 2 (Elbow): Pitch motion
#   - a2 = 258.5mm: Elbow to wrist distance
#   - alpha2 = 0: Parallel rotation axes
#
# Joint 3 (Wrist Roll): First wrist axis
#   - alpha3 = π/2: Perpendicular to create spherical wrist
#
# Joint 4 (Wrist Pitch): Second wrist axis
#   - alpha4 = π/2: Spherical wrist configuration
#
# Joint 5 (Wrist Yaw): Third wrist axis + end-effector
#   - d5 = 70mm: End-effector extension
#   - alpha5 = π/2: Complete spherical wrist

# ----------------------------
# Command Line Arguments
# ----------------------------
def parse_arguments():
    """Parse command line arguments for home position configuration."""
    parser = argparse.ArgumentParser(description='Robot Arm Forward Kinematics Monitor')
    parser.add_argument('--home', type=str, default='home_position.csv',
                        help='Path to home position CSV file (default: home_position.csv)')
    return parser.parse_args()

# ----------------------------
# Load Home Position
# ----------------------------
def load_home_position(filename):
    """
    Loads the home position from a CSV file.
    Returns a list of 6 joint angles in radians.
    """
    try:
        home_pos = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                home_pos.append(float(row['position_rad']))
        
        if len(home_pos) != 6:
            print(f"Warning: Expected 6 joint positions, got {len(home_pos)}")
            return None
        
        print(f"✓ Home position loaded from: {filename}")
        print("\nHome joint angles (absolute encoder values when arm points up):")
        for i, pos in enumerate(home_pos):
            print(f"  {joint_names[i]:<20} {pos:>10.4f} rad  ({math.degrees(pos):>10.2f}°)")
        return home_pos
        
    except FileNotFoundError:
        print(f"Error: Home position file '{filename}' not found!")
        print("Please run calibration.py first to create the home position file.")
        return None
    except Exception as e:
        print(f"Error loading home position: {e}")
        return None

# ----------------------------
# Zero Torque Read Function
# ----------------------------
def read_motor_passive(motor_id):
    """
    Reads motor position with zero torque applied.
    Returns position divided by gear ratio for actual joint angle.
    """
    # Read current sensor value
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    current_position = data.q
    
    # Re-command with zero gains (no torque)
    cmd.q = current_position
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    
    # Divide by gear ratio to get actual joint angle
    joint_position = data.q / gear_ratio
    
    return joint_position

def read_all_motors_passive():
    """
    Reads all motor positions with zero torque.
    Returns list of joint angles in radians.
    """
    joint_angles = []
    for motor_id in motor_ids:
        angle = read_motor_passive(motor_id)
        joint_angles.append(angle)
    return joint_angles

# ----------------------------
# Matrix Operations (Simple implementation without numpy)
# ----------------------------
def matrix_multiply_4x4(A, B):
    """Multiply two 4x4 matrices."""
    result = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += A[i][k] * B[k][j]
    return result

def identity_matrix_4x4():
    """Create 4x4 identity matrix."""
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

# ----------------------------
# Forward Kinematics Functions
# ----------------------------
def dh_transform(a, alpha, d, theta):
    """
    Computes the homogeneous transformation matrix using Standard DH Convention.
    
    Standard DH Convention (Craig's notation):
    T(i-1,i) = Rot(Z, theta) * Trans(Z, d) * Trans(X, a) * Rot(X, alpha)
    
    Parameters:
    - theta: joint angle (radians) - rotation about Z(i-1)
    - d: link offset (mm) - translation along Z(i-1)
    - a: link length (mm) - translation along X(i)
    - alpha: link twist (radians) - rotation about X(i)
    
    Returns: 4x4 transformation matrix (as list of lists)
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    
    # Standard DH transformation matrix
    return [
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1]
    ]

def forward_kinematics(joint_angles, dh_params, home_offsets=None):
    """
    Computes forward kinematics using DH parameters.
    
    Parameters:
    - joint_angles: list of 6 joint angles (radians, absolute actuator positions)
    - dh_params: dictionary with 'a', 'd', 'alpha' parameters
    - home_offsets: list of 6 home position angles (radians) where robot points up
    
    Returns:
    - transforms: list of 4x4 transformation matrices for each link
    - T_final: final end-effector transformation matrix
    """
    transforms = []
    T = identity_matrix_4x4()  # Start with identity (world frame)
    
    for i in range(6):
        a = dh_params['a'][i]
        d = dh_params['d'][i]
        alpha = dh_params['alpha'][i]
        
        # Calculate relative joint angle from home position
        # When actuator is at home_offset, the joint angle for FK should be 0
        if home_offsets is not None:
            theta = joint_angles[i] - home_offsets[i]
        else:
            theta = joint_angles[i]
        
        # Compute DH transformation for this joint
        T_i = dh_transform(a, alpha, d, theta)
        T = matrix_multiply_4x4(T, T_i)
        
        # Store a copy of the current transformation
        transforms.append([row[:] for row in T])
    
    return transforms, T

def extract_position(T):
    """Extract (x, y, z) position from 4x4 transformation matrix."""
    return [T[0][3], T[1][3], T[2][3]]

def extract_orientation(T):
    """
    Extracts the end-effector orientation as rotation axes.
    Returns x, y, z unit vectors representing end-effector frame.
    """
    x_axis = [T[0][0], T[1][0], T[2][0]]
    y_axis = [T[0][1], T[1][1], T[2][1]]
    z_axis = [T[0][2], T[1][2], T[2][2]]
    return x_axis, y_axis, z_axis

def vector_magnitude(v):
    """Calculate magnitude of a 3D vector."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

# ----------------------------
# Display Functions
# ----------------------------
def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end='')

def display_robot_state(joint_angles, transforms, T_final, home_position):
    """Display complete robot state in terminal."""
    clear_screen()
    
    print("=" * 80)
    print("  6-DOF ROBOT ARM - FORWARD KINEMATICS MONITOR (Zero Torque Mode)")
    print("=" * 80)
    
    # Joint angles - show both absolute and relative
    print("\n" + "─" * 80)
    print("  JOINT ANGLES")
    print("─" * 80)
    print(f"{'Joint':<20} {'Absolute (rad)':<16} {'Absolute (deg)':<16} {'Relative (deg)':<16}")
    print("─" * 80)
    for i, angle in enumerate(joint_angles):
        abs_deg = math.degrees(angle)
        rel_deg = math.degrees(angle - home_position[i]) if home_position else abs_deg
        print(f"{joint_names[i]:<20} {angle:>10.4f}       {abs_deg:>10.2f}       {rel_deg:>10.2f}")
    
    # Link positions
    print("\n" + "─" * 80)
    print("  LINK POSITIONS (World Frame, mm)")
    print("─" * 80)
    print(f"{'Link':<20} {'X':<15} {'Y':<15} {'Z':<15}")
    print("─" * 80)
    
    # Base
    print(f"{'Base':<20} {0.0:>10.2f}      {0.0:>10.2f}      {0.0:>10.2f}")
    
    # Each joint
    for i, T in enumerate(transforms):
        pos = extract_position(T)
        print(f"{joint_names[i]:<20} {pos[0]:>10.2f}      {pos[1]:>10.2f}      {pos[2]:>10.2f}")
    
    # End-effector details
    ee_pos = extract_position(T_final)
    x_axis, y_axis, z_axis = extract_orientation(T_final)
    
    print("\n" + "─" * 80)
    print("  END-EFFECTOR STATE")
    print("─" * 80)
    print(f"Position (mm):  X = {ee_pos[0]:>10.2f}  Y = {ee_pos[1]:>10.2f}  Z = {ee_pos[2]:>10.2f}")
    print(f"\nOrientation (Unit Vectors):")
    print(f"  X-axis: [{x_axis[0]:>7.4f}, {x_axis[1]:>7.4f}, {x_axis[2]:>7.4f}]")
    print(f"  Y-axis: [{y_axis[0]:>7.4f}, {y_axis[1]:>7.4f}, {y_axis[2]:>7.4f}]")
    print(f"  Z-axis: [{z_axis[0]:>7.4f}, {z_axis[1]:>7.4f}, {z_axis[2]:>7.4f}]")
    
    # Calculate reach (distance from base)
    reach = vector_magnitude(ee_pos)
    print(f"\nReach from base: {reach:.2f} mm")
    
    # Expected height at home position
    expected_height = 238.1 + 208.923 + 258.5 + 70
    print(f"Expected height at home (all joints = 0°): {expected_height:.2f} mm")
    
    print("=" * 80)
    print("NOTE: 'Absolute' = raw actuator encoder | 'Relative' = joint angle from home")
    print("At HOME (relative = 0°), robot points straight up along Z-axis")
    print("\nManually move the robot arm - display updates in real-time")
    print("Press Ctrl+C to exit")
    print("=" * 80)

# ----------------------------
# CSV Logging
# ----------------------------
def setup_logging():
    """Setup CSV file for logging end-effector trajectory."""
    csv_file = open("fk_trajectory_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", 
        "j0_rad", "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad",
        "ee_x_mm", "ee_y_mm", "ee_z_mm",
        "reach_mm"
    ])
    return csv_file, csv_writer

def log_state(csv_writer, joint_angles, ee_pos):
    """Log current state to CSV."""
    reach = vector_magnitude(ee_pos)
    csv_writer.writerow([
        time.time(),
        *joint_angles,
        ee_pos[0], ee_pos[1], ee_pos[2],
        reach
    ])

# ----------------------------
# Main Routine
# ----------------------------
if __name__ == '__main__':
    csv_file = None
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        print("\n" + "=" * 80)
        print("  ROBOT ARM FORWARD KINEMATICS MONITOR")
        print("=" * 80)
        
        # Load home position
        home_position = load_home_position(args.home)
        if home_position is None:
            print("\nContinuing without home position reference...")
            home_position = [0.0] * 6
        
        print("\nInitializing motors in ZERO TORQUE mode...")
        
        # Initialize all motors in passive mode
        for motor_id in motor_ids:
            read_motor_passive(motor_id)
        
        print("✓ Motors initialized successfully")
        print("✓ Zero torque mode active - arm is free to move manually")
        
        # Setup logging
        csv_file, csv_writer = setup_logging()
        print("✓ Logging trajectory to: fk_trajectory_log.csv")
        
        input("\nPress ENTER to start real-time monitoring...")
        
        # Main monitoring loop
        last_log_time = time.time()
        
        while True:
            # Read current joint angles (zero torque)
            joint_angles = read_all_motors_passive()
            
            # Compute forward kinematics with home position offsets
            transforms, T_final = forward_kinematics(joint_angles, dh_params, home_position)
            
            # Get end-effector position
            ee_pos = extract_position(T_final)
            
            # Update terminal display
            display_robot_state(joint_angles, transforms, T_final, home_position)
            
            # Log every 0.2 seconds
            current_time = time.time()
            if current_time - last_log_time >= 0.2:
                log_state(csv_writer, joint_angles, ee_pos)
                csv_file.flush()
                last_log_time = current_time
            
            # Update at ~5 Hz
            time.sleep(0.2)
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Exiting gracefully...\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if csv_file is not None:
            csv_file.close()
            print("✓ Trajectory log saved")
        sys.exit(0)