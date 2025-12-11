#!/usr/bin/env python
import time
import math
import sys
import argparse
import csv
import threading
from pynput import keyboard

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

# Define impedance settings for each joint
impedance_settings = {
    0: {"kp": 0.5, "kd": 0.05},
    1: {"kp": 2.5, "kd": 0.15},
    2: {"kp": 1.0, "kd": 0.15},
    3: {"kp": 0.3, "kd": 0.025},
    4: {"kp": 0.3, "kd": 0.025},
    5: {"kp": 0.3, "kd": 0.025},
}

# ----------------------------
# DH Parameters (Standard DH Convention)
# ----------------------------
dh_params = {
    'a':     [0,        208.923,  258.5,   0,         0,         0],
    'd':     [238.1,    0,        0,       0,         0,         70],
    'alpha': [math.pi/2, 0,       0,       math.pi/2, math.pi/2, math.pi/2]
}

# Robot dimensions for IK
L1 = 238.1   # Base height
L2 = 208.923 # Shoulder to elbow
L3 = 258.5   # Elbow to wrist
L4 = 70      # Wrist to end-effector

# Joint limits (relative to home position, in radians)
JOINT_LIMITS = {
    0: {'min': math.radians(-180), 'max': math.radians(180)},  # Base: ±180°
    1: {'min': math.radians(-180), 'max': math.radians(180)},  # Shoulder: ±180°
    2: {'min': math.radians(-180), 'max': math.radians(180)},  # Elbow: ±180°
    3: {'min': math.radians(-180), 'max': math.radians(180)},  # Wrist Roll: ±180°
    4: {'min': math.radians(-180), 'max': math.radians(180)},  # Wrist Pitch: ±180°
    5: {'min': math.radians(-180), 'max': math.radians(180)},  # Wrist Yaw: ±180°
}

# ----------------------------
# Teleoperation Parameters
# ----------------------------
# Step size for end-effector movements (mm)
position_step_xy = 5.0  # mm per key press for X/Y
position_step_z = 5.0   # mm per key press for Z

# Maximum joint velocity (rad/s) for smooth motion
MAX_JOINT_STEP = 0.1  # radians per control cycle (~5.7 degrees at 20Hz)

# Key state tracking
key_flags = {
    'up': False,
    'down': False,
    'left': False,
    'right': False,
    'w': False,
    's': False,
}
key_flags_lock = threading.Lock()

# Target end-effector position (will be initialized from current pose)
target_ee_pos = [0.0, 0.0, 0.0]
target_ee_pos_lock = threading.Lock()

# Desired joint angles (for logging)
desired_joint_angles = [0.0] * 6
desired_joint_angles_lock = threading.Lock()

# ----------------------------
# Command Line Arguments
# ----------------------------
def parse_arguments():
    """Parse command line arguments for home position configuration."""
    parser = argparse.ArgumentParser(description='Robot Arm Inverse Kinematics Teleoperation')
    parser.add_argument('--home', type=str, default='home_position.csv',
                        help='Path to home position CSV file (default: home_position.csv)')
    return parser.parse_args()

# ----------------------------
# Load Home Position
# ----------------------------
def load_home_position(filename):
    """Loads the home position from a CSV file."""
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
        return home_pos
        
    except FileNotFoundError:
        print(f"Error: Home position file '{filename}' not found!")
        print("Please run calibration.py first to create the home position file.")
        return None
    except Exception as e:
        print(f"Error loading home position: {e}")
        return None

# ----------------------------
# Motor Control Functions
# ----------------------------
def read_motor_passive(motor_id):
    """
    Read motor position with zero torque applied.
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

def read_motor_position(motor_id):
    """Read motor position in passive mode (for initialization)."""
    return read_motor_passive(motor_id)

def control_motor(motor_id, desired_angle):
    """
    Send control command to motor with impedance control.
    Returns actual angle read from motor.
    """
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = desired_angle * gear_ratio
    cmd.dq = 0.0
    cmd.kp = impedance_settings[motor_id]["kp"]
    cmd.kd = impedance_settings[motor_id]["kd"]
    cmd.tau = 0.0
    
    serial.sendRecv(cmd, data)
    
    return data.q / gear_ratio

def control_motor_safe(motor_id, desired_angle, current_angle, max_step=0.1):
    """
    Motor control with rate limiting to prevent sudden movements.
    
    Parameters:
    - motor_id: ID of the motor
    - desired_angle: target angle (radians)
    - current_angle: current angle (radians)
    - max_step: maximum angle change per call (radians)
    
    Returns: actual angle
    """
    # Limit the step size
    angle_error = desired_angle - current_angle
    if abs(angle_error) > max_step:
        desired_angle = current_angle + math.copysign(max_step, angle_error)
    
    return control_motor(motor_id, desired_angle)

def get_current_joint_angles():
    """Read current joint angles from all motors in passive mode."""
    angles = []
    for motor_id in motor_ids:
        angle = read_motor_passive(motor_id)
        angles.append(angle)
    return angles

# ----------------------------
# Forward Kinematics (for initialization)
# ----------------------------
def forward_kinematics_position(joint_angles, home_offsets):
    """
    Compute end-effector position using forward kinematics.
    Returns [x, y, z] position in mm.
    """
    # Calculate relative angles from home
    theta = [joint_angles[i] - home_offsets[i] for i in range(6)]
    
    # Simplified FK for position (first 3 joints dominate position)
    # Joint 0: Base rotation
    # Joint 1: Shoulder pitch
    # Joint 2: Elbow pitch
    
    # Base rotation
    c0 = math.cos(theta[0])
    s0 = math.sin(theta[0])
    
    # Shoulder and elbow angles
    q1 = theta[1]
    q2 = theta[2]
    
    # Position in arm plane
    r = L2 * math.cos(q1) + L3 * math.cos(q1 + q2) + L4 * math.cos(q1 + q2)
    z = L1 + L2 * math.sin(q1) + L3 * math.sin(q1 + q2) + L4 * math.sin(q1 + q2)
    
    # Project to world coordinates
    x = r * c0
    y = r * s0
    
    return [x, y, z]

# ----------------------------
# Inverse Kinematics (Optimized)
# ----------------------------
def inverse_kinematics_fast(target_pos, home_offsets, current_angles):
    """
    Fast analytical inverse kinematics for target end-effector position.
    Optimized version with minimal checks for real-time control.
    
    Parameters:
    - target_pos: [x, y, z] desired end-effector position (mm)
    - home_offsets: home position offsets
    - current_angles: current joint angles for wrist orientation reference
    
    Returns:
    - joint_angles: [θ0, θ1, θ2, θ3, θ4, θ5] in radians (absolute)
    - success: True if solution found
    """
    x, y, z = target_pos
    
    # Joint 0: Base rotation (simple atan2) - fastest
    theta0 = math.atan2(y, x)
    
    # Quick limit check for base
    theta0_rel = theta0
    if theta0_rel < -math.pi:
        theta0_rel += 2 * math.pi
    elif theta0_rel > math.pi:
        theta0_rel -= 2 * math.pi
    theta0 = theta0_rel
    
    # Calculate planar distance and adjusted height
    r = math.sqrt(x*x + y*y)
    z_adj = z - L1
    
    # Distance from shoulder to target
    d_sq = r*r + z_adj*z_adj
    d = math.sqrt(d_sq)
    
    # Quick reachability check
    max_reach = L2 + L3 + L4
    if d > max_reach * 1.01:  # Small margin
        return None, False
    
    # Wrist distance (simplified - treat as fixed offset)
    wrist_dist = d - L4
    if wrist_dist < 10:  # Minimum valid distance
        wrist_dist = 10
    
    # Elbow angle - law of cosines (optimized)
    wrist_dist_sq = wrist_dist * wrist_dist
    L2_sq = L2 * L2
    L3_sq = L3 * L3
    
    cos_q2 = (wrist_dist_sq - L2_sq - L3_sq) / (2.0 * L2 * L3)
    
    # Clamp and compute (branchless where possible)
    cos_q2 = max(-0.999, min(0.999, cos_q2))
    q2 = math.acos(cos_q2)
    
    # Shoulder angle - law of cosines
    alpha = math.atan2(z_adj, r)
    cos_beta = (L2_sq + wrist_dist_sq - L3_sq) / (2.0 * L2 * wrist_dist)
    cos_beta = max(-0.999, min(0.999, cos_beta))
    beta = math.acos(cos_beta)
    q1 = alpha - beta
    
    # Keep wrist joints at current values (fast - no computation)
    current_rel = [current_angles[i] - home_offsets[i] for i in range(3, 6)]
    
    # Assemble solution (direct assignment)
    joint_angles = [
        theta0 + home_offsets[0],
        q1 + home_offsets[1],
        q2 + home_offsets[2],
        current_rel[0] + home_offsets[3],
        current_rel[1] + home_offsets[4],
        current_rel[2] + home_offsets[5]
    ]
    
    return joint_angles, True

# Wrapper with caching for redundant calls
_last_target = [0.0, 0.0, 0.0]
_last_solution = None
_cache_tolerance = 0.1  # mm

def inverse_kinematics(target_pos, home_offsets, current_angles):
    """
    Cached inverse kinematics wrapper for performance.
    Returns cached solution if target hasn't changed significantly.
    """
    global _last_target, _last_solution
    
    # Check if target changed significantly
    dx = abs(target_pos[0] - _last_target[0])
    dy = abs(target_pos[1] - _last_target[1])
    dz = abs(target_pos[2] - _last_target[2])
    
    if dx < _cache_tolerance and dy < _cache_tolerance and dz < _cache_tolerance and _last_solution is not None:
        return _last_solution, True
    
    # Compute new solution
    solution, success = inverse_kinematics_fast(target_pos, home_offsets, current_angles)
    
    if success:
        # Update cache
        _last_target[0] = target_pos[0]
        _last_target[1] = target_pos[1]
        _last_target[2] = target_pos[2]
        _last_solution = solution
    
    return solution, success

# ----------------------------
# Keyboard Input Handler
# ----------------------------
def process_keyboard_input():
    """Handles keyboard events for teleoperation."""
    def on_press(key):
        with key_flags_lock:
            try:
                if key == keyboard.Key.up:
                    key_flags['up'] = True
                elif key == keyboard.Key.down:
                    key_flags['down'] = True
                elif key == keyboard.Key.left:
                    key_flags['left'] = True
                elif key == keyboard.Key.right:
                    key_flags['right'] = True
                elif hasattr(key, 'char'):
                    if key.char == 'w':
                        key_flags['w'] = True
                    elif key.char == 's':
                        key_flags['s'] = True
            except:
                pass

    def on_release(key):
        with key_flags_lock:
            try:
                if key == keyboard.Key.up:
                    key_flags['up'] = False
                elif key == keyboard.Key.down:
                    key_flags['down'] = False
                elif key == keyboard.Key.left:
                    key_flags['left'] = False
                elif key == keyboard.Key.right:
                    key_flags['right'] = False
                elif hasattr(key, 'char'):
                    if key.char == 'w':
                        key_flags['w'] = False
                    elif key.char == 's':
                        key_flags['s'] = False
            except:
                pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

# ----------------------------
# Display Functions
# ----------------------------
def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end='')

def display_status(current_angles, desired_angles, target_pos, home_offsets):
    """Display current status."""
    clear_screen()
    
    print("=" * 80)
    print("  6-DOF ROBOT ARM - INVERSE KINEMATICS TELEOPERATION")
    print("=" * 80)
    
    print("\n" + "─" * 80)
    print("  JOINT LIMITS: All joints ±180° from home position")
    print("─" * 80)
    
    print("\n" + "─" * 80)
    print("  CONTROLS")
    print("─" * 80)
    print("  Arrow Keys: Move in X-Y plane")
    print("    ↑ : +Y direction")
    print("    ↓ : -Y direction")
    print("    ← : -X direction")
    print("    → : +X direction")
    print("  W/S: Move in Z axis")
    print("    W : +Z (up)")
    print("    S : -Z (down)")
    print("  Ctrl+C: Exit")
    
    print("\n" + "─" * 80)
    print("  TARGET END-EFFECTOR POSITION")
    print("─" * 80)
    print(f"  X = {target_pos[0]:>8.2f} mm")
    print(f"  Y = {target_pos[1]:>8.2f} mm")
    print(f"  Z = {target_pos[2]:>8.2f} mm")
    
    print("\n" + "─" * 80)
    print("  JOINT ANGLES")
    print("─" * 80)
    print(f"{'Joint':<20} {'Desired (deg)':<16} {'Actual (deg)':<16} {'Error (deg)':<16}")
    print("─" * 80)
    for i in range(6):
        des_deg = math.degrees(desired_angles[i])
        act_deg = math.degrees(current_angles[i])
        err_deg = des_deg - act_deg
        
        # Check if at limit
        rel_angle = current_angles[i] - home_offsets[i]
        at_limit = ""
        if abs(rel_angle - JOINT_LIMITS[i]['min']) < 0.01:
            at_limit = " [MIN LIMIT]"
        elif abs(rel_angle - JOINT_LIMITS[i]['max']) < 0.01:
            at_limit = " [MAX LIMIT]"
        
        print(f"{joint_names[i]:<20} {des_deg:>12.2f}    {act_deg:>12.2f}    {err_deg:>12.2f}{at_limit}")
    
    print("=" * 80)

# ----------------------------
# CSV Logging
# ----------------------------
def setup_logging():
    """Setup CSV file for logging teleoperation data."""
    csv_file = open("ik_teleop_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp",
        "target_x_mm", "target_y_mm", "target_z_mm",
        "j0_desired_rad", "j1_desired_rad", "j2_desired_rad", 
        "j3_desired_rad", "j4_desired_rad", "j5_desired_rad",
        "j0_actual_rad", "j1_actual_rad", "j2_actual_rad",
        "j3_actual_rad", "j4_actual_rad", "j5_actual_rad",
        "j0_error_deg", "j1_error_deg", "j2_error_deg",
        "j3_error_deg", "j4_error_deg", "j5_error_deg"
    ])
    return csv_file, csv_writer

def log_state(csv_writer, target_pos, desired_angles, actual_angles):
    """Log current state to CSV."""
    # Calculate errors in degrees
    errors_deg = [math.degrees(desired_angles[i] - actual_angles[i]) for i in range(6)]
    
    csv_writer.writerow([
        time.time(),
        target_pos[0], target_pos[1], target_pos[2],
        *desired_angles,
        *actual_angles,
        *errors_deg
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
        print("  ROBOT ARM INVERSE KINEMATICS TELEOPERATION")
        print("=" * 80)
        
        # Load home position
        home_position = load_home_position(args.home)
        if home_position is None:
            print("\nError: Cannot proceed without home position calibration.")
            sys.exit(1)
        
        print("\nInitializing motors in ZERO TORQUE mode...")
        
        # Initialize all motors in passive mode (zero torque)
        for motor_id in motor_ids:
            read_motor_passive(motor_id)
        
        print("✓ Motors initialized in zero torque mode")
        print("✓ You can manually move the robot arm to desired starting position")
        
        input("\nMove the arm to your starting position, then press ENTER to begin IK control...")
        
        # Read current joint angles after user positions the arm
        current_angles = get_current_joint_angles()
        print("\n✓ Current joint positions captured")
        
        # Initialize desired angles to current
        with desired_joint_angles_lock:
            for i in range(6):
                desired_joint_angles[i] = current_angles[i]
        
        # Compute current end-effector position from manually set pose
        current_ee_pos = forward_kinematics_position(current_angles, home_position)
        
        # Initialize target position to current position
        with target_ee_pos_lock:
            target_ee_pos[0] = current_ee_pos[0]
            target_ee_pos[1] = current_ee_pos[1]
            target_ee_pos[2] = current_ee_pos[2]
        
        print(f"✓ Starting IK from position: X={current_ee_pos[0]:.1f}, Y={current_ee_pos[1]:.1f}, Z={current_ee_pos[2]:.1f} mm")
        
        # Setup logging
        csv_file, csv_writer = setup_logging()
        print("✓ Logging to: ik_teleop_log.csv")
        
        # Start keyboard input thread
        keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
        keyboard_thread.start()
        print("✓ Keyboard control active")
        
        print("\n" + "=" * 80)
        print("  IK CONTROL STARTING NOW")
        print("=" * 80)
        print("Use arrow keys (←→↑↓) for X-Y motion, W/S for Z motion")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")
        
        # Main control loop
        last_log_time = time.time()
        
        while True:
            # Update target position based on key presses
            with key_flags_lock:
                delta_x = 0.0
                delta_y = 0.0
                delta_z = 0.0
                
                if key_flags['right']:
                    delta_x += position_step_xy
                if key_flags['left']:
                    delta_x -= position_step_xy
                if key_flags['up']:
                    delta_y += position_step_xy
                if key_flags['down']:
                    delta_y -= position_step_xy
                if key_flags['w']:
                    delta_z += position_step_z
                if key_flags['s']:
                    delta_z -= position_step_z
            
            # Update target position
            with target_ee_pos_lock:
                target_ee_pos[0] += delta_x
                target_ee_pos[1] += delta_y
                target_ee_pos[2] += delta_z
                current_target = target_ee_pos[:]
            
            # Compute inverse kinematics
            ik_angles, success = inverse_kinematics(current_target, home_position, current_angles)
            
            if success and ik_angles is not None:
                # Update desired angles
                with desired_joint_angles_lock:
                    for i in range(6):
                        desired_joint_angles[i] = ik_angles[i]
                
                # Send commands to motors with rate limiting for safety
                for i, motor_id in enumerate(motor_ids):
                    actual = control_motor_safe(motor_id, ik_angles[i], current_angles[i], MAX_JOINT_STEP)
                    current_angles[i] = actual
            else:
                # If IK fails, hold current position
                with desired_joint_angles_lock:
                    for i in range(6):
                        desired_joint_angles[i] = current_angles[i]
                
                for i, motor_id in enumerate(motor_ids):
                    actual = control_motor(motor_id, current_angles[i])
                    current_angles[i] = actual
            
            # Get current desired angles for display and logging
            with desired_joint_angles_lock:
                current_desired = desired_joint_angles[:]
            
            # Display status
            display_status(current_angles, current_desired, current_target, home_position)
            
            # Log every 0.1 seconds
            current_time = time.time()
            if current_time - last_log_time >= 0.1:
                log_state(csv_writer, current_target, current_desired, current_angles)
                csv_file.flush()
                last_log_time = current_time
            
            # Control loop rate
            time.sleep(0.05)  # 20 Hz
        
    except KeyboardInterrupt:
        print("\n\nTeleoperation stopped by user.")
        print("Holding current position...")
        
        # Hold final position
        for i, motor_id in enumerate(motor_ids):
            control_motor(motor_id, current_angles[i])
        
        print("Exiting gracefully...\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if csv_file is not None:
            csv_file.close()
            print("✓ Log saved to: ik_teleop_log.csv")
        sys.exit(1)