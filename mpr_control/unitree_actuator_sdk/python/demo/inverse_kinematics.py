import time
import math
import sys
import threading
import numpy as np
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# -----------------------------
# Robot and DH Parameter Setup
# -----------------------------
# Serial connection and command/data objects for the motors
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# DH parameters for the robot (units: mm for distances, radians for angles)
dh_params = {
    'a': [238.1, 19.65, 208.923, 0, 258.5, 0],
    'd': [0, 56.25, 39.875, 66.032, 0, 0],
    'alpha': [0, 0, 0, math.pi/2, math.pi/2, math.pi/2]
}

# Motor (joint) IDs for a 6-DOF robot arm
motor_ids = [0, 1, 2, 3, 4, 5]

# Impedance control gains (PD gains) for each joint (different values per joint)
impedance_settings = {
    0: {"kp": 0.2, "kd": 0.05},
    1: {"kp": 0.3, "kd": 0.06},
    2: {"kp": 0.25, "kd": 0.04},
    3: {"kp": 0.22, "kd": 0.07},
    4: {"kp": 0.18, "kd": 0.05},
    5: {"kp": 0.18, "kd": 0.05},
}

# -----------------------------
# Query Current Joint Angles
# -----------------------------
# Instead of using hard-coded angles, query each motor to get its current position.
current_joint_angles = []
for motor_id in motor_ids:
    cmd.id = motor_id
    # Set dummy command values (no motion) just to trigger a reading.
    cmd.q = 0.0
    cmd.dq = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    current_joint_angles.append(data.q)
print("Initial joint positions (radians):", current_joint_angles)

# -----------------------------
# Coordinate Frame Definitions
# -----------------------------
def dh_transformation(theta, a, d, alpha):
    """Compute the individual DH transformation matrix."""
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0,  sa,       ca,       d],
        [0,  0,        0,        1]
    ])

def forward_kinematics(joint_angles):
    """
    Compute the end-effector pose [x, y, z, theta] using the DH parameters.
    x, y, z are in mm; theta is an approximate yaw (rotation about Z) in radians.
    """
    T = np.eye(4)
    for i in range(6):
        theta_i = joint_angles[i]
        a = dh_params['a'][i]
        d = dh_params['d'][i]
        alpha = dh_params['alpha'][i]
        T = np.dot(T, dh_transformation(theta_i, a, d, alpha))
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    theta_ee = math.atan2(T[1, 0], T[0, 0])
    return np.array([x, y, z, theta_ee])

# Define the world origin as the end-effector pose when all joint angles are zero.
world_origin = forward_kinematics(np.zeros(6))
# The starting pose (local origin) is computed from the current joint angles (queried above).
starting_pose = forward_kinematics(np.array(current_joint_angles))

# -----------------------------
# End-Effector Desired Pose Setup (Local Coordinates)
# -----------------------------
# In the local frame, the origin is the starting_pose.
# The local desired offset (in mm and radians) starts at zero.
local_desired = {
    'x': 0.0,     # offset from starting_pose x
    'y': 0.0,     # offset from starting_pose y
    'z': 0.0,     # offset from starting_pose z
    'theta': 0.0  # offset from starting_pose yaw
}

# Movement step sizes
step_xy = 10.0             # mm for X-Y movement (arrow keys)
step_z = 10.0              # mm for Z movement (W/S keys)
step_theta = math.radians(5)  # 5° for orientation adjustments (A/D keys)

# -----------------------------
# Keyboard Input Handling Setup
# -----------------------------
key_flags = {
    'up': False,
    'down': False,
    'left': False,
    'right': False,
    'w': False,
    's': False,
    'a': False,
    'd': False,
}
key_flags_lock = threading.Lock()

def process_keyboard_input():
    """Thread to handle keyboard input events for end-effector control."""
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
                elif key.char.lower() == 'w':
                    key_flags['w'] = True
                elif key.char.lower() == 's':
                    key_flags['s'] = True
                elif key.char.lower() == 'a':
                    key_flags['a'] = True
                elif key.char.lower() == 'd':
                    key_flags['d'] = True
            except AttributeError:
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
                elif key.char.lower() == 'w':
                    key_flags['w'] = False
                elif key.char.lower() == 's':
                    key_flags['s'] = False
                elif key.char.lower() == 'a':
                    key_flags['a'] = False
                elif key.char.lower() == 'd':
                    key_flags['d'] = False
            except AttributeError:
                pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

# Start the keyboard listener thread
keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
keyboard_thread.start()

# -----------------------------
# Analytical Inverse Kinematics Function
# -----------------------------
def analytical_inverse_kinematics(desired, current_angles):
    """
    Compute an analytical IK solution for a 6-DOF arm.
    
    The desired pose is given as [x, y, z, theta] in mm and radians.
    
    This simplified solution assumes:
      - Joint 0 rotates about the Z-axis (theta1 = atan2(y, x)).
      - The wrist center coincides with the desired position.
      - The first three joints determine the end-effector position.
      - The wrist joints (joints 4–6) adjust the overall yaw.
    
    For demonstration, we compute:
      theta1 = atan2(y, x)
      theta2 = (approximate planar solution based on the effective arm links)
      theta3 = set to zero
      theta4 = desired_yaw - (theta1 + theta2)
      theta5, theta6 = 0
    Note: This is a very rough decoupled solution and may require refinement for your specific robot.
    """
    x, y, z, theta = desired
    # Compute base rotation
    theta1 = math.atan2(y, x)
    
    # Use the world origin (q = [0,...,0]) to get the base height.
    base_pose = forward_kinematics(np.zeros(6))
    base_z = base_pose[2]
    
    # Compute horizontal distance and vertical displacement
    r = math.sqrt(x*x + y*y)
    s = z - base_z
    
    # Use effective link lengths from the first two joints.
    L1 = dh_params['a'][0]      # e.g., 238.1 mm
    L2 = dh_params['a'][2]      # e.g., 208.923 mm
    
    # Compute distance from the effective joint 1 (assumed at L1 along X) to the desired position.
    D = math.sqrt((r - L1)**2 + s**2)
    if D > L2:
        D = L2
    
    # Compute theta2 using a simple planar two-link solution.
    try:
        angle_offset = math.acos((L2**2 + (r - L1)**2 + s**2) / (2 * L2 * math.sqrt((r - L1)**2 + s**2)))
    except ValueError:
        angle_offset = 0.0
    theta2 = math.atan2(s, r - L1) - angle_offset
    
    # Set theta3 = 0 (i.e., no additional bending in the position solution)
    theta3 = 0.0
    
    # For the wrist, assume that the overall end-effector yaw should equal theta.
    phi = theta1 + theta2 + theta3
    theta4 = theta - phi
    theta5 = 0.0
    theta6 = 0.0
    
    return [theta1, theta2, theta3, theta4, theta5, theta6]

# -----------------------------
# Motor Control Function (Impedance Control)
# -----------------------------
def control_motor(motor_id, position):
    """
    Send an impedance control (PD) command to a specific motor.
    Uses individual kp and kd values for each joint.
    """
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = position
    cmd.dq = 0.0
    cmd.kp = impedance_settings[motor_id]["kp"]
    cmd.kd = impedance_settings[motor_id]["kd"]
    cmd.tau = 0.0  # No additional feedforward torque
    serial.sendRecv(cmd, data)
    return data.q  # Return the actual position reading

# -----------------------------
# Main Control Loop
# -----------------------------
print("Control the end-effector using:")
print("  Arrow keys: move in the X-Y plane (local)")
print("  W/S: move end-effector up/down (local Z)")
print("  A/D: rotate end-effector (adjust local theta)")
print("Local origin = starting pose, world origin = pose at q=[0,...,0]")
print("Press Ctrl+C to exit.")

last_print_time = time.time()
try:
    while True:
        # Update the local desired pose based on keyboard input
        with key_flags_lock:
            if key_flags['up']:
                local_desired['x'] += step_xy
            if key_flags['down']:
                local_desired['x'] -= step_xy
            if key_flags['right']:
                local_desired['y'] += step_xy
            if key_flags['left']:
                local_desired['y'] -= step_xy
            if key_flags['w']:
                local_desired['z'] += step_z
            if key_flags['s']:
                local_desired['z'] -= step_z
            if key_flags['d']:
                local_desired['theta'] += step_theta
            if key_flags['a']:
                local_desired['theta'] -= step_theta

        # Convert local desired pose into world coordinates.
        # The world desired pose is the starting_pose (local origin) plus the local offset.
        desired_vector = np.array([
            starting_pose[0] + local_desired['x'],
            starting_pose[1] + local_desired['y'],
            starting_pose[2] + local_desired['z'],
            starting_pose[3] + local_desired['theta']
        ])

        # Compute new joint angles via the analytical IK solver.
        new_joint_angles = analytical_inverse_kinematics(desired_vector, current_joint_angles)

        # Command each motor using the computed joint angles (impedance control)
        for i, motor_id in enumerate(motor_ids):
            current_joint_angles[i] = control_motor(motor_id, new_joint_angles[i])
            time.sleep(0.01)  # Small delay between commands

        # Compute the current end-effector pose in world coordinates.
        current_pose = forward_kinematics(np.array(current_joint_angles))

        # Print a summary every 0.1 seconds.
        current_time = time.time()
        if current_time - last_print_time >= 0.1:
            print("\n--- End-Effector Status ---")
            print(f"Local Desired Offset: x={local_desired['x']:.1f} mm, y={local_desired['y']:.1f} mm, "
                  f"z={local_desired['z']:.1f} mm, theta={math.degrees(local_desired['theta']):.1f}°")
            print(f"World Desired Pose: x={desired_vector[0]:.1f} mm, y={desired_vector[1]:.1f} mm, "
                  f"z={desired_vector[2]:.1f} mm, theta={math.degrees(desired_vector[3]):.1f}°")
            print(f"Current Pose: x={current_pose[0]:.1f} mm, y={current_pose[1]:.1f} mm, "
                  f"z={current_pose[2]:.1f} mm, theta={math.degrees(current_pose[3]):.1f}°")
            error = desired_vector - current_pose
            print(f"Pose Error: dx={error[0]:.1f} mm, dy={error[1]:.1f} mm, dz={error[2]:.1f} mm, "
                  f"dtheta={math.degrees(error[3]):.1f}°")
            last_print_time = current_time

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")

finally:
    # Safely stop all motors by holding their current positions.
    for i, motor_id in enumerate(motor_ids):
        cmd.id = motor_id
        cmd.q = current_joint_angles[i]
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
    print("All motors stopped.")
