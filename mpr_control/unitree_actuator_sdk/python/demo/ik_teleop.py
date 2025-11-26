import time
import math
import sys
import threading
from pynput import keyboard
sys.path.append('../lib')
import numpy as np
from unitree_actuator_sdk import *

# Initialize serial communication
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the daisy-chained motors
motor_ids = [0, 1, 2]  # Controlling only the first 3 motors for now

# DH Parameters
a = [307, 19.65, 408.923, 0, 458.5, 0]
d = [0, 56.25, 39.875, 66.032, 0, 0]
alpha = [0, 0, 0, math.pi/2, math.pi/2, math.pi/2]

# Movement step sizes (in millimeters)
step_xy = 10.0  # Step size for X and Y movements
step_z = 10.0   # Step size for Z movements

# Starting end-effector position (relative to home position)
# Assuming home position corresponds to [0, 0, 0] in task space
desired_px = 0.0
desired_py = 0.0
desired_pz = 0.0

# Lock for thread-safe operations
position_lock = threading.Lock()

# Key mappings
key_map = {
    "move_up": keyboard.Key.up,
    "move_down": keyboard.Key.down,
    "move_left": keyboard.Key.left,
    "move_right": keyboard.Key.right,
    "move_upward": keyboard.KeyCode(char='w'),
    "move_downward": keyboard.KeyCode(char='s'),
}

# Thread-safe flags to track key states
key_flags = {action: False for action in key_map.keys()}
key_flags_lock = threading.Lock()

def inverse_kinematics(px, py, pz, R=np.eye(3)):
    """
    Compute inverse kinematics for the first three joints.
    Returns joint angles in degrees.
    """
    # Compute wrist center (assuming d6 = 0)
    wc_x = px - 0 * R[0, 2]
    wc_y = py - 0 * R[1, 2]
    wc_z = pz - 0 * R[2, 2]

    # Solve for theta1
    theta1 = np.arctan2(wc_y, wc_x)

    # Compute planar distance
    planar_dist = np.sqrt(wc_x**2 + wc_y**2) - a[0]
    L2 = planar_dist
    L3 = a[2]

    # Solve for theta3 using cosine law
    cos_theta3 = (L2**2 + d[2]**2 - L3**2) / (2 * L2 * d[2])
    # Clamp cos_theta3 to the valid range [-1, 1] to avoid numerical issues
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)

    # Solve for theta2
    theta2 = np.arctan2(wc_z - d[1], L2) - np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))

    # For simplicity, set theta4, theta5, theta6 to zero
    theta4 = 0.0
    theta5 = 0.0
    theta6 = 0.0

    return np.degrees([theta1, theta2, theta3, theta4, theta5, theta6])

def forward_kinematics(joint_angles_deg):
    """
    Compute forward kinematics for the first three joints.
    Returns the end-effector position (px, py, pz).
    """
    joint_angles = np.radians(joint_angles_deg[:3])  # Only first 3 joints
    T = np.eye(4)
    for i in range(3):
        theta = joint_angles[i]
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])

        A = np.array([
            [ct, -st*ca, st*sa, a[i]*ct],
            [st, ct*ca, -ct*sa, a[i]*st],
            [0, sa, ca, d[i]],
            [0, 0, 0, 1]
        ])
        T = T @ A

    # Extract position from transformation matrix
    px = T[0, 3]
    py = T[1, 3]
    pz = T[2, 3]
    return px, py, pz

def process_keyboard_input():
    """Thread to handle keyboard input events."""
    def on_press(key):
        with key_flags_lock:
            for action, mapped_key in key_map.items():
                if key == mapped_key:
                    key_flags[action] = True

    def on_release(key):
        with key_flags_lock:
            for action, mapped_key in key_map.items():
                if key == mapped_key:
                    key_flags[action] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

def control_motor(motor_id, position_rad):
    """Send control commands to a specific motor."""
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = position_rad
    cmd.dq = 0.0
    cmd.kp = 0.2  # Proportional gain
    cmd.kd = 0.05 # Derivative gain
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print(f"Motor {motor_id}:")
    print(f"  q: {math.degrees(data.q):.2f} degrees")
    print(f"  dq: {data.dq}")
    print(f"  temp: {data.temp}")
    print(f"  merror: {data.merror}")

def initialize_motors(initial_angles_deg):
    """Initialize motors to the starting joint angles."""
    initial_angles_rad = np.radians(initial_angles_deg[:3])
    for i, motor_id in enumerate(motor_ids):
        control_motor(motor_id, initial_angles_rad[i])
        time.sleep(0.1)  # Small delay to ensure commands are sent sequentially

def main():
    global desired_px, desired_py, desired_pz

    # Starting joint angles in degrees for the first three joints
    starting_angles_deg = [10.0, 10.0, 45.0]  # [θ1, θ2, θ3]

    # Initialize desired end-effector position based on starting angles
    with position_lock:
        desired_px, desired_py, desired_pz = forward_kinematics(starting_angles_deg)

    # Start keyboard input thread
    keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
    keyboard_thread.start()

    print("Control the end-effector with the following keys:")
    print("Arrow Up: Move Forward")
    print("Arrow Down: Move Backward")
    print("Arrow Left: Move Left")
    print("Arrow Right: Move Right")
    print("W: Move Upward")
    print("S: Move Downward")
    print("Press 'Ctrl+C' to exit.")

    # Initialize motors to starting angles
    initialize_motors(starting_angles_deg)

    try:
        while True:
            # Update desired position based on key flags
            with key_flags_lock:
                if key_flags["move_up"]:
                    with position_lock:
                        desired_py += step_xy
                if key_flags["move_down"]:
                    with position_lock:
                        desired_py -= step_xy
                if key_flags["move_left"]:
                    with position_lock:
                        desired_px -= step_xy
                if key_flags["move_right"]:
                    with position_lock:
                        desired_px += step_xy
                if key_flags["move_upward"]:
                    with position_lock:
                        desired_pz += step_z
                if key_flags["move_downward"]:
                    with position_lock:
                        desired_pz -= step_z

            # Compute inverse kinematics to get joint angles
            with position_lock:
                desired_angles_deg = inverse_kinematics(desired_px, desired_py, desired_pz)

            # Send commands to the first three motors
            for i, motor_id in enumerate(motor_ids):
                angle_rad = math.radians(desired_angles_deg[i])
                control_motor(motor_id, angle_rad)

            # Compute forward kinematics to get actual end-effector position
            actual_angles_deg = desired_angles_deg  # Assuming motors reach desired angles perfectly
            actual_px, actual_py, actual_pz = forward_kinematics(actual_angles_deg)

            # Log desired and actual positions
            print(f"Desired Position: px={desired_px:.2f} mm, py={desired_py:.2f} mm, pz={desired_pz:.2f} mm")
            print(f"Actual Position:  px={actual_px:.2f} mm, py={actual_py:.2f} mm, pz={actual_pz:.2f} mm")
            print(f"Desired Angles: {desired_angles_deg[:3]}")
            print(f"Actual Angles:  {actual_angles_deg[:3]}")
            print("-" * 50)

            time.sleep(0.1)  # Adjust the loop rate as needed

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
    finally:
        # Safely stop all motors by holding their current positions
        with position_lock:
            final_angles_deg = inverse_kinematics(desired_px, desired_py, desired_pz)
        final_angles_rad = np.radians(final_angles_deg[:3])
        for i, motor_id in enumerate(motor_ids):
            cmd.id = motor_id
            cmd.q = final_angles_rad[i]  # Hold the current position
            cmd.dq = 0.0
            cmd.tau = 0.0
            serial.sendRecv(cmd, data)
        print("All motors stopped.")

if __name__ == "__main__":
    main()
