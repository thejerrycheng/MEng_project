import time
import math
import sys
import threading
import numpy as np
import csv
from pynput import keyboard

sys.path.append('../lib')
from unitree_actuator_sdk import *

# ----------------------------
# Robot and Communication Setup
# ----------------------------
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Retrieve the gear ratio for this motor type.
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)

# Motor IDs for the daisy-chained motors (6 motors).
motor_ids = [0, 1, 2, 3, 4, 5]

# Define impedance settings for each joint.
impedance_settings = {
    0: {"kp": 0.5, "kd": 0.05},
    1: {"kp": 2.5, "kd": 0.15},
    2: {"kp": 1, "kd": 0.15},
    3: {"kp": 0.3, "kd": 0.025},
    4: {"kp": 0.3, "kd": 0.025},
    5: {"kp": 0.3, "kd": 0.025},
}


# ----------------------------
# DH Parameters (units: mm for distances, radians for angles)
# ----------------------------
dh_params = {
    'a': [238.1, 19.65, 208.923, 0, 258.5, 0],
    'd': [0, 56.25, 39.875, 66.032, 0, 0],
    'alpha': [0, 0, 0, math.pi/2, math.pi/2, math.pi/2]
}

# ----------------------------
# Teleoperation Parameters & HOME Configuration
# ----------------------------
# Hard-coded HOME positions (in shaft radians) are stored separately.
home_positions = [
    math.radians(0),
    math.radians(135) / gear_ratio,
    math.radians(45) / gear_ratio,
    0.0,
    0.0,
    0.0
]

# Step size for joint adjustments (in shaft radians)
position_step = math.radians(2) / gear_ratio

# Key mappings for teleop and HOME command (SPACE moves toward HOME)
key_map = {
    "motor1_increase": keyboard.KeyCode(char='q'),
    "motor1_decrease": keyboard.KeyCode(char='a'),
    "motor2_increase": keyboard.KeyCode(char='w'),
    "motor2_decrease": keyboard.KeyCode(char='s'),
    "motor3_increase": keyboard.KeyCode(char='e'),
    "motor3_decrease": keyboard.KeyCode(char='d'),
    "motor4_increase": keyboard.KeyCode(char='r'),
    "motor4_decrease": keyboard.KeyCode(char='f'),
    "motor5_increase": keyboard.KeyCode(char='t'),
    "motor5_decrease": keyboard.KeyCode(char='g'),
    "motor6_increase": keyboard.KeyCode(char='y'),
    "motor6_decrease": keyboard.KeyCode(char='h'),
    "go_home": keyboard.Key.space,
}

# Thread-safe flags for key states.
key_flags = {action: False for action in key_map.keys()}
key_flags_lock = threading.Lock()

def process_keyboard_input():
    """Handles keyboard events for teleoperation."""
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

# ----------------------------
# Forward Kinematics Functions
# ----------------------------
def dh_transform(a, alpha, d, theta):
    """
    Computes the homogeneous transformation matrix for one joint.
    a: link length (mm)
    alpha: link twist (radians)
    d: link offset (mm)
    theta: joint angle (radians)
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])

def forward_kinematics(joint_angles, dh_params):
    """
    Computes the forward kinematics transformation matrix using the provided DH parameters.
    joint_angles: list of 6 joint angles (in shaft radians)
    dh_params: dictionary with keys 'a', 'd', 'alpha'
    Returns the cumulative 4x4 transformation matrix of the end-effector.
    """
    T = np.eye(4)
    for i in range(6):
        a = dh_params['a'][i]
        d = dh_params['d'][i]
        alpha = dh_params['alpha'][i]
        theta = joint_angles[i]
        T_i = dh_transform(a, alpha, d, theta)
        T = np.dot(T, T_i)
    return T

# ----------------------------
# Motor Control Functions
# ----------------------------
def control_motor(motor_id, shaft_desired):
    """
    Sends a control command to the specified motor.
    shaft_desired: desired shaft angle (radians)
    Converts the shaft angle to motor internal units and sends the command.
    Returns a dictionary with the motor's actual shaft position (radians) and other info.
    """
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    # Convert desired shaft angle to motor internal units.
    cmd.q = shaft_desired * gear_ratio
    cmd.dq = 0.0
    cmd.kp = impedance_settings[motor_id]["kp"]
    cmd.kd = impedance_settings[motor_id]["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    # Convert motor encoder reading to shaft position.
    shaft_actual = data.q / gear_ratio
    error_shaft = shaft_desired - shaft_actual
    return {
        "id": motor_id,
        "desired": shaft_desired,
        "actual": shaft_actual,
        "error": error_shaft,
        "dq": data.dq,
        "temp": data.temp,
        "merror": data.merror
    }

def get_initial_positions():
    """
    Reads the current motor encoder readings (converted to shaft radians)
    and returns a list of positions.
    """
    positions = []
    for motor_id in motor_ids:
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = motor_id
        # Use a dummy command to update sensor data.
        cmd.q = 0.0
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
        positions.append(data.q / gear_ratio)
    return positions

# ----------------------------
# Logging Setup
# ----------------------------
# Log end-effector position (world coordinates in mm) to CSV.
ee_log_file = open("end_effector_log.csv", "w", newline="")
ee_csv_writer = csv.writer(ee_log_file)
ee_csv_writer.writerow(["timestamp", "x (mm)", "y (mm)", "z (mm)"])

# ----------------------------
# Initialization
# ----------------------------
# At startup, set desired positions to the current positions.
desired_positions = get_initial_positions()

# Start keyboard input thread.
keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
keyboard_thread.start()

print("Control the motors with the following keys:")
print("  Motor 1: 'q' to increase, 'a' to decrease.")
print("  Motor 2: 'w' to increase, 's' to decrease.")
print("  Motor 3: 'e' to increase, 'd' to decrease.")
print("  Motor 4: 'r' to increase, 'f' to decrease.")
print("  Motor 5: 't' to increase, 'g' to decrease.")
print("  Motor 6: 'y' to increase, 'h' to decrease.")
print("Press SPACE to gradually move to the HOME position.")
print("Press Ctrl+C to exit.")

# Initialize motors to the current positions.
for i, motor_id in enumerate(motor_ids):
    control_motor(motor_id, desired_positions[i])
    time.sleep(0.1)

last_print_time = time.time()

# ----------------------------
# Main Loop
# ----------------------------
try:
    while True:
        with key_flags_lock:
            if key_flags["go_home"]:
                # Gradually move each desired position toward its HOME position.
                for i in range(len(desired_positions)):
                    if desired_positions[i] < home_positions[i]:
                        desired_positions[i] += position_step
                        if desired_positions[i] > home_positions[i]:
                            desired_positions[i] = home_positions[i]
                    elif desired_positions[i] > home_positions[i]:
                        desired_positions[i] -= position_step
                        if desired_positions[i] < home_positions[i]:
                            desired_positions[i] = home_positions[i]
            else:
                # Process manual key inputs.
                for i, (inc_key, dec_key) in enumerate([
                    ("motor1_increase", "motor1_decrease"),
                    ("motor2_increase", "motor2_decrease"),
                    ("motor3_increase", "motor3_decrease"),
                    ("motor4_increase", "motor4_decrease"),
                    ("motor5_increase", "motor5_decrease"),
                    ("motor6_increase", "motor6_decrease"),
                ]):
                    if key_flags[inc_key]:
                        desired_positions[i] += position_step
                    if key_flags[dec_key]:
                        desired_positions[i] -= position_step

        # Update motor commands and collect status info.
        motor_info_list = []
        for i, motor_id in enumerate(motor_ids):
            info = control_motor(motor_id, desired_positions[i])
            motor_info_list.append(info)

        # Compute forward kinematics using the actual shaft positions.
        joint_angles = [info["actual"] for info in motor_info_list]
        T = forward_kinematics(joint_angles, dh_params)
        # End-effector world coordinate (translation component, in mm).
        ee_pos = T[0:3, 3]
        timestamp = time.time()
        ee_csv_writer.writerow([timestamp, ee_pos[0], ee_pos[1], ee_pos[2]])
        ee_log_file.flush()

        current_time = time.time()
        if current_time - last_print_time >= 0.1:
            print("\n--- Motor Status Summary (Shaft Values) ---")
            print(f"{'Motor':>5} | {'Desired (deg)':>14} | {'Actual (deg)':>14} | {'Error (deg)':>12}")
            print("-" * 60)
            for info in motor_info_list:
                print(f"{info['id']:>5} | {math.degrees(info['desired']):>14.2f} | "
                      f"{math.degrees(info['actual']):>14.2f} | {math.degrees(info['error']):>12.2f}")
            print("\nEnd-effector Position (World, mm): x = {:.2f}, y = {:.2f}, z = {:.2f}".format(
                ee_pos[0], ee_pos[1], ee_pos[2]
            ))
            last_print_time = current_time

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")

finally:
    # On termination, hold each motor at its current (actual) position.
    for motor_id in motor_ids:
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = motor_id
        # Update sensor data using the last desired command.
        cmd.q = desired_positions[motor_id] * gear_ratio
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
        shaft_actual = data.q / gear_ratio
        desired_positions[motor_id] = shaft_actual
        cmd.q = shaft_actual * gear_ratio
        cmd.dq = 0.0
        cmd.kp = impedance_settings[motor_id]["kp"]
        cmd.kd = impedance_settings[motor_id]["kd"]
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
    ee_log_file.close()
    print("All motors held at current positions with original impedance settings.")
