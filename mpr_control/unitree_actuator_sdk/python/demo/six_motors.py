import time
import math
import sys
import threading
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial port and command/data structures.
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
    2: {"kp": 2, "kd": 0.15},
    3: {"kp": 1.3, "kd": 0.125},
    4: {"kp": 1.5, "kd": 0.125},
    5: {"kp": 1.5, "kd": 0.125},
}

# Hard-coded HOME positions (in shaft radians).
home_positions = [
    math.radians(0),                   # Motor 0 HOME shaft angle
    math.radians(135) / gear_ratio,      # Motor 1 HOME shaft angle
    math.radians(45) / gear_ratio,       # Motor 2 HOME shaft angle
    0.0,                               # Motor 3 HOME shaft angle
    0.0,                               # Motor 4 HOME shaft angle
    0.0                                # Motor 5 HOME shaft angle
]

# A step used for incrementing/decrementing desired positions.
position_step = math.radians(2) / gear_ratio  # in shaft radians

# Key mappings for motors.
# The SPACE key is mapped to "go_home" for moving toward the HOME location.
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

# Thread-safe flags to track key states.
key_flags = {action: False for action in key_map.keys()}
key_flags_lock = threading.Lock()

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

def control_motor(motor_id, shaft_desired):
    """
    Sends a control command to the specified motor.
    Converts the desired shaft position (in rad) to the motor’s internal units by multiplying by the gear ratio.
    After sending the command, it converts the measured motor encoder position (dividing by the gear ratio)
    and computes the error.
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
    and returns a list of positions. This is used to initialize the desired positions.
    """
    positions = []
    for motor_id in motor_ids:
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = motor_id
        # Use the last commanded value to update sensor data without forcing a position change.
        cmd.q = 0.0
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
        positions.append(data.q / gear_ratio)
    return positions

# At startup, set the desired positions to the current positions.
desired_positions = get_initial_positions()

# Start keyboard input thread.
keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
keyboard_thread.start()

print("Control the motors with the following keys:")
print("Motor 1: 'q' to increase, 'a' to decrease.")
print("Motor 2: 'w' to increase, 's' to decrease.")
print("Motor 3: 'e' to increase, 'd' to decrease.")
print("Motor 4: 'r' to increase, 'f' to decrease.")
print("Motor 5: 't' to increase, 'g' to decrease.")
print("Motor 6: 'y' to increase, 'h' to decrease.")
print("Press SPACE to gradually move to the HOME position.")
print("Press 'Ctrl+C' to exit.")

# Initialize motors to the current positions.
for i, motor_id in enumerate(motor_ids):
    control_motor(motor_id, desired_positions[i])
    time.sleep(0.1)

last_print_time = time.time()

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

        current_time = time.time()
        if current_time - last_print_time >= 0.1:
            print("\n--- Motor Status Summary (Shaft Values) ---")
            print(f"{'Motor':>5} | {'Desired (deg)':>14} | {'Actual (deg)':>14} | {'Error (deg)':>12}")
            print("-" * 60)
            for info in motor_info_list:
                print(f"{info['id']:>5} | {math.degrees(info['desired']):>14.2f} | "
                      f"{math.degrees(info['actual']):>14.2f} | {math.degrees(info['error']):>12.2f}")
            last_print_time = current_time

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")

finally:
    # On termination, for each motor, re-read its current position without forcing any change
    # and then issue a hold command with the original impedance settings.
    for i, motor_id in enumerate(motor_ids):
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = motor_id
        # Use the last desired command value to update sensor data,
        # rather than forcing q=0.0. This minimizes any sudden jumps.
        cmd.q = desired_positions[i] * gear_ratio
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
        # Read the current sensor value.
        shaft_actual = data.q / gear_ratio
        # Update the desired position with the sensor reading.
        desired_positions[i] = shaft_actual
        # Issue a hold command using the current sensor reading.
        cmd.q = shaft_actual * gear_ratio
        cmd.dq = 0.0
        cmd.kp = impedance_settings[motor_id]["kp"]
        cmd.kd = impedance_settings[motor_id]["kd"]
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
    print("All motors are now holding their current positions with original impedance settings.")
