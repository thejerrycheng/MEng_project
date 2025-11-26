import time
import math
import sys
import threading
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the daisy-chained motors (0 to 5)
motor_ids = [0, 1, 2, 3, 4, 5]

# Fixed gear ratio for all motors
gear_ratio = 6.33  # Adjust this value based on your setup

# Initial desired positions in degrees for each motor
initial_positions_deg = [10, 45, 10, 0, 0, 0]

# Convert initial desired positions to motor radians considering gear ratio
# desired_output_rad = math.radians(desired_output_deg)
# desired_motor_rad = desired_output_rad * gear_ratio
desired_positions_motor_rad = [
    math.radians(pos_deg) * gear_ratio for pos_deg in initial_positions_deg
]

# Define step size for output shaft in degrees
position_step_output_deg = 1.0  # 1 degree step for output shaft

# Convert step size to motor radians considering gear ratio
position_step_motor_rad = math.radians(position_step_output_deg) * gear_ratio  # 1° output -> motor rotation

# Proportional Gain (Kp) for each motor
kp_values = [1, 5, 2, 1, 1, 1]

# Derivative Gain (Kd) for all motors (can be customized if needed)
kd_value = 0.05

# Key mappings for motors
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
}

# Thread-safe flags to track key states
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

def control_motor(motor_id, desired_position_motor_rad, kp):
    """
    Sends a control command to a specific motor, considering gear ratio.

    Args:
        motor_id (int): The ID of the motor to control.
        desired_position_motor_rad (float): The desired position in radians for the motor.
        kp (float): Proportional gain for the motor.
    """
    # Convert desired output position to motor radians
    desired_position_motor_command_rad = desired_position_motor_rad

    # Configure motor command
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = desired_position_motor_command_rad
    cmd.dq = 0.0
    cmd.kp = kp  # Proportional gain specific to each motor
    cmd.kd = kd_value  # Derivative gain common to all motors
    cmd.tau = 0.0  # No torque applied
    serial.sendRecv(cmd, data)
    
    # Retrieve current motor position
    current_position_motor_rad = data.q
    current_position_output_rad = current_position_motor_rad / gear_ratio
    current_position_output_deg = math.degrees(current_position_output_rad)

    # Desired output position in degrees
    desired_position_output_rad = desired_position_motor_rad / gear_ratio
    desired_position_output_deg = math.degrees(desired_position_output_rad)

    # Torque at output, scaled linearly by gear ratio
    torque_output = data.tau * gear_ratio  # Torque scaling: τ_output = τ_motor * gear_ratio

    temperature = data.temp

    # Streamlined logging
    print(f"Motor {motor_id} | Desired Pos: {desired_position_output_deg:.2f}° | "
          f"Current Pos: {current_position_output_deg:.2f}° | Torque: {torque_output:.2f} Nm | "
          f"Temp: {temperature}°C")

def initialize_motor_positions():
    """
    Initialize motor positions by sending the desired initial positions and verifying them.
    This function ensures that each motor starts from a known position.
    """
    timeout = 5  # Timeout in seconds

    for i, motor_id in enumerate(motor_ids):
        start_time = time.time()
        # Send the desired initial position command to the motor
        cmd.id = motor_id
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.q = desired_positions_motor_rad[i]
        cmd.dq = 0.0
        cmd.kp = kp_values[i]
        cmd.kd = kd_value
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)

        # Wait for the motor to reach the desired position
        while True:
            serial.sendRecv(cmd, data)
            current_position_motor_rad = data.q
            current_position_output_rad = current_position_motor_rad / gear_ratio
            current_position_output_deg = math.degrees(current_position_output_rad)
            desired_position_output_rad = desired_positions_motor_rad[i] / gear_ratio
            desired_position_output_deg = math.degrees(desired_position_output_rad)

            # Check if the motor has reached within a tolerance of 0.5 degrees
            if abs(current_position_output_deg - initial_positions_deg[i]) <= 0.5:
                print(f"Motor {motor_id} initialized to: {current_position_output_deg:.2f}°")
                break
            else:
                print(f"Motor {motor_id} | Current Pos: {current_position_output_deg:.2f}° | "
                      f"Desired Pos: {desired_position_output_deg:.2f}°. Retrying...")
            
            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"Error: Motor {motor_id} could not be initialized within {timeout} seconds. Exiting.")
                sys.exit(1)

            time.sleep(0.1)  # Small delay before retrying

# Uncomment the following line to initialize motor positions at startup
# initialize_motor_positions()

# Start keyboard input thread
keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
keyboard_thread.start()

print("Control the motors with the following keys:")
print("Motor 1: 'q' to increase, 'a' to decrease.")
print("Motor 2: 'w' to increase, 's' to decrease.")
print("Motor 3: 'e' to increase, 'd' to decrease.")
print("Motor 4: 'r' to increase, 'f' to decrease.")
print("Motor 5: 't' to increase, 'g' to decrease.")
print("Motor 6: 'y' to increase, 'h' to decrease.")
print("Press 'Ctrl+C' to exit.")

try:
    while True:
        # Process key flags and update desired positions
        with key_flags_lock:
            for i, (inc_key, dec_key) in enumerate([
                ("motor1_increase", "motor1_decrease"),
                ("motor2_increase", "motor2_decrease"),
                ("motor3_increase", "motor3_decrease"),
                ("motor4_increase", "motor4_decrease"),
                ("motor5_increase", "motor5_decrease"),
                ("motor6_increase", "motor6_decrease"),
            ]):
                if key_flags[inc_key]:
                    # Update desired position in motor radians
                    desired_positions_motor_rad[i] += math.radians(position_step_output_deg) * gear_ratio
                    desired_position_output_deg = math.degrees(desired_positions_motor_rad[i] / gear_ratio)
                    print(f"Motor {motor_ids[i]} | Desired Position Updated: {desired_position_output_deg:.2f}°")
                if key_flags[dec_key]:
                    # Update desired position in motor radians
                    desired_positions_motor_rad[i] -= math.radians(position_step_output_deg) * gear_ratio
                    desired_position_output_deg = math.degrees(desired_positions_motor_rad[i] / gear_ratio)
                    print(f"Motor {motor_ids[i]} | Desired Position Updated: {desired_position_output_deg:.2f}°")

        # Update motor commands
        for i, motor_id in enumerate(motor_ids):
            control_motor(motor_id, desired_positions_motor_rad[i], kp_values[i])

        time.sleep(0.05)  # Adjusted delay to accommodate gear ratio scaling

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Safely stop all motors by holding their current positions
    print("Stopping all motors...")
    for i, motor_id in enumerate(motor_ids):
        cmd.id = motor_id
        cmd.q = desired_positions_motor_rad[i]  # Hold the current motor position
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)

        # Retrieve final positions and torque
        current_position_motor_rad = data.q
        current_position_output_rad = current_position_motor_rad / gear_ratio
        current_position_output_deg = math.degrees(current_position_output_rad)
        desired_position_output_deg = math.degrees(desired_positions_motor_rad[i] / gear_ratio)
        torque_output = data.tau * gear_ratio  # Linear torque scaling
        temperature = data.temp

        # Log the final state
        print(f"Motor {motor_id} | Desired Pos: {desired_position_output_deg:.2f}° | "
              f"Current Pos: {current_position_output_deg:.2f}° | Torque: {torque_output:.2f} Nm | "
              f"Temp: {temperature}°C")
    print("All motors stopped.")
