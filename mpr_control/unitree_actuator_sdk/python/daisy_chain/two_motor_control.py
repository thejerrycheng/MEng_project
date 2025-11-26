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

# Motor IDs for the daisy-chained motors (only 2 motors)
motor_ids = [4, 5]

# Initialize desired positions for each motor (in radians)
desired_positions = [0.0, 0.0]  # Motor 1 and Motor 2
position_step = math.radians(5)  # 1 degree in radians

# Key mappings for motors (only Motor 1 and Motor 2)
key_map = {
    "motor1_increase": keyboard.KeyCode(char='q'),
    "motor1_decrease": keyboard.KeyCode(char='a'),
    "motor2_increase": keyboard.KeyCode(char='w'),
    "motor2_decrease": keyboard.KeyCode(char='s'),
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

def control_motor(motor_id, position):
    """
    Sends a control command to a specific motor.

    Args:
        motor_id (int): The ID of the motor to control.
        position (float): The desired position in radians.
    """
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = position
    cmd.dq = 0.0
    cmd.kp = 0.5  # Proportional gain for position control
    cmd.kd = 0.01 # Derivative gain for position control
    cmd.tau = 0.0 # No torque applied
    serial.sendRecv(cmd, data)
    
    # Streamlined logging
    desired_deg = math.degrees(position)
    current_deg = math.degrees(data.q)
    torque = data.tau
    temperature = data.temp
    print(f"Motor {motor_id} | Desired Position: {desired_deg:.2f}° | Current Position: {current_deg:.2f}° | Torque: {torque:.2f} Nm | Temperature: {temperature}°C")

def initialize_motor_positions():
    """
    Initialize motor positions by sending a 0 position command and reading the current position.
    This function ensures that each motor starts from a known position.
    """
    timeout = 2  # Timeout in seconds
    start_time = time.time()

    for i, motor_id in enumerate(motor_ids):
        # Send a 0 position command to the motor
        cmd.id = motor_id
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.q = 0.0  # Set initial position to 0 radians
        cmd.dq = 0.0
        cmd.kp = 1.0
        cmd.kd = 0.05
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)

        # Wait for the motor to respond with its current position
        while True:
            serial.sendRecv(cmd, data)
            current_position_deg = math.degrees(data.q)

            # Sanity check: Ensure the position is within 0-360 degrees
            if 0 <= current_position_deg <= 360:
                desired_positions[i] = data.q  # Set the current position as the initial desired position
                print(f"Motor {motor_id} initialized to: {current_position_deg:.2f}°")
                break
            else:
                print(f"Motor {motor_id} position out of range: {current_position_deg:.2f}°. Retrying...")

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
print("Press 'Ctrl+C' to exit.")

try:
    while True:
        # Process key flags and update desired positions
        with key_flags_lock:
            for i, (inc_key, dec_key) in enumerate([
                ("motor1_increase", "motor1_decrease"),
                ("motor2_increase", "motor2_decrease"),
            ]):
                if key_flags[inc_key]:
                    desired_positions[i] += position_step
                    print(f"Motor {motor_ids[i]} | Desired Position Updated: {math.degrees(desired_positions[i]):.2f}°")
                if key_flags[dec_key]:
                    desired_positions[i] -= position_step
                    print(f"Motor {motor_ids[i]} | Desired Position Updated: {math.degrees(desired_positions[i]):.2f}°")

        # Update motor commands
        for i, motor_id in enumerate(motor_ids):
            control_motor(motor_id, desired_positions[i])
            
        print("")

        time.sleep(0.005)  # Small delay to prevent excessive CPU usage

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Safely stop all motors by holding their current positions
    print("Stopping all motors...")
    for i, motor_id in enumerate(motor_ids):
        cmd.id = motor_id
        cmd.q = desired_positions[i]  # Hold the current position
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
        # Log the final state
        desired_deg = math.degrees(desired_positions[i])
        current_deg = math.degrees(data.q)
        torque = data.tau
        temperature = data.temp
        print(f"Motor {motor_id} | Desired Position: {desired_deg:.2f}° | Current Position: {current_deg:.2f}° | Torque: {torque:.2f} Nm | Temperature: {temperature}°C")
    print("All motors stopped.")
