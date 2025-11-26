import time
import math
import sys
import threading
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the daisy-chained motors
motor_ids = [0, 1, 2]

# Initialize desired positions for each motor
desired_positions = [0.0, 0.0, 0.0]  # Placeholder, will be updated with current positions
position_step = 1 * math.pi / 180  # 10 degrees in radians

# Key mappings for motors
key_map = {
    "motor1_increase": keyboard.KeyCode(char='q'),
    "motor1_decrease": keyboard.KeyCode(char='a'),
    "motor2_increase": keyboard.KeyCode(char='w'),
    "motor2_decrease": keyboard.KeyCode(char='s'),
    "motor3_increase": keyboard.KeyCode(char='e'),
    "motor3_decrease": keyboard.KeyCode(char='d'),
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
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = position
    cmd.dq = 0.0
    cmd.kp = 0.2  # Example proportional gain for position control
    cmd.kd = 0.05
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print(f"Motor {motor_id}:")
    print(f"  q: {data.q}")
    print(f"  dq: {data.dq}")
    print(f"  temp: {data.temp}")
    print(f"  merror: {data.merror}")

def initialize_motor_positions():
    """Initialize motor positions by sending a 0 position command and reading the current position."""
    timeout = 2  # Timeout in seconds
    start_time = time.time()

    for i, motor_id in enumerate(motor_ids):
        # Send a 0 position command to the motor
        cmd.id = motor_id
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.q = 0.0  # Set initial position to 0
        cmd.dq = 0.0
        cmd.kp = 1
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
                print(f"Motor {motor_id} initialized to: {current_position_deg:.2f} degrees")
                break
            else:
                print(f"Motor {motor_id} position out of range: {current_position_deg:.2f} degrees. Retrying...")

            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"Error: Motor {motor_id} could not be initialized within {timeout} seconds. Exiting.")
                # sys.exit(1)

            time.sleep(0.1)  # Small delay before retrying

# Initialize motor positions
# initialize_motor_positions()

# Start keyboard input thread
keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
keyboard_thread.start()

print("Control the motors with the following keys:")
print("Motor 1: 'q' to increase, 'a' to decrease.")
print("Motor 2: 'w' to increase, 's' to decrease.")
print("Motor 3: 'e' to increase, 'd' to decrease.")
print("Press 'Ctrl+C' to exit.")

try:
    while True:
        # Process key flags and update desired positions
        with key_flags_lock:
            for i, (inc_key, dec_key) in enumerate([
                ("motor1_increase", "motor1_decrease"),
                ("motor2_increase", "motor2_decrease"),
                ("motor3_increase", "motor3_decrease"),
            ]):
                if key_flags[inc_key]:
                    desired_positions[i] += position_step
                    print(f"Motor {motor_ids[i]}: Increasing to {math.degrees(desired_positions[i]):.2f} degrees")
                if key_flags[dec_key]:
                    desired_positions[i] -= position_step
                    print(f"Motor {motor_ids[i]}: Decreasing to {math.degrees(desired_positions[i]):.2f} degrees")

        # Update motor commands
        for i, motor_id in enumerate(motor_ids):
            control_motor(motor_id, desired_positions[i])

        time.sleep(0.005)  # Small delay to prevent excessive CPU usage

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Safely stop all motors
    for i, motor_id in enumerate(motor_ids):
        cmd.id = motor_id
        cmd.q = data.q  # Hold the current position
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
    print("All motors stopped.")