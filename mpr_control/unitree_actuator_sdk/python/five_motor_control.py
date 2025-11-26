import time
import math
import sys
import threading
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication and motor commands
serials = [SerialPort(f'/dev/ttyUSB{i}') for i in range(5)]
commands = [MotorCmd() for _ in range(5)]
data = [MotorData() for _ in range(5)]

# Initialize parameters
gearratio = queryGearRatio(MotorType.GO_M8010_6)
desired_positions = [0.0] * 5  # Initial desired positions for five motors
kp_min = 0.1  # Minimum proportional gain
kp_max = 10.0  # Maximum proportional gain
kd = 0.01  # Derivative gain
position_step = 10 * math.pi / 180  # 10 degrees in radians

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

# Start keyboard input thread
keyboard_thread = threading.Thread(target=process_keyboard_input, daemon=True)
keyboard_thread.start()

print("Control the motors with the following keys:")
print("Motor 1: 'q' to increase, 'a' to decrease.")
print("Motor 2: 'w' to increase, 's' to decrease.")
print("Motor 3: 'e' to increase, 'd' to decrease.")
print("Motor 4: 'r' to increase, 'f' to decrease.")
print("Motor 5: 't' to increase, 'g' to decrease.")
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
            ]):
                if key_flags[inc_key]:
                    desired_positions[i] += position_step
                    desired_positions[i] = min(desired_positions[i], 2 * math.pi * gearratio)  # Clamp to max
                    print(f"Motor {i+1}: Increasing to {math.degrees(desired_positions[i] / gearratio):.2f} degrees")
                if key_flags[dec_key]:
                    desired_positions[i] -= position_step
                    desired_positions[i] = max(desired_positions[i], 0)  # Clamp to min
                    print(f"Motor {i+1}: Decreasing to {math.degrees(desired_positions[i] / gearratio):.2f} degrees")

        # Update motor commands
        for i in range(5):  # Loop through all five motors
            serial = serials[i]
            cmd = commands[i]
            dat = data[i]

            # Read current motor position
            dat.motorType = MotorType.GO_M8010_6
            serial.sendRecv(cmd, dat)
            current_position = dat.q

            # Compute position error
            position_error = abs(desired_positions[i] - current_position)

            # Adjust kp dynamically based on the error
            kp = kp_min if position_error > 1.0 else kp_max * (1 - position_error) + kp_min * position_error

            # Command motor
            cmd.motorType = MotorType.GO_M8010_6
            cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
            cmd.id = 0
            cmd.q = desired_positions[i]
            cmd.dq = 0.0
            cmd.kp = kp
            cmd.kd = kd
            cmd.tau = 0

            # Send command to the motor
            serial.sendRecv(cmd, dat)

        # Small delay to prevent excessive CPU usage
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Safely stop all motors
    for i, serial in enumerate(serials):
        cmd = commands[i]
        dat = data[i]
        cmd.q = dat.q  # Hold the current position
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, dat)
    print("All motors stopped.")
