import time
import math
import sys
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication and motor commands
serials = [SerialPort('/dev/ttyUSB0'), SerialPort('/dev/ttyUSB1')]
commands = [MotorCmd(), MotorCmd()]
data = [MotorData(), MotorData()]

# Initialize parameters
gearratio = queryGearRatio(MotorType.GO_M8010_6)
desired_positions = [0.0, 0.0]  # Initial desired positions for two motors
kp_min = 0.1  # Minimum proportional gain
kp_max = 1.0  # Maximum proportional gain
kd = 0.01  # Derivative gain
position_step = 10 * math.pi / 180  # 10 degrees in radians
key_flags = {
    "motor1_increase": False,
    "motor1_decrease": False,
    "motor2_increase": False,
    "motor2_decrease": False,
}

# Key mappings for motors
key_map = {
    "motor1_increase": keyboard.KeyCode(char='q'),
    "motor1_decrease": keyboard.KeyCode(char='a'),
    "motor2_increase": keyboard.KeyCode(char='w'),
    "motor2_decrease": keyboard.KeyCode(char='s'),
}

# Key press event handler
def on_press(key):
    for action, mapped_key in key_map.items():
        if key == mapped_key:
            key_flags[action] = True

# Key release event handler
def on_release(key):
    for action, mapped_key in key_map.items():
        if key == mapped_key:
            key_flags[action] = False

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Control the motors with the following keys:")
print("Motor 1: 'q' to increase, 'a' to decrease.")
print("Motor 2: 'w' to increase, 's' to decrease.")
print("Press 'Ctrl+C' to exit.")

try:
    while True:
        # Check key flags and update desired positions
        if key_flags["motor1_increase"]:
            desired_positions[0] += position_step
            desired_positions[0] = min(desired_positions[0], 2 * math.pi * gearratio)  # Clamp to max
            print(f"Motor 1: Increasing to {math.degrees(desired_positions[0] / gearratio):.2f} degrees")
            key_flags["motor1_increase"] = False
        if key_flags["motor1_decrease"]:
            desired_positions[0] -= position_step
            desired_positions[0] = max(desired_positions[0], 0)  # Clamp to min
            print(f"Motor 1: Decreasing to {math.degrees(desired_positions[0] / gearratio):.2f} degrees")
            key_flags["motor1_decrease"] = False
        if key_flags["motor2_increase"]:
            desired_positions[1] += position_step
            desired_positions[1] = min(desired_positions[1], 2 * math.pi * gearratio)  # Clamp to max
            print(f"Motor 2: Increasing to {math.degrees(desired_positions[1] / gearratio):.2f} degrees")
            key_flags["motor2_increase"] = False
        if key_flags["motor2_decrease"]:
            desired_positions[1] -= position_step
            desired_positions[1] = max(desired_positions[1], 0)  # Clamp to min
            print(f"Motor 2: Decreasing to {math.degrees(desired_positions[1] / gearratio):.2f} degrees")
            key_flags["motor2_decrease"] = False

        # Update motor commands
        for i in range(2):  # Loop through both motors
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

        # Small delay to control loop rate
        time.sleep(0.01)

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
