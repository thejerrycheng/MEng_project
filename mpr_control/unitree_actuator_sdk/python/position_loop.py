import time
import math
import sys
from pynput import keyboard  # For key handling
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication and motor command
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Initialize parameters
gearratio = queryGearRatio(MotorType.GO_M8010_6)
desired_position = 0.0  # Initial desired position in radians
kp_min = 0.1  # Minimum proportional gain
kp_max = 1.0  # Maximum proportional gain
kd = 0.0  # Derivative gain
position_step = 1 * math.pi / 180   # 10 degrees in radians
key_flags = {"increase": False, "decrease": False}  # Flags for key presses

# Key press event handler
def on_press(key):
    global key_flags
    try:
        if key == keyboard.Key.right:  # Increase position
            key_flags["increase"] = True
        elif key == keyboard.Key.left:  # Decrease position
            key_flags["decrease"] = True
    except AttributeError:
        pass

# Key release event handler
def on_release(key):
    global key_flags
    if key == keyboard.Key.right:
        key_flags["increase"] = False
    elif key == keyboard.Key.left:
        key_flags["decrease"] = False

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Press Right Arrow to increase position by 10 degrees, Left Arrow to decrease position by 10 degrees. Press 'Ctrl+C' to exit.")

try:
    while True:
        # Check for key flags and update desired position
        if key_flags["increase"]:
            desired_position += position_step
            print(f"Increasing desired position to: {math.degrees(desired_position / gearratio):.2f} degrees")
            key_flags["increase"] = False
        if key_flags["decrease"]:
            desired_position -= position_step
            print(f"Decreasing desired position to: {math.degrees(desired_position / gearratio):.2f} degrees")
            key_flags["decrease"] = False

        # Read current motor position
        data.motorType = MotorType.GO_M8010_6
        serial.sendRecv(cmd, data)
        current_position = data.q  # Current motor position in radians

        # Compute position error
        position_error = abs(desired_position - current_position)

        # Adjust kp dynamically based on the error
        if position_error > 1.0:  # Large error
            kp = kp_min
        else:  # Small error
            kp = kp_max * (1 - position_error) + kp_min * position_error

        # Command motor to move to the desired position
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q = desired_position
        cmd.dq = 0.0
        cmd.kp = kp
        cmd.kd = kd
        cmd.tau = 0

        # Send command to the motor
        serial.sendRecv(cmd, data)

        # Print debugging information
        print('\n')
        print(f"Current Position (q): {data.q / gearratio:.2f} degrees")
        print(f"Desired Position: {desired_position / gearratio:.2f} degrees")
        print(f"Position Error: {position_error:.2f} radians")
        print(f"Current Kp: {kp:.2f}")
        print(f"Current Velocity (dq): {data.dq / gearratio:.2f} degrees/s")
        print(f"Temperature: {data.temp:.2f} C")
        print(f"Error Status: {data.merror}")
        print('\n')

        # Small delay to control loop rate
        time.sleep(0.001)  # 10 ms delay

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Safely stop the motor
    cmd.q = data.q  # Hold the current position
    cmd.dq = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
