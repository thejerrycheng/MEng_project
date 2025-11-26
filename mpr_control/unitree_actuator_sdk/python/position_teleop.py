import time
import math
import sys
from pynput import keyboard  # For teleoperation
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication and motor command
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Closed-loop position control parameters
kp = 0.01  # Position control gain
kd = 0.0  # Velocity control gain
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)  # Query gear ratio for scaling
desired_position = 0.0  # Initial desired position (in radians)
position_step = math.radians(5)  # Teleoperation step size (5 degrees)

# Key flags for teleoperation
key_flags = {"up": False, "down": False, "quit": False}

# Function to handle key presses
def on_press(key):
    global key_flags
    try:
        if key == keyboard.Key.up:
            key_flags["up"] = True
        elif key == keyboard.Key.down:
            key_flags["down"] = True
        elif key.char == 'q':  # Quit the program
            key_flags["quit"] = True
    except AttributeError:
        pass

# Function to handle key releases
def on_release(key):
    global key_flags
    if key == keyboard.Key.up:
        key_flags["up"] = False
    elif key == keyboard.Key.down:
        key_flags["down"] = False

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Function to stop the motor
def stop_motor():
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = 0
    cmd.q = data.q  # Maintain the current position
    cmd.dq = 0.0  # Set velocity to 0
    cmd.kp = 0.0  # Turn off position control
    cmd.kd = 0.0  # Turn off damping
    cmd.tau = 0.0  # Zero torque
    serial.sendRecv(cmd, data)

# Main control loop
try:
    while True:
        if key_flags["quit"]:
            print("Exiting program...")
            stop_motor()
            break

        # Adjust desired position based on key flags
        if key_flags["up"]:
            desired_position += position_step
            print(f"Increasing desired position to: {math.degrees(desired_position):.2f} degrees")
        if key_flags["down"]:
            desired_position -= position_step
            print(f"Decreasing desired position to: {math.degrees(desired_position):.2f} degrees")

        # Read current position from motor
        serial.sendRecv(cmd, data)
        current_position = data.q  # Current position in radians (scaled by gear ratio)

        # Compute position error
        position_error = desired_position - current_position

        # Update motor command
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q =  1.52 #desired_position * gear_ratio #current_position + kp * position_error  # Update desired position using proportional control
        cmd.dq = 0.0  # No damping velocity
        cmd.kp = kp
        cmd.kd = kd
        cmd.tau = 0.0  # Feedforward torque (not used)

        # Send motor command
        serial.sendRecv(cmd, data)

        # Print motor state for debugging
        # print('\n')
        print("Current position (q): {:.2f} degrees".format(math.degrees(current_position)))
        # print("Desired position (q): {:.2f} degrees".format(math.degrees(desired_position)))
        print("Position error: {:.2f} degrees".format(math.degrees(position_error)))
        # print("dq (current velocity): {:. 2f} radians/s".format(data.dq))
        print("temp: {:.2f} C".format(data.temp))
        print("merror: " + str(data.merror))
        print('\n')

        # Sleep for a short duration to control loop timing
        time.sleep(0.01)  # 10 ms loop time

except KeyboardInterrupt:
    print("Program interrupted. Stopping motor...")
    stop_motor()
