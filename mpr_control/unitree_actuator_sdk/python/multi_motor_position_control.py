import time
import math
import sys
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication for three motors
serials = [SerialPort('/dev/ttyUSB0'), SerialPort('/dev/ttyUSB1'), SerialPort('/dev/ttyUSB2')]
commands = [MotorCmd() for _ in range(3)]
data = [MotorData() for _ in range(3)]
gear_ratios = [queryGearRatio(MotorType.GO_M8010_6)] * 3

# Initialize parameters for each motor
desired_positions = [0.0] * 3  # Initial desired positions in radians
kp_min, kp_max, kd = 0.1, 10.0, 0.01
position_step = 10 * math.pi / 180  # 10 degrees in radians
key_states = {  # Track key states (pressed or not)
    "motor1_inc": False, "motor1_dec": False, 
    "motor2_inc": False, "motor2_dec": False, 
    "motor3_inc": False, "motor3_dec": False
}

# Key mappings for each motor
key_map = {
    "motor1_inc": keyboard.KeyCode(char='w'),
    "motor1_dec": keyboard.KeyCode(char='s'),
    "motor2_inc": keyboard.KeyCode(char='e'),
    "motor2_dec": keyboard.KeyCode(char='d'),
    "motor3_inc": keyboard.KeyCode(char='r'),
    "motor3_dec": keyboard.KeyCode(char='f')
}

# Key press event handler
def on_press(key):
    global key_states
    for action, mapped_key in key_map.items():
        if key == mapped_key:
            key_states[action] = True

# Key release event handler
def on_release(key):
    global key_states
    for action, mapped_key in key_map.items():
        if key == mapped_key:
            key_states[action] = False

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Control motors with the following keys:")
print("Motor 1: 'w' to increase, 's' to decrease.")
print("Motor 2: 'e' to increase, 'd' to decrease.")
print("Motor 3: 'r' to increase, 'f' to decrease.")
print("Press 'Ctrl+C' to exit.")

try:
    while True:
        # Process key states for each motor
        for i, (inc_key, dec_key) in enumerate([("motor1_inc", "motor1_dec"),
                                                ("motor2_inc", "motor2_dec"),
                                                ("motor3_inc", "motor3_dec")]):
            if key_states[inc_key] and not key_states[dec_key]:  # Increase only if decrease key is not pressed
                desired_positions[i] += position_step
                desired_positions[i] = min(desired_positions[i], 2 * math.pi * gear_ratios[i])  # Clamp max
                print(f"Motor {i+1}: Increasing to {math.degrees(desired_positions[i] / gear_ratios[i]):.2f} degrees")
            elif key_states[dec_key] and not key_states[inc_key]:  # Decrease only if increase key is not pressed
                desired_positions[i] -= position_step
                desired_positions[i] = max(desired_positions[i], 0)  # Clamp min
                print(f"Motor {i+1}: Decreasing to {math.degrees(desired_positions[i] / gear_ratios[i]):.2f} degrees")

        # Update motor commands
        for i in range(3):
            serial = serials[i]
            cmd = commands[i]
            dat = data[i]

            # Read current motor position
            dat.motorType = MotorType.GO_M8010_6
            serial.sendRecv(cmd, dat)
            current_position = dat.q

            # Compute position error and adjust kp
            position_error = abs(desired_positions[i] - current_position)
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
