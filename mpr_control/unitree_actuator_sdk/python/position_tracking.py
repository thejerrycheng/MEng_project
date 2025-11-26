import time
import math
import sys
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication and motor command
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Sine wave parameters for smooth motion
amplitude = math.radians(45)  # 90 degrees in radians
frequency = 0.05  # Frequency of the motion (Hz)
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)  # Query gear ratio

# Closed-loop control gains
kp = 0.25  # Proportional gain for position control
kd = 0.0  # Derivative gain for damping

cmd.motorType = MotorType.GO_M8010_6
cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
cmd.id = 0

# Time tracking
start_time = time.time()

while True:
    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Compute desired position based on sine wave
    desired_position = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)

    # Update motor command
    cmd.q = desired_position * gear_ratio  # Scale position by gear ratio
    cmd.dq = 0.0  # Set desired velocity to zero for position control
    cmd.kp = kp
    cmd.kd = kd
    cmd.tau = 0.0  # No feedforward torque

    # Send command and receive data
    serial.sendRecv(cmd, data)

    # Print debug information
    print('\n')
    print(f"q (current position): {math.degrees(data.q / gear_ratio):.2f} degrees")
    print(f"desired q: {math.degrees(desired_position):.2f} degrees")
    print(f"dq (current velocity): {data.dq:.2f} radians/s")
    print(f"temp: {data.temp:.2f} °C")
    print(f"merror: {data.merror}")
    print(f"Time elapsed: {elapsed_time:.2f} s")
    print('\n')

    # Control loop timing
    time.sleep(0.0002) # 200 us
    # time.sleep(0.1)  # 10 ms loop
