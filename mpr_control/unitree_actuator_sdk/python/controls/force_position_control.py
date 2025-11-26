import time
import math
import sys
from pynput import keyboard  # For detecting key presses

sys.path.append('../lib')
from unitree_actuator_sdk import *

serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor parameters
gear_ratio = 6.33
kp = 0.02  # Position gain
kd = 0.01  # Velocity gain
kf = 0.01  # Force gain (if applicable)
desired_position = 3.14  # Target position in radians
desired_velocity = 0.0  # Target velocity in rad/s
feedforward_torque = 0.1  # Feedforward torque in Nm
epsilon = 0.01  # Sliding mode smoothing factor
position_step = 0.1  # Step size for position adjustment

# Function to initialize the motor
def initialize_motor():
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = 0
    cmd.q = 0
    cmd.dq = 0
    cmd.tau = 0
    cmd.kp = kp
    cmd.kd = kd
    for _ in range(10):  # Send initial commands to ensure the motor is ready
        serial.sendRecv(cmd, data)
        time.sleep(0.1)
    print("Motor initialized.")

def control_motor(position, velocity, torque):
    """Advanced control for FOC motor."""
    global desired_position

    # Validate data feedback
    if not (math.isfinite(position) and math.isfinite(velocity)):
        print("[ERROR] Invalid motor feedback. Skipping this iteration.")
        return
    
    data = MotorData()

    # Compute position and velocity error
    position_error = desired_position - position
    velocity_error = desired_velocity - velocity

    # PD control with feedforward torque
    control_torque = kp * position_error + kd * velocity_error + torque

    # Sliding mode control for robustness
    sliding_surface = velocity_error + 0.5 * position_error
    smc_torque = -0.1 * sliding_surface / (abs(sliding_surface) + epsilon)

    # Total control
    total_torque = control_torque + smc_torque

    # Send command
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = 0
    cmd.q = desired_position * gear_ratio
    cmd.dq = desired_velocity * gear_ratio
    cmd.tau = total_torque
    cmd.kp = kp
    cmd.kd = kd
    serial.sendRecv(cmd, data)

    # Print feedback
    print(f"Desired Position: {desired_position:.4f} rad")
    print(f"Position: {data.q} rad")
    print(f"Velocity: {data.dq } rad/s")
    print(f"Torque: {data.tau} Nm")
    print(f"Temperature: {data.temp} °C")

# Define key press handler
def on_press(key):
    global desired_position
    try:
        if key == keyboard.Key.right:  # Increase desired position
            desired_position += position_step
        elif key == keyboard.Key.left:  # Decrease desired position
            desired_position -= position_step
    except AttributeError:
        pass

# Start the listener in a separate thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    initialize_motor()  # Initialize the motor before starting the loop
    while True:
        control_motor(data.q / gear_ratio, data.dq / gear_ratio, feedforward_torque)
        time.sleep(0.005)
except KeyboardInterrupt:
    print("Control stopped.")
