import time
import math
import sys
# Removed unused import threading and pynput.keyboard since we're not handling keyboard input
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Initialize serial communication
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the daisy-chained motors
motor_ids = [0, 1, 2, 3, 4, 5]  # Total of 6 motors

# Initialize desired positions for each motor in radians
# Motors 4, 5, and 6 are set to 0 radians (stationary)
desired_positions = [
    0.0,                      # Motor 1: Start at 0°
    math.radians(30),         # Motor 2: Start at 30°
    0.0,                      # Motor 3: Start at 0°
    0.0,                      # Motor 4: Stationary
    0.0,                      # Motor 5: Stationary
    0.0                       # Motor 6: Stationary
]

# Define step size (1 degree in radians)
position_step = math.radians(1)  # Approximately 0.01745 radians

# Define min and max positions for motors 1, 2, and 3
min_positions = [
    math.radians(0),   # Motor 1 minimum: 0°
    math.radians(30),  # Motor 2 minimum: 30°
    math.radians(0),   # Motor 3 minimum: 0°
    0.0,                # Motor 4: Stationary
    0.0,                # Motor 5: Stationary
    0.0                 # Motor 6: Stationary
]

max_positions = [
    math.radians(15),  # Motor 1 maximum: 15°
    math.radians(60),  # Motor 2 maximum: 60°
    math.radians(30),  # Motor 3 maximum: 30°
    0.0,                # Motor 4: Stationary
    0.0,                # Motor 5: Stationary
    0.0                 # Motor 6: Stationary
]

# Direction flags: 1 for increasing, -1 for decreasing
directions = [1, 1, 1, 0, 0, 0]  # Only motors 1, 2, and 3 move

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
    cmd.kd = 0.05 # Derivative gain for position control
    cmd.tau = 0.0 # No torque applied
    serial.sendRecv(cmd, data)
    
    # Debugging information
    print(f"Motor {motor_id}:")
    print(f"  q: {math.degrees(data.q):.2f} degrees")
    print(f"  dq: {data.dq}")
    print(f"  temp: {data.temp}")
    print(f"  merror: {data.merror}\n")

def main():
    print("Starting predefined path control for Motors 1, 2, and 3.")
    print("Motors 4, 5, and 6 remain stationary at 0°. Press 'Ctrl+C' to exit.\n")

    try:
        while True:
            # Update desired positions for Motors 1, 2, and 3
            for i in range(3):  # Indices 0, 1, 2 correspond to Motors 1, 2, 3
                # Update the desired position based on the current direction
                desired_positions[i] += directions[i] * position_step

                # Check if the motor has reached or exceeded its limits
                if desired_positions[i] > max_positions[i]:
                    desired_positions[i] = max_positions[i]
                    directions[i] = -1  # Reverse direction
                    print(f"Motor {motor_ids[i]} reached maximum position ({math.degrees(max_positions[i])}°). Reversing direction.")
                elif desired_positions[i] < min_positions[i]:
                    desired_positions[i] = min_positions[i]
                    directions[i] = 1   # Reverse direction
                    print(f"Motor {motor_ids[i]} reached minimum position ({math.degrees(min_positions[i])}°). Reversing direction.")
                
                # Debugging information for desired positions
                print(f"Motor {motor_ids[i]} desired position: {math.degrees(desired_positions[i]):.2f}°")

            print("")  # Add a newline for readability

            # Send control commands to all motors
            for i, motor_id in enumerate(motor_ids):
                control_motor(motor_id, desired_positions[i])

            # Wait for a short period before the next update
            time.sleep(0.05)  # 50 milliseconds

    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting gracefully...")
    finally:
        # Safely stop all motors by holding their current positions
        print("\nStopping all motors...")
        for i, motor_id in enumerate(motor_ids):
            cmd.id = motor_id
            cmd.q = desired_positions[i]  # Hold the current position
            cmd.dq = 0.0
            cmd.tau = 0.0
            serial.sendRecv(cmd, data)
        print("All motors have been stopped.")

if __name__ == "__main__":
    main()
