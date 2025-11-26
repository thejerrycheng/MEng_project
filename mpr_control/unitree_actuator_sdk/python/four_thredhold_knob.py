import time
import sys
import termios
import tty
sys.path.append('../lib')
from unitree_actuator_sdk import *

serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Initial parameters
gearratio = queryGearRatio(MotorType.GO_M8010_6)

# Define target positions in radians: 0°, 90°, 180°, and 270°
position1 = 0
position2 = 90 * 3.14 / 180 * gearratio
position3 = 180 * 3.14 / 180 * gearratio
position4 = 270 * 3.14 / 180 * gearratio

# Define threshold positions as midpoints between the target positions
threshold1 = (position1 + position2) / 2
threshold2 = (position2 + position3) / 2
threshold3 = (position3 + position4) / 2
threshold4 = (position4 + position1 + 2 * 3.14 * gearratio) / 2  # Wraps around to 0°

def get_key():
    # Function to get the pressed key from the keyboard without waiting for Enter
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

print("Press 'q' to exit.")

try:
    desired_position = position1  # Start with the desired position at 0 degrees
    while True:
        # Update motor data
        data.motorType = MotorType.GO_M8010_6
        serial.sendRecv(cmd, data)  # Get current motor position data
        current_position = data.q

        # Determine the desired position based on thresholds
        if current_position < threshold1:
            desired_position = position1
        elif threshold1 <= current_position < threshold2:
            desired_position = position2
        elif threshold2 <= current_position < threshold3:
            desired_position = position3
        elif threshold3 <= current_position < threshold4:
            desired_position = position4
        else:
            desired_position = position1  # Wrap around to the initial position

        # Command motor to move to the desired position
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q = desired_position
        cmd.dq = 0.0
        cmd.kp = 0.05  # Increase proportional gain for quicker snapping
        cmd.kd = 0.0  # Set a small derivative gain for damping
        cmd.tau = 0.0

        # Send command to the motor
        serial.sendRecv(cmd, data)
        print('\n')
        print(f"Current Position (q): {data.q / gearratio} degrees")
        print(f"Desired Position: {desired_position / gearratio} degrees")
        print(f"Current Velocity (dq): {data.dq / gearratio}")
        print("Temperature: " + str(data.temp))
        print("Error Status: " + str(data.merror))
        print('\n')
        
        # Small delay to control command rate
        time.sleep(0.0002)  # 200 microseconds delay

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Stop the motor safely
    cmd.q = data.q  # Hold the current position
    cmd.dq = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
