import time
import sys
import termios
import tty
import select
sys.path.append('../lib')
from unitree_actuator_sdk import *

serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

kp = 0.02
kd = 0.0

# Define gear ratio and bounds
gearratio = queryGearRatio(MotorType.GO_M8010_6)
min_position = 0.0  # 0 radians
max_position = 2 * 3.14159 * gearratio  # 2*pi radians adjusted by gear ratio

# Helper function to read key press
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

print("Position Control Program")
print("Enter desired position in radians (0 to 2*pi after conversion to gear ratio).")
print("Press 'q' to exit the program.")

try:
    target_position = 0.0  # Start with the initial position
    while True:
        # Send command to the motor
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q = target_position
        cmd.dq = 0.0
        cmd.kp = kp
        cmd.kd = kd
        cmd.tau = 0.0

        serial.sendRecv(cmd, data)
        print('\n')
        print(f"Target Position: {target_position}")
        print(f"Current Position (q): {data.q}")
        print(f"Current Velocity (dq): {data.dq}")
        print(f"Torque (tau): {cmd.tau}")
        print(f"Temperature: {data.temp}")
        print(f"Error Status: {data.merror}")
        print('\n')

        current_position = data.q

        # PD control to calculate desired torque
        position_error = target_position - current_position

        # Check if the motor has reached the target position (within a small tolerance)
        if abs(position_error) < 0.01 * gearratio:
            print("Target position reached.")
            # Prompt the user for the next target position
            while True:
                user_input = input("Enter the next desired position in radians (0 to 2*pi): ").strip()
                if user_input.lower() == 'q':
                    print("Exiting program...")
                    raise KeyboardInterrupt
                try:
                    desired_position_angle = float(user_input)
                    if 0.0 <= desired_position_angle <= 2 * 3.14159:
                        target_position = desired_position_angle * gearratio  # Convert to motor scale
                        print(f"New target position set to: {target_position}")
                        break
                    else:
                        print("Position must be within 0 to 2*pi radians. Try again.")
                except ValueError:
                    print("Invalid input. Enter a numeric value between 0 and 2*pi or 'q' to exit.")

        # Small delay for loop control rate
        time.sleep(0.0002)  # 200 microseconds

except KeyboardInterrupt:
    print("\nProgram interrupted by user. Exiting gracefully...")
finally:
    # Stop the motor safely
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
