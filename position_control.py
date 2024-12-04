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

# Define target positions
gearratio = queryGearRatio(MotorType.GO_M8010_6)
position_1 = 0.0
position_2 = 0.78 * gearratio
target_position = position_2  # Start by moving to position_2

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

print("Press 'q' to exit the program.")

try:
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
        current_velocity = data.dq

        # PD control to calculate desired torque
        position_error = target_position - current_position

        # Check if the motor has reached the target position (within a small tolerance)
        if abs(position_error) < 0.01 * gearratio:
            print("Target position reached. Holding...")
            time.sleep(2)  # Hold the position for 2 seconds
            # Switch target position
            target_position = position_1 if target_position == position_2 else position_2

        # Check for 'q' key to exit
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            if get_key() == 'q':
                print("Exiting program...")
                break

        # Small delay for loop control rate
        time.sleep(0.0002)  # 200 microseconds

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    # Stop the motor safely
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
