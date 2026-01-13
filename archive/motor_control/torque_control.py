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
torque = 0.0
increment = 0.1
decrement = 0.1

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

print("Control motor torque with arrow keys:")
print("↑ : Increase torque by 0.1")
print("↓ : Decrease torque by 0.1")
print("Press 'q' to exit.")

try:
    while True:
        # Set motor type and control mode
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q = 0.0
        cmd.dq = 0.0
        cmd.kp = 0.0
        cmd.kd = 0.05
        cmd.tau = torque  # Set current torque

        # Send command to the motor
        serial.sendRecv(cmd, data)
        print('\n')
        print(f"Current torque (tau): {cmd.tau}")
        print("Temperature: " + str(data.temp))
        print("Error status: " + str(data.merror))
        print('\n')
        
        # Check for user input
        key = get_key()
        
        if key == '\x1b':  # ESC character indicates an arrow key
            next_key = get_key()  # Read next character for arrow direction
            if next_key == '[':
                direction_key = get_key()
                if direction_key == 'A':  # Arrow Up
                    torque += increment
                    print("Increasing torque.")
                elif direction_key == 'B':  # Arrow Down
                    torque -= decrement
                    print("Decreasing torque.")
        elif key == 'q':
            print("Exiting...")
            break

        # Small delay for control rate
        time.sleep(0.0002)  # 200 microseconds delay

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Set torque to zero to stop the motor safely
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
