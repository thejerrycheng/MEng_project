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
velocity = 3.14 * gearratio
increment = 0.2 * gearratio
decrement = 0.2 * gearratio

def get_key():
    # Function to get the pressed key from the keyboard without waiting for Enter
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

print("Control motor velocity with arrow keys:")
print("↑ : Increase velocity by 0.2 * gearratio")
print("↓ : Decrease velocity by 0.2 * gearratio")
print("Press 'q' to exit.")

try:
    while True:
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q = 0.0
        cmd.dq = velocity
        cmd.kp = 0.0
        cmd.kd = 0.01
        cmd.tau = 0.0

        serial.sendRecv(cmd, data)
        print('\n')
        print(f"Current velocity (dq): {data.dq / gearratio}")
        print("Temperature: " + str(data.temp))
        print("Error status: " + str(data.merror))
        print('\n')
        
        # Check for user input
        key = get_key()
        
        if key == '\x1b':  # This is the ESC character indicating an arrow key
            next_key = get_key()  # Read the next character to determine arrow direction
            if next_key == '[':
                direction_key = get_key()
                if direction_key == 'A':  # Arrow Up
                    velocity += increment
                    print("Increasing velocity.")
                elif direction_key == 'B':  # Arrow Down
                    velocity -= decrement
                    print("Decreasing velocity.")
        elif key == 'q':
            print("Exiting...")
            break

        # Small delay to control command rate
        time.sleep(0.0002)  # 200 us delay

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")
finally:
    # Optionally, you could add commands here to stop the motor safely.
    cmd.dq = 0  # Set velocity to zero to stop the motor
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
