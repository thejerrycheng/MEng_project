import time
import sys
sys.path.append('../lib')
from unitree_actuator_sdk import *

serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Define the four speeds using the gear ratio
gearratio = queryGearRatio(MotorType.GO_M8010_6)
speeds = [3.14 * gearratio, 6.28 * gearratio, 9.32 * gearratio, 1.28 * gearratio]
speed_index = 0  # To keep track of the current speed in the loop

while True:
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = 0
    cmd.q = 0.0
    cmd.dq = speeds[speed_index]  # Set the desired speed based on the current index
    cmd.kp = 0.0
    cmd.kd = 0.01
    cmd.tau = 0.0

    serial.sendRecv(cmd, data)
    print('\n')
    print("q: " + str(data.q))
    print("dq: " + str(data.dq))
    print("temp: " + str(data.temp))
    print("merror: " + str(data.merror))
    print('\n')
    
    # Sleep for 200 microseconds
    time.sleep(0.0002)
    
    # Move to the next speed after a certain time period
    # Example: Switch speed every 2 seconds
    time.sleep(2)
    speed_index = (speed_index + 1) % len(speeds)  # Loop back to the first speed after the last one
