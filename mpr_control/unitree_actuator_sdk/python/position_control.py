import time
import sys
sys.path.append('../lib')
from unitree_actuator_sdk import *


serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)  # Query gear ratio

desired_position = 0.0  # Initial desired position (in radians)

while True:
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6,MotorMode.FOC)
    cmd.id = 0
    cmd.q    = desired_position
    cmd.dq   = 0
    cmd.kp   = 0.0
    cmd.kd   = 0.05
    cmd.tau  = 0.0
    serial.sendRecv(cmd, data)
    print('\n')
    print("q: " + str(data.q))
    print("dq: " + str(data.dq))
    print("temp: " + str(data.temp))
    print("merror: " + str(data.merror))
    print('\n') 
    time.sleep(0.0002) # 200 us