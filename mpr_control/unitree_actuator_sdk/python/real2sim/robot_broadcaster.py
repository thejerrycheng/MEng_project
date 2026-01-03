import os
import sys
import time
import socket
import json
import numpy as np

# Path setup for Unitree SDK
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_LIB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../lib'))
sys.path.append(SDK_LIB_PATH)

from unitree_actuator_sdk import *

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Robot Config
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
SERIAL_PORT = '/dev/ttyUSB0'

class RobotBroadcaster:
    def __init__(self):
        self.serial = SerialPort(SERIAL_PORT)
        self.cmd = MotorCmd()
        self.data = MotorData()
        self.gear_ratio = queryGearRatio(MotorType.GO_M8010_6)
        
        # Set passive mode
        for mid in MOTOR_IDS:
            self.cmd.id = mid
            self.cmd.kp, self.cmd.kd, self.cmd.tau = 0.0, 0.0, 0.0
            self.serial.sendRecv(self.cmd, self.data)

    def get_angles(self):
        angles = []
        for mid in MOTOR_IDS:
            self.cmd.id = mid
            self.serial.sendRecv(self.cmd, self.data)
            angles.append(self.data.q / self.gear_ratio)
        return angles

if __name__ == "__main__":
    robot = RobotBroadcaster()
    print(f"Broadcasting robot state to {UDP_IP}:{UDP_PORT}...")
    
    try:
        while True:
            q = robot.get_angles()
            # Send as JSON string for easy parsing
            msg = json.dumps({"q": q}).encode()
            sock.sendto(msg, (UDP_IP, UDP_PORT))
            time.sleep(0.01) # 100Hz
    except KeyboardInterrupt:
        print("Broadcaster stopped.")