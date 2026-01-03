#!/usr/bin/env python
import os
import sys

# --- FIX: Address the GLIBCXX Error ---
# We set LD_PRELOAD before the SDK tries to load the .so file
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'

import time
import csv
import numpy as np
import mujoco
import mujoco.viewer

# 1. Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_LIB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../../lib'))

if SDK_LIB_PATH not in sys.path:
    sys.path.insert(0, SDK_LIB_PATH)

try:
    from unitree_actuator_sdk import *
    print("✓ Unitree SDK loaded successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ----------------------------
# Configuration
# ----------------------------
SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
CALIBRATION_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../home_position.csv'))
MODEL_XML = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../mujoco_sim/assets/robot.xml'))

class RealRobotInterface:
    def __init__(self, port, ids):
        self.serial = SerialPort(port)
        self.ids = ids
        self.cmd = MotorCmd()
        self.data = MotorData()
        self.gear_ratio = queryGearRatio(MotorType.GO_M8010_6)
        self.offsets = np.zeros(len(ids))
        
        for mid in self.ids:
            self._set_passive(mid)
            
    def _set_passive(self, mid):
        self.cmd.motorType = MotorType.GO_M8010_6
        self.cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        self.cmd.id = mid
        self.cmd.kp = 0.0
        self.cmd.kd = 0.0
        self.cmd.tau = 0.0
        self.serial.sendRecv(self.cmd, self.data)
        self.cmd.q = self.data.q
        self.serial.sendRecv(self.cmd, self.data)

    def load_calibration(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, mode='r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < len(self.offsets):
                        self.offsets[i] = float(row['position_rad'])
            print("✓ Calibration applied.")

    def get_joint_angles(self):
        angles = []
        for i, mid in enumerate(self.ids):
            self.cmd.id = mid
            self.serial.sendRecv(self.cmd, self.data)
            angles.append((self.data.q / self.gear_ratio) - self.offsets[i])
        return np.array(angles)

def main():
    robot = RealRobotInterface(SERIAL_PORT, MOTOR_IDS)
    robot.load_calibration(CALIBRATION_FILE)

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Real2Sim Streaming Active...")
        while viewer.is_running():
            start = time.time()
            data.qpos[:6] = robot.get_joint_angles()
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0, 0.01 - (time.time() - start)))

if __name__ == '__main__':
    main()