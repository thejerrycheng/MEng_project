#!/usr/bin/env python
import os
import sys
import time
import csv
import numpy as np
import mujoco
import mujoco.viewer

# 1. Path Setup - Ensuring we can find the Unitree SDK
# Based on your ls, SDK is in ../mpr_control/unitree_actuator_sdk/lib
SDK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mpr_control/unitree_actuator_sdk/lib'))
sys.path.append(SDK_PATH)

try:
    from unitree_actuator_sdk import *
except ImportError:
    print(f"Error: Could not find Unitree SDK at {SDK_PATH}")
    sys.exit(1)

# ----------------------------
# Configuration
# ----------------------------
SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
# Adjust this to your specific XML model file in the assets folder
MODEL_XML = 'assets/robot.xml' 
CALIBRATION_FILE = '../mpr_control/unitree_actuator_sdk/python/home_position.csv'

# ----------------------------
# Robot Communication Class
# ----------------------------
class RealRobotInterface:
    def __init__(self, port, ids):
        self.serial = SerialPort(port)
        self.ids = ids
        self.cmd = MotorCmd()
        self.data = MotorData()
        self.gear_ratio = queryGearRatio(MotorType.GO_M8010_6)
        self.offsets = np.zeros(len(ids))
        
        # Initialize motors in passive mode
        for mid in self.ids:
            self._send_passive(mid)
            
    def _send_passive(self, mid):
        self.cmd.motorType = MotorType.GO_M8010_6
        self.cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        self.cmd.id = mid
        self.cmd.kp = 0.0
        self.cmd.kd = 0.0
        self.cmd.tau = 0.0
        # Read current to maintain state but apply no torque
        self.serial.sendRecv(self.cmd, self.data)
        self.cmd.q = self.data.q
        self.serial.sendRecv(self.cmd, self.data)

    def load_calibration(self, filepath):
        print(f"Loading calibration from {filepath}...")
        try:
            with open(filepath, mode='r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < len(self.offsets):
                        self.offsets[i] = float(row['position_rad'])
            print("✓ Offsets loaded.")
        except FileNotFoundError:
            print("Warning: Calibration file not found. Using zero offsets.")

    def get_joint_angles(self):
        angles = []
        for i, mid in enumerate(self.ids):
            self.cmd.id = mid
            self.serial.sendRecv(self.cmd, self.data)
            # Actual Angle = (Current Reading / Gear Ratio) - Offset
            current_angle = (self.data.q / self.gear_ratio) - self.offsets[i]
            angles.append(current_angle)
        return np.array(angles)

# ----------------------------
# Main Execution
# ----------------------------
def main():
    # Initialize Real Robot
    robot = RealRobotInterface(SERIAL_PORT, MOTOR_IDS)
    robot.load_calibration(CALIBRATION_FILE)

    # Initialize MuJoCo
    if not os.path.exists(MODEL_XML):
        print(f"Error: XML model not found at {MODEL_XML}")
        return

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    print("\n" + "="*50)
    print("  REAL2SIM BRIDGE ACTIVE")
    print("  Move the physical arm to see movement in MuJoCo.")
    print("  Press ESC in the viewer to exit.")
    print("="*50 + "\n")

    # Start Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. Read from Physical Robot
            physical_q = robot.get_joint_angles()

            # 2. Update Simulation Data
            # Note: Ensure the order in physical_q matches your XML joint order
            data.qpos[:6] = physical_q

            # 3. Step Simulation (Forward Kinematics update)
            mujoco.mj_forward(model, data)

            # 4. Update Viewer
            viewer.sync()

            # Maintain roughly 100Hz
            time_until_next_step = 0.01 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()