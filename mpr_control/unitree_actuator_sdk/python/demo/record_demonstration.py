#!/usr/bin/env python3
import time
import sys
import os
import csv
from datetime import datetime

sys.path.append('../lib')
from unitree_actuator_sdk import *

# ============================
# Robot + Communication Setup
# ============================
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

motor_ids = [0, 1, 2, 3, 4, 5]
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)

# ============================
# Passive Read (Zero Torque)
# ============================
def read_motor_passive(motor_id):
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)

    # Recommand same position with zero gains
    cmd.q = data.q
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)

    return data.q / gear_ratio

def read_all_joints():
    return [read_motor_passive(i) for i in motor_ids]

# ============================
# Main Demonstration Recorder
# ============================
if __name__ == "__main__":
    print("\nInitializing motors in ZERO TORQUE mode...")
    for i in motor_ids:
        read_motor_passive(i)

    print("✓ Motors ready")
    print("✓ Arm is free to move\n")

    input("Press ENTER to start recording demonstration...")
    print("Recording... Move the robot arm freely.")
    print("Press Ctrl+C to stop.\n")

    os.makedirs("demonstrations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"demonstrations/{timestamp}_demo.csv"

    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "motor1","motor2","motor3",
                "motor4","motor5","motor6"
            ])

            while True:
                joints = read_all_joints()
                writer.writerow([time.time(), *joints])
                f.flush()
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    print(f"✓ Demonstration saved to: {out_path}")
    print("✓ You can now replay this with teach-and-repeat or repeatability_test.py")
