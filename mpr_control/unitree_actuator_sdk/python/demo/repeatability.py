#!/usr/bin/env python3
import time
import sys
import csv
import argparse

sys.path.append('../lib')
from unitree_actuator_sdk import *

# --- Configuration ---
motor_ids = [0, 1, 2, 3, 4, 5]
impedance_settings = {
    0: {"kp": 1.5, "kd": 0.05},
    1: {"kp": 2.5, "kd": 0.15},
    2: {"kp": 2.5, "kd": 0.15},
    3: {"kp": 1.0, "kd": 0.025},
    4: {"kp": 1.0, "kd": 0.125},
    5: {"kp": 1.0, "kd": 0.125},
}

serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

def actuate_motor(motor_id, q):
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = q
    cmd.dq = 0.0
    cmd.kp = impedance_settings[motor_id]["kp"]
    cmd.kd = impedance_settings[motor_id]["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)

def move_to_start(target_q):
    """Smoothly interpolates to the first point of the demo."""
    print("Synchronizing to start position...")
    for _ in range(100): # ~1 second ramp
        for i, m_id in enumerate(motor_ids):
            actuate_motor(m_id, target_q[i])
        time.sleep(0.01)

def play_once(demo_rows):
    t0_rec = float(demo_rows[0][0])
    t0_play = time.time()
    
    for row in demo_rows:
        target_time = float(row[0]) - t0_rec
        while (time.time() - t0_play) < target_time:
            time.sleep(0.001)
        
        # Actuate all 6 motors
        for i in range(6):
            actuate_motor(motor_ids[i], float(row[i+1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True, help="Path to CSV file")
    parser.add_argument("--num", type=int, default=1, help="Number of repetitions")
    args = parser.parse_args()

    # Load CSV
    with open(args.demo, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        rows = list(reader)

    if not rows:
        print("Empty demo file.")
        sys.exit(1)

    start_pos = [float(x) for x in rows[0][1:]]

    try:
        for i in range(args.num):
            print(f"--- Starting Repeat {i+1}/{args.num} ---")
            move_to_start(start_pos)
            play_once(rows)
            print(f"Repeat {i+1} complete.")
        
        print("All repetitions finished. Holding final position...")
        final_pos = [float(x) for x in rows[-1][1:]]
        while True:
            for i in range(6):
                actuate_motor(motor_ids[i], final_pos[i])
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")