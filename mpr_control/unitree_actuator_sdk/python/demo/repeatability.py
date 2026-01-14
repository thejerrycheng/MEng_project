#!/usr/bin/env python
import time
import math
import sys
import threading
import csv
import numpy as np
import datetime

# Robot and Communication Setup
sys.path.append('../lib')
try:
    from unitree_actuator_sdk import *
except ImportError:
    print("Error: unitree_actuator_sdk not found.")
    sys.exit(1)

# ----------------------------
# Configuration
# ----------------------------
SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_IDS = [0, 1, 2, 3, 4, 5]
REPEAT_COUNT = 10
MOVE_DURATION = 6.0    # Time to move between points
RECORD_FREQ = 0.01     # 10ms (100Hz)

# Impedance settings for Active Motion
IMPEDANCE_SETTINGS = {
    0: {"kp": 8.0, "kd": 0.2},
    1: {"kp": 8.0, "kd": 0.2},
    2: {"kp": 8.0, "kd": 0.2},
    3: {"kp": 6.0, "kd": 0.1},
    4: {"kp": 6.0, "kd": 0.1},
    5: {"kp": 6.0, "kd": 0.1},
}

# Global Variables
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()

manual_log_data = []  # Stores the full manual path
waypoints = []        # Stores only the points marked with 'x'
is_recording = False

# Generate a unique timestamp string for this run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------
# Hardware Functions
# ----------------------------

def read_all_motors(passive=True):
    """
    Reads motor states.
    If passive=True, commands 0 gains (Gravity Comp/Limp).
    """
    positions = []
    for motor_id in MOTOR_IDS:
        data.motorType = MOTOR_TYPE
        cmd.motorType = MOTOR_TYPE
        cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        cmd.id = motor_id
        
        serial.sendRecv(cmd, data)
        positions.append(data.q)

        if passive:
            cmd.q = data.q # Follow current pos
            cmd.dq = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0
            cmd.tau = 0.0
            serial.sendRecv(cmd, data)
            
    return positions

def actuate_all(target_qs):
    """
    Commands all motors to target_qs with active impedance.
    Returns ACTUAL positions.
    """
    actuals = []
    for i, mid in enumerate(MOTOR_IDS):
        data.motorType = MOTOR_TYPE
        cmd.motorType = MOTOR_TYPE
        cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        cmd.id = mid
        cmd.q = target_qs[i]
        cmd.dq = 0.0
        cmd.kp = IMPEDANCE_SETTINGS[mid]["kp"]
        cmd.kd = IMPEDANCE_SETTINGS[mid]["kd"]
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
        actuals.append(data.q)
    return actuals

# ----------------------------
# Background Recorder
# ----------------------------
def recording_task():
    global manual_log_data, is_recording
    print("   [Background] Continuous recording started...")
    start_time = time.time()
    
    while is_recording:
        # Read passive (limp)
        current_pos = read_all_motors(passive=True)
        
        manual_log_data.append({
            "time": time.time(),
            "elapsed": time.time() - start_time,
            "positions": current_pos
        })
        time.sleep(RECORD_FREQ)
    print("   [Background] Recording stopped.")

# ----------------------------
# Motion Logic
# ----------------------------

def move_segment(start_pos, end_pos, duration, cycle_num, segment_name, log_list):
    """
    Moves smoothly from start_pos to end_pos using Cosine interpolation.
    Logs execution data to log_list.
    """
    t_start = time.time()
    
    while True:
        now = time.time()
        elapsed = now - t_start
        if elapsed > duration:
            break
            
        # Interpolation (0 to 1)
        alpha = elapsed / duration
        alpha_smooth = 0.5 * (1 - math.cos(math.pi * alpha))
        
        # Calculate Reference
        current_cmd = []
        for i in range(len(start_pos)):
            val = start_pos[i] + (end_pos[i] - start_pos[i]) * alpha_smooth
            current_cmd.append(val)
        
        # Actuate
        actual_pos = actuate_all(current_cmd)
        
        # Log: [timestamp, cycle, segment, t0..t5, a0..a5]
        row = [now, cycle_num, segment_name] + current_cmd + actual_pos
        log_list.append(row)
        
        time.sleep(0.01)

    # Ensure final convergence
    actuate_all(end_pos)
    return end_pos

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    try:
        # 1. Initialization
        print("Initializing Motors (Passive)...")
        home_pos = read_all_motors(passive=True)
        print("Done.")

        # 2. Start Recording & Teaching
        if input("\nStart Recording & Teaching? (y/n): ").strip().lower() != 'y':
            sys.exit(0)

        is_recording = True
        rec_thread = threading.Thread(target=recording_task)
        rec_thread.start()

        print("\n=== TEACHING MODE ===")
        print(" - Move robot manually.")
        print(" - Press 'x' + Enter to mark a Set Point.")
        print(" - Press 'f' + Enter to Finish teaching.\n")

        while True:
            # We use input() which blocks, but that's fine for waypoints.
            # The background thread handles the high-freq logging.
            cmd_in = input("Command (x/f): ").strip().lower()
            
            # Capture current state (passive read)
            # We access the latest from motors, ignoring the thread for a moment
            current_snap = read_all_motors(passive=True)
            
            if cmd_in == 'x':
                waypoints.append(current_snap)
                print(f" -> Point {len(waypoints)} Captured.")
            elif cmd_in == 'f':
                break
            else:
                print("Invalid. Use 'x' or 'f'.")

        is_recording = False
        rec_thread.join()

        # 3. Save Manual Log (With Timestamp)
        manual_filename = f"manual_session_{run_timestamp}.csv"
        print(f"\nSaving full manual path ({len(manual_log_data)} samples) to {manual_filename}...")
        
        with open(manual_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "elapsed"] + [f"j{i}" for i in MOTOR_IDS])
            for entry in manual_log_data:
                writer.writerow([entry["time"], entry["elapsed"]] + entry["positions"])
        print("Saved.")

        if len(waypoints) < 2:
            print("Need at least 2 points for iteration. Exiting.")
            sys.exit(0)

        # 4. Repeatability Test
        if input(f"\nStart {REPEAT_COUNT}x Iteration Test? (y/n): ").strip().lower() != 'y':
            sys.exit(0)

        repeatability_log = []
        
        # -- Move Home -> Point 1 --
        print("\nMoving to First Set Point...")
        # Start from current (wherever user left it)
        current_robot = read_all_motors(passive=False) 
        current_robot = move_segment(current_robot, waypoints[0], 
                                     duration=3.0, 
                                     cycle_num=0, 
                                     segment_name="HomeToStart", 
                                     log_list=repeatability_log)
        time.sleep(0.5)

        # -- Loop 10x --
        for cycle in range(1, REPEAT_COUNT + 1):
            print(f"Cycle {cycle}/{REPEAT_COUNT}...")
            
            # Iterate through all taught points: P1 -> P2 -> P3 ...
            for i in range(len(waypoints)):
                
                target = waypoints[i]
                
                # Optimization: If we are already at P0 (at start of loop), skip move
                dist = sum(abs(current_robot[j] - target[j]) for j in range(6))
                if dist < 0.1: 
                    continue

                seg_name = f"ToPoint_{i+1}"
                current_robot = move_segment(current_robot, target, 
                                             duration=MOVE_DURATION, 
                                             cycle_num=cycle, 
                                             segment_name=seg_name, 
                                             log_list=repeatability_log)
                time.sleep(0.2)

            # Cycle Wrap: If we are at the last point, we need to go back to P0 
            # to start the next cycle (unless we are at the very end of all cycles).
            if cycle < REPEAT_COUNT:
                current_robot = move_segment(current_robot, waypoints[0], 
                                             duration=MOVE_DURATION, 
                                             cycle_num=cycle, 
                                             segment_name="WrapToStart", 
                                             log_list=repeatability_log)

        # -- Return Home --
        print("Test Complete. Returning Home...")
        move_segment(current_robot, home_pos, 
                     duration=3.0, 
                     cycle_num=999, 
                     segment_name="ReturnHome", 
                     log_list=repeatability_log)

        # 5. Save Test Data (With Timestamp)
        test_filename = f"repeatability_test_{run_timestamp}.csv"
        print(f"Saving test data to {test_filename}...")
        
        with open(test_filename, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp", "cycle", "segment"] + \
                     [f"tgt_m{i}" for i in MOTOR_IDS] + \
                     [f"act_m{i}" for i in MOTOR_IDS]
            writer.writerow(header)
            writer.writerows(repeatability_log)

        print("Done. Holding Home.")
        while True:
            actuate_all(home_pos)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)