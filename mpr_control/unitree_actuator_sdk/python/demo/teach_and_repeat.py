#!/usr/bin/env python
import time
import math
import sys
import threading
import csv

# For this example, we use input() to control recording.
sys.path.append('../lib')
from unitree_actuator_sdk import *

# ----------------------------
# Robot and Communication Setup
# ----------------------------
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the 6 joints.
motor_ids = [0, 1, 2, 3, 4, 5]

# Impedance settings for active playback (nonzero gains).
impedance_settings = {
    0: {"kp": 0.5, "kd": 0.05},
    1: {"kp": 2.5, "kd": 0.15},
    2: {"kp": 1.0, "kd": 0.15},
    3: {"kp": 0.3, "kd": 0.025},
    4: {"kp": 0.3, "kd": 0.025},
    5: {"kp": 0.3, "kd": 0.025},
}

# ----------------------------
# Passive Read Function (Demonstration Mode)
# ----------------------------
def read_motor(motor_id):
    """
    Reads the motor state in passive mode.
    To avoid applying any torque, we first read the current sensor value and then
    re-command that same value with zero gains.
    """
    # Read current sensor value.
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    current_position = data.q

    # Re-command current position with zero corrective gains.
    cmd.q = current_position
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    
    return {
         "id": motor_id,
         "position": data.q,
         "velocity": data.dq,
         "temperature": data.temp,
         "motor_error": data.merror
    }

def read_all_motors():
    """
    Reads all motor positions in passive mode and returns a list of positions.
    """
    positions = []
    for motor_id in motor_ids:
        motor_data = read_motor(motor_id)
        positions.append(motor_data["position"])
    return positions

# ----------------------------
# Recording Demonstration Data
# ----------------------------
# Each element is a dict: { "time": <timestamp>, "positions": [pos_motor0, pos_motor1, ...] }
demonstration_data = []
recording = True  # Global flag to control recording

def record_demonstration():
    """
    In demonstration mode the human manually moves the robot arm.
    This function continuously reads the motor sensor data in passive mode and records the joint positions.
    """
    global demonstration_data, recording
    last_print_time = time.time()
    try:
        while recording:
            current_positions = read_all_motors()
            demonstration_data.append({
                "time": time.time(),
                "positions": current_positions.copy()
            })
            # Optionally print current positions every 0.1 seconds.
            if time.time() - last_print_time >= 0.1:
                pos_str = "  ".join("{:.4f}".format(p) for p in current_positions)
                print("Recorded positions: " + pos_str)
                last_print_time = time.time()
            time.sleep(0.005)
    except Exception as e:
        print("Error during demonstration recording:", e)

# ----------------------------
# Active Actuation Function (Playback Mode)
# ----------------------------
def actuate_motor(motor_id, desired_position):
    """
    Actively command the motor to move to the desired_position using nonzero impedance gains.
    This function is used during playback and hold phases.
    """
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = desired_position
    cmd.dq = 0.0
    cmd.kp = impedance_settings[motor_id]["kp"]
    cmd.kd = impedance_settings[motor_id]["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

# ----------------------------
# Move Robot to Starting Position (Playback Mode)
# ----------------------------
def move_to_starting_position(start_positions):
    """
    Gradually moves the robot to the starting joint configuration (first recorded sample) during playback.
    If 5 seconds elapse without reaching the target, playback will commence from the current positions.
    """
    print("Moving to starting position...")
    threshold = 0.01  # Acceptable error in position
    start_time = time.time()
    while True:
        current_positions = read_all_motors()
        errors = [abs(current_positions[i] - start_positions[i]) for i in range(len(motor_ids))]
        if all(e < threshold for e in errors):
            break
        if time.time() - start_time > 5:
            print("Timeout reached while moving to starting position. Starting playback from current positions.")
            break
        for i, motor_id in enumerate(motor_ids):
            delta = start_positions[i] - current_positions[i]
            step = delta * 0.1  # Proportional step
            new_cmd = current_positions[i] + step
            actuate_motor(motor_id, new_cmd)
        time.sleep(0.1)
    print("Starting position reached (or timeout).")

# ----------------------------
# Playback the Recorded Demonstration
# ----------------------------
def playback_demonstration(demo_data):
    """
    Replays the demonstration path by actively commanding the motors to follow the recorded joint positions.
    The recorded timestamps are used to preserve the relative timing.
    """
    if not demo_data:
        print("No demonstration data recorded!")
        return

    print("Replaying demonstration...")
    t0 = demo_data[0]["time"]
    relative_times = [sample["time"] - t0 for sample in demo_data]
    playback_start = time.time()
    
    for i, sample in enumerate(demo_data):
        # Wait until it's time for the next command.
        desired_time = relative_times[i]
        while time.time() - playback_start < desired_time:
            time.sleep(0.001)
        positions = sample["positions"]
        for j, motor_id in enumerate(motor_ids):
            actuate_motor(motor_id, positions[j])
        time.sleep(0.005)
    print("Playback complete.")

# ----------------------------
# Main Routine
# ----------------------------
if __name__ == '__main__':
    try:
        # Initialize motors in passive mode (no torque applied).
        print("Initializing motors in passive mode (no torque applied)...")
        for motor_id in motor_ids:
            read_motor(motor_id)
        print("Motors are in passive state. You may now manually move the arm.")

        # Ask user to begin demonstration.
        answer = input("Are you ready to start the demonstration? (y/n): ")
        if answer.strip().lower() != 'y':
            print("Exiting demonstration program.")
            sys.exit(0)

        print("\n=== Expert Demonstration Mode ===")
        print("Manually move the robot arm. No active control is applied.")
        print("When finished, press ENTER to stop recording.\n")

        # Start recording demonstration data in a separate thread.
        recording = True
        demo_thread = threading.Thread(target=record_demonstration)
        demo_thread.start()

        # Wait for user to indicate the demonstration is finished.
        input("Press ENTER to stop recording the demonstration...")
        recording = False
        demo_thread.join()

        # Save the recorded data to a CSV file.
        csv_filename = "demonstration_data.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["timestamp"] + [f"motor{i+1}" for i in range(len(motor_ids))]
            writer.writerow(header)
            for sample in demonstration_data:
                row = [sample["time"]] + sample["positions"]
                writer.writerow(row)
        print("Demonstration data saved to", csv_filename)

        # Ask user if playback should be performed.
        repeat = input("Repeat the path? (y/n): ")
        if repeat.strip().lower() == 'y':
            # Move the robot to the starting configuration using active control.
            starting_positions = demonstration_data[0]["positions"]
            move_to_starting_position(starting_positions)
            # Replay the demonstration using active control.
            playback_demonstration(demonstration_data)
            # Use the last recorded positions for the final hold.
            final_positions = demonstration_data[-1]["positions"]
            print("Playback complete. Holding final position.")
        else:
            print("Playback not initiated. Holding current position.")
            final_positions = read_all_motors()

        # Hold the final position indefinitely.
        print("Holding final position. Press Ctrl+C to exit.")
        while True:
            for i, motor_id in enumerate(motor_ids):
                actuate_motor(motor_id, final_positions[i])
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
        # Instead of cutting power, hold the final position before exiting.
        print("Holding final position. Program terminated by user.")
