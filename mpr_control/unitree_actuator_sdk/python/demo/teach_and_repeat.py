#!/usr/bin/env python
import time
import math
import sys
import threading
import csv
import numpy as np

# Robot and Communication Setup
sys.path.append('../lib')
try:
    from unitree_actuator_sdk import *
except ImportError:
    print("Error: unitree_actuator_sdk not found. Ensure the library path is correct.")
    sys.exit(1)

# ----------------------------
# Configuration
# ----------------------------
SERIAL_PORT = '/dev/ttyUSB0'
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_IDS = [0, 1, 2, 3, 4, 5]

# Impedance settings for active playback (nonzero gains).
IMPEDANCE_SETTINGS = {
    0: {"kp": 1.5, "kd": 0.05},
    1: {"kp": 2.5, "kd": 0.15},
    2: {"kp": 2.5, "kd": 0.15},
    3: {"kp": 1.0, "kd": 0.025},
    4: {"kp": 1.0, "kd": 0.125},
    5: {"kp": 1.0, "kd": 0.125},
}

# Initialize Hardware
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()

# ----------------------------
# Motor Functions
# ----------------------------

def read_motor(motor_id):
    """
    Reads motor state in passive mode.
    Commanding current position with zero gains ensures transparency.
    """
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    
    current_pos = data.q
    # Re-command with zero gains to stay passive
    cmd.q = current_pos
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    
    return {
        "id": motor_id,
        "position": data.q,
        "velocity": data.dq,
        "temp": data.temp,
        "error": data.merror
    }

def read_all_motors():
    return [read_motor(mid)["position"] for mid in MOTOR_IDS]

def actuate_motor(motor_id, desired_position):
    """Commands a motor with active impedance gains."""
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = desired_position
    cmd.dq = 0.0
    cmd.kp = IMPEDANCE_SETTINGS[motor_id]["kp"]
    cmd.kd = IMPEDANCE_SETTINGS[motor_id]["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

# ----------------------------
# Recording Logic
# ----------------------------
demonstration_data = []
recording = False

def record_demonstration():
    global demonstration_data, recording
    last_print_time = time.time()
    try:
        while recording:
            current_positions = read_all_motors()
            demonstration_data.append({
                "time": time.time(),
                "positions": current_positions
            })
            
            if time.time() - last_print_time >= 0.1:
                pos_str = " ".join(f"{p:.4f}" for p in current_positions)
                print(f"Recording: {pos_str}")
                last_print_time = time.time()
            
            time.sleep(0.005) # 200Hz sampling
    except Exception as e:
        print(f"Recording error: {e}")

# ----------------------------
# Smooth Motion Logic
# ----------------------------

def move_to_starting_position(target_positions, duration=3.0):
    """
    Moves the robot to the target configuration using a Sine-wave interpolation
    to ensure 0 velocity at start and end, providing smooth torque continuity.
    """
    print(f"Moving to starting position smoothly over {duration}s...")
    
    initial_positions = read_all_motors()
    start_time = time.time()
    
    # Run control loop at 100Hz for smoothness
    while True:
        elapsed = time.time() - start_time
        if elapsed > duration:
            break
            
        # alpha goes from 0.0 to 1.0
        alpha = elapsed / duration
        
        # Sine-based interpolation (Smoothstep profile)
        # alpha_smooth = 0 at start, 1 at end, with 0 derivative at both.
        alpha_smooth = 0.5 * (1 - math.cos(math.pi * alpha))
        
        for i, motor_id in enumerate(MOTOR_IDS):
            # Calculate interpolated setpoint
            pos_cmd = initial_positions[i] + (target_positions[i] - initial_positions[i]) * alpha_smooth
            actuate_motor(motor_id, pos_cmd)
            
        time.sleep(0.01) # 100Hz command frequency

    # Final check: Ensure we hold the exact target
    for i, motor_id in enumerate(MOTOR_IDS):
        actuate_motor(motor_id, target_positions[i])
    print("Start position reached.")

def playback_demonstration(demo_data):
    if not demo_data:
        print("No data to playback.")
        return

    print("Replaying path...")
    t0 = demo_data[0]["time"]
    playback_start = time.time()
    
    for sample in demo_data:
        desired_time = sample["time"] - t0
        # Sync loop to recorded timing
        while (time.time() - playback_start) < desired_time:
            time.sleep(0.001)
            
        for i, motor_id in enumerate(MOTOR_IDS):
            actuate_motor(motor_id, sample["positions"][i])
            
    print("Playback finished.")

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    try:
        # Passive initialization
        print("Initializing motors (Passive Mode)...")
        for mid in MOTOR_IDS:
            read_motor(mid)
        
        if input("Start demonstration? (y/n): ").lower() != 'y':
            sys.exit(0)

        print("\n=== Passive Recording Active ===")
        print("Guiding the arm manually. Press ENTER to stop.\n")

        recording = True
        demo_thread = threading.Thread(target=record_demonstration)
        demo_thread.start()

        input("Recording... Press ENTER to stop.")
        recording = False
        demo_thread.join()

        # Save to CSV
        csv_filename = "demonstration_data.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + [f"m{i}" for i in MOTOR_IDS])
            for s in demonstration_data:
                writer.writerow([s["time"]] + s["positions"])
        print(f"Data saved to {csv_filename}")

        # Playback sequence
        if input("\nRepeat path? (y/n): ").lower() == 'y':
            start_conf = demonstration_data[0]["positions"]
            # 1. Smoothly move to start to avoid jerks
            move_to_starting_position(start_conf, duration=4.0)
            # 2. Replay recorded trajectory
            playback_demonstration(demonstration_data)
            
            final_pos = demonstration_data[-1]["positions"]
        else:
            final_pos = read_all_motors()

        # Infinite hold
        print("\nSequence complete. Holding position. Ctrl+C to exit.")
        while True:
            for i, mid in enumerate(MOTOR_IDS):
                actuate_motor(mid, final_pos[i])
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted. Holding safety position.")
        try:
            # Hold current position one last time before exiting
            safe_pos = read_all_motors()
            for i, mid in enumerate(MOTOR_IDS):
                actuate_motor(mid, safe_pos[i])
        except:
            pass
        print("Program closed.")