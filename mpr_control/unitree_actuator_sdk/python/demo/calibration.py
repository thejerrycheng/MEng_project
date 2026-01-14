#!/usr/bin/env python
import time
import sys
import csv
import os
import threading

# ----------------------------
# SDK Setup
# ----------------------------
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
CALIB_FILE = "calibration_data.csv"

# Hardware Init
serial = SerialPort(SERIAL_PORT)
cmd = MotorCmd()
data = MotorData()
GEAR = queryGearRatio(MOTOR_TYPE)

# ----------------------------
# Functions
# ----------------------------
def read_motor_raw(motor_id):
    """
    Reads motor output shaft angle (Radians).
    Raw Motor Angle / GearRatio
    """
    data.motorType = MOTOR_TYPE
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    
    # Send/Recv
    serial.sendRecv(cmd, data)
    current_raw = data.q
    
    # Keep alive (Passive)
    cmd.q = current_raw
    cmd.dq, cmd.kp, cmd.kd, cmd.tau = 0.0, 0.0, 0.0, 0.0
    serial.sendRecv(cmd, data)
    
    # Return angle at output shaft
    return data.q / GEAR

def read_all_raw():
    return [read_motor_raw(mid) for mid in MOTOR_IDS]

# ----------------------------
# Main
# ----------------------------
def main():
    print("==============================================")
    print(f" DIFFERENTIAL WRIST CALIBRATION ")
    print(f" Gear Ratio: {GEAR}")
    print("==============================================")
    print("INSTRUCTIONS:")
    print("1) Move Robot to HOME:")
    print("   - Arm vertical")
    print("   - Wrist Pitch = 0 (Aligned with arm)")
    print("   - Wrist Roll  = 0 (facing forward/default)")
    print("2) Press ENTER to save RAW motor offsets.")
    print("==============================================")

    # Monitor Thread
    stop_monitor = False
    def monitor_loop():
        while not stop_monitor:
            try:
                qs = read_all_raw()
                # Show raw motor values
                txt = " | ".join([f"M{i}:{q:6.2f}" for i, q in zip(MOTOR_IDS, qs)])
                print(f"\rRaw Motors: {txt}", end='', flush=True)
                time.sleep(0.1)
            except: pass
            
    t = threading.Thread(target=monitor_loop)
    t.start()

    try:
        input("\n\n>>> Press ENTER to Capture Home <<<\n")
    except KeyboardInterrupt:
        stop_monitor = True
        t.join()
        sys.exit(0)

    stop_monitor = True
    t.join()

    # Capture Average
    print("\nCapturing...")
    avg_offsets = [0.0] * 6
    SAMPLES = 20
    for _ in range(SAMPLES):
        qs = read_all_raw()
        for i in range(6): avg_offsets[i] += qs[i]
        time.sleep(0.01)
    
    avg_offsets = [x/SAMPLES for x in avg_offsets]

    # Save
    with open(CALIB_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["motor_id", "zero_offset_rad"])
        for i, mid in enumerate(MOTOR_IDS):
            writer.writerow([mid, avg_offsets[i]])
            
    print(f"\n[Success] Offsets saved to {CALIB_FILE}")
    print("Offsets (Raw Radians):", [round(x,3) for x in avg_offsets])

if __name__ == "__main__":
    main()