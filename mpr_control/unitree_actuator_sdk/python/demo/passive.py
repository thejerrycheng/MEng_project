import time
import math
import sys
import threading
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *
# Initialize the serial port and command/data structures.
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the daisy-chained motors (6 motors).
motor_ids = [0, 1, 2, 3, 4, 5]

def read_motor(motor_id):
    """
    Reads the motor state in passive mode.
    No active control is applied since kp, kd, and tau are set to zero.
    """
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    # Use FOC mode but command zero corrective gains.
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    # Command a reference value of 0 with no torque output.
    cmd.q = 0.0  
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    # Send command and receive motor data.
    serial.sendRecv(cmd, data)
    return {
         "id": motor_id,
         "position": data.q,
         "velocity": data.dq,
         "temperature": data.temp,
         "motor_error": data.merror
    }

print("Logging motor joint positions in passive mode (no torque output).")
print("Press Ctrl+C to exit.")

last_print_time = time.time()

try:
    while True:
        motor_info_list = []
        # Read state for each motor.
        for motor_id in motor_ids:
            info = read_motor(motor_id)
            motor_info_list.append(info)

        # Print summary every 0.1 seconds.
        current_time = time.time()
        if current_time - last_print_time >= 0.1:
            print("\n--- Motor Status Summary ---")
            print(f"{'Motor':>5} | {'Position (rad)':>15} | {'Velocity':>10} | {'Temp':>6} | {'Error':>6}")
            print("-" * 60)
            for info in motor_info_list:
                print(f"{info['id']:>5} | {info['position']:>15.4f} | {info['velocity']:>10.4f} | "
                      f"{info['temperature']:>6.2f} | {info['motor_error']:>6.2f}")
            last_print_time = current_time

        time.sleep(0.005)  # Small delay to prevent high CPU usage.

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully...")

finally:
    # Optionally, set all motors to a safe passive state by sending zero commands.
    for motor_id in motor_ids:
        cmd.id = motor_id
        cmd.q = 0.0
        cmd.dq = 0.0
        cmd.kp = 0.0
        cmd.kd = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)
    print("All motors set to passive state (no torque output).")
