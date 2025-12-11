#!/usr/bin/env python
import time
import sys
import csv
from datetime import datetime

# Add SDK to path
sys.path.append('../lib')
from unitree_actuator_sdk import *

# ----------------------------
# Robot and Communication Setup
# ----------------------------
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Motor IDs for the 6 joints
motor_ids = [0, 1, 2, 3, 4, 5]

# Get gear ratio for position conversion
gear_ratio = queryGearRatio(MotorType.GO_M8010_6)

# Joint names for better readability
joint_names = [
    "Base (J0)",
    "Shoulder (J1)", 
    "Elbow (J2)",
    "Wrist Roll (J3)",
    "Wrist Pitch (J4)",
    "Wrist Yaw (J5)"
]

# ----------------------------
# Zero Torque Read Function
# ----------------------------
def read_motor_passive(motor_id):
    """
    Reads motor position with zero torque applied.
    Commands the current position back with zero gains to maintain passive state.
    Returns position divided by gear ratio for actual joint angle.
    """
    # Read current sensor value
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    current_position = data.q
    
    # Re-command with zero gains (no torque)
    cmd.q = current_position
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    
    # Divide by gear ratio to get actual joint angle
    joint_position = data.q / gear_ratio
    
    return {
        "id": motor_id,
        "position": joint_position,
        "velocity": data.dq / gear_ratio,
        "temperature": data.temp,
        "motor_error": data.merror
    }

def read_all_motors_passive():
    """
    Reads all motor positions with zero torque.
    Returns list of motor data dictionaries.
    """
    motors_data = []
    for motor_id in motor_ids:
        motor_data = read_motor_passive(motor_id)
        motors_data.append(motor_data)
    return motors_data

# ----------------------------
# Display Functions
# ----------------------------
def display_positions(motors_data, title="Current Joint Positions"):
    """
    Displays joint positions in a nicely formatted table.
    """
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    print(f"{'Joint':<20} {'Position (rad)':<15} {'Position (deg)':<15} {'Temp (°C)'}")
    print("-"*70)
    
    for i, motor_data in enumerate(motors_data):
        pos_rad = motor_data["position"]
        pos_deg = pos_rad * 180.0 / 3.14159
        temp = motor_data["temperature"]
        print(f"{joint_names[i]:<20} {pos_rad:>10.4f}      {pos_deg:>10.2f}      {temp:>6.1f}")
    
    print("="*70 + "\n")

def display_live_positions():
    """
    Continuously displays current positions until user is ready.
    Updates every 0.2 seconds.
    """
    print("\n" + "="*70)
    print("  LIVE POSITION MONITOR (Zero Torque Mode)")
    print("="*70)
    print("Manually move the robot arm to HOME position:")
    print("  → Arm should be extended straight up along the Z-axis")
    print("  → All joints aligned for vertical configuration")
    print("\nPress ENTER when robot is in HOME position...\n")
    
    try:
        import select
        import tty
        import termios
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            while True:
                # Read motors
                motors_data = read_all_motors_passive()
                
                # Clear previous lines (move cursor up and clear)
                print("\033[F" * (len(motor_ids) + 2), end='')
                
                # Display current positions
                print(f"{'Joint':<20} {'Position (rad)':<15} {'Position (deg)':<15}")
                print("-"*55)
                for i, motor_data in enumerate(motors_data):
                    pos_rad = motor_data["position"]
                    pos_deg = pos_rad * 180.0 / 3.14159
                    print(f"{joint_names[i]:<20} {pos_rad:>10.4f}      {pos_deg:>10.2f}")
                
                # Check if enter was pressed
                if select.select([sys.stdin], [], [], 0)[0]:
                    break
                    
                time.sleep(0.2)
                
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    except Exception as e:
        # Fallback for systems without select/termios
        print("Live update not available. Showing static display...\n")
        motors_data = read_all_motors_passive()
        display_positions(motors_data, "Current Joint Positions")
        input("Press ENTER when robot is in HOME position...")
        return read_all_motors_passive()
    
    return motors_data

# ----------------------------
# Save Calibration Data
# ----------------------------
def save_home_position(motors_data, filename="home_position.csv"):
    """
    Saves the home position to a CSV file with timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["timestamp", "joint_name", "motor_id", "position_rad", "position_deg"])
        
        # Write data for each joint
        for i, motor_data in enumerate(motors_data):
            pos_rad = motor_data["position"]
            pos_deg = pos_rad * 180.0 / 3.14159
            writer.writerow([
                timestamp,
                joint_names[i],
                motor_data["id"],
                f"{pos_rad:.6f}",
                f"{pos_deg:.2f}"
            ])
    
    print(f"✓ Home position saved to: {filename}")

# ----------------------------
# Main Calibration Routine
# ----------------------------
if __name__ == '__main__':
    try:
        print("\n" + "="*70)
        print("  ROBOT ARM CALIBRATION - HOME POSITION SETUP")
        print("="*70)
        print("\nInitializing motors in ZERO TORQUE mode...")
        
        # Initialize all motors in passive mode
        for motor_id in motor_ids:
            read_motor_passive(motor_id)
        
        print("✓ Motors initialized successfully")
        print("✓ Zero torque mode active - arm is free to move manually\n")
        
        # Show initial positions
        motors_data = read_all_motors_passive()
        display_positions(motors_data, "Initial Joint Positions")
        
        # Wait for user confirmation
        answer = input("Ready to start calibration? (y/n): ")
        if answer.strip().lower() != 'y':
            print("\nCalibration cancelled. Exiting...")
            sys.exit(0)
        
        # Live position monitoring
        print("\nStarting live position monitor...")
        time.sleep(1)
        motors_data = display_live_positions()
        
        # Record home position
        print("\n\n" + "="*70)
        print("  RECORDING HOME POSITION")
        print("="*70)
        
        # Read positions one more time to ensure accuracy
        motors_data = read_all_motors_passive()
        display_positions(motors_data, "HOME Position Captured")
        
        # Automatically save when ENTER is pressed
        save_home_position(motors_data)
        
        print("\n" + "="*70)
        print("  CALIBRATION COMPLETE")
        print("="*70)
        print("✓ Home position recorded successfully")
        print("✓ Robot remains in zero torque mode")
        print("\nYou can now:")
        print("  • Move the arm manually to test")
        print("  • Press Ctrl+C to exit the program")
        print("="*70 + "\n")
        
        # Keep motors in passive mode
        print("Maintaining zero torque mode...")
        while True:
            read_all_motors_passive()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
        print("Exiting gracefully...\n")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)