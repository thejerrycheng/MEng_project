#!/usr/bin/env python3
"""
IRIS Cinema Arm — Simple Mode Console

Modes (press ENTER to open the menu, then type a number):
  1) Position Hold        - capture current pose and hold it with impedance control
  2) No Torque (Free Drag)- zero gains; manually move the arm
  3) Go Home (Slow Linear)- move all joints linearly & slowly to HOME, then hold there
  0) Quit

Edit HOME_POSITIONS and HOME_SPEED_RAD_S to taste.
"""

import sys
import time
import threading
from typing import List

# ----------------------------
# Robot SDK
# ----------------------------
sys.path.append('../lib')
from unitree_actuator_sdk import *  # noqa

# ----------------------------
# Config
# ----------------------------
# Motor IDs for the 6 joints
motor_ids = [0, 1, 2, 3, 4, 5]

# Impedance gains for active positioning/holding
impedance_settings = {
    0: {"kp": 1.0, "kd": 0.10},
    1: {"kp": 3.7, "kd": 0.200},
    2: {"kp": 2.5, "kd": 0.150},
    3: {"kp": 0.7, "kd": 0.035},
    4: {"kp": 0.3, "kd": 0.025},
    5: {"kp": 0.3, "kd": 0.025},
}

# Define a safe HOME pose (radians). Adjust for your rig.
HOME_POSITIONS: List[float] = [0.0, -0.6, 1.2, -0.8, 0.0, 0.0]

# Max joint speed during Go Home (rad/s) — small for steady, slow motion
HOME_SPEED_RAD_S = 0.02

# Control period (s)
CTRL_DT = 0.01

# ----------------------------
# Low-level setup
# ----------------------------
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

def _setup_cmd_for(motor_id: int):
    data.motorType = MotorType.GO_M8010_6
    cmd.motorType  = MotorType.GO_M8010_6
    cmd.mode       = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    cmd.id         = motor_id

def read_motor(motor_id: int):
    """
    Passive read: echo current position with zero gains to avoid torque.
    """
    _setup_cmd_for(motor_id)
    serial.sendRecv(cmd, data)   # read
    q = data.q

    cmd.q   = q
    cmd.dq  = 0.0
    cmd.kp  = 0.0
    cmd.kd  = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)

    return data.q, data.dq

def read_all_positions() -> List[float]:
    return [read_motor(m_id)[0] for m_id in motor_ids]

def actuate_motor(motor_id: int, q_des: float):
    _setup_cmd_for(motor_id)
    cmd.q   = q_des
    cmd.dq  = 0.0
    cmd.kp  = impedance_settings[motor_id]["kp"]
    cmd.kd  = impedance_settings[motor_id]["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

def actuate_all(q_des_list: List[float]) -> List[float]:
    measured = []
    for i, m_id in enumerate(motor_ids):
        measured.append(actuate_motor(m_id, q_des_list[i]))
    return measured

def zero_torque_step():
    """
    One control tick of zero-torque mode: for each joint, read then re-command with zero gains.
    """
    for m_id in motor_ids:
        _setup_cmd_for(m_id)
        serial.sendRecv(cmd, data)  # read
        cmd.q   = data.q
        cmd.dq  = 0.0
        cmd.kp  = 0.0
        cmd.kd  = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data)

# ----------------------------
# Utilities
# ----------------------------
class EnterWaiter:
    """Background ENTER listener to break out of a mode."""
    def __init__(self, prompt: str):
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._wait, args=(prompt,), daemon=True)

    def _wait(self, prompt: str):
        try:
            input(prompt)
        except EOFError:
            pass
        self._stop.set()

    def start(self):
        self._thr.start()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

# ----------------------------
# Modes
# ----------------------------
def mode_hold_position():
    """
    Capture the current pose and hold it with impedance control.
    Press ENTER to exit back to the main menu.
    """
    target = read_all_positions()
    print("\n[HOLD] Captured current pose. Holding. Press ENTER to stop.\n")
    waiter = EnterWaiter("")
    waiter.start()
    try:
        while not waiter.stop_requested():
            actuate_all(target)
            time.sleep(CTRL_DT)
    except KeyboardInterrupt:
        pass
    print("[HOLD] Exiting hold mode.\n")

def mode_free_drag():
    """
    Zero torque mode: sends zero gains continuously so the arm can be dragged freely.
    Press ENTER to exit back to the main menu.
    """
    print("\n[FREE-DRAG] Zero-torque enabled. Gently move the arm by hand.")
    print("Press ENTER to stop.\n")
    waiter = EnterWaiter("")
    waiter.start()
    try:
        while not waiter.stop_requested():
            zero_torque_step()
            time.sleep(CTRL_DT)
    except KeyboardInterrupt:
        pass
    print("[FREE-DRAG] Exiting free-drag mode.\n")

def mode_go_home_linear():
    """
    Move all joints linearly & slowly to HOME, then hold there with impedance.
    """
    # Read start pose
    q_start = read_all_positions()
    q_home  = HOME_POSITIONS[:]

    # Compute max distance to determine total duration for constant speed
    import math
    distances = [abs(q_home[i] - q_start[i]) for i in range(len(motor_ids))]
    max_dist  = max(distances) if distances else 0.0
    if max_dist < 1e-6:
        print("\n[GO HOME] Already at HOME. Holding HOME.\n")
        _hold_pose(q_home)
        return

    # Duration so that the *slowest* joint (largest move) runs at HOME_SPEED_RAD_S
    duration = max_dist / max(HOME_SPEED_RAD_S, 1e-4)
    t0 = time.time()

    print(f"\n[GO HOME] Linear, synchronized motion to HOME over ~{duration:.1f}s.")
    print("Press ENTER to abort and return to menu (will stop the motion).\n")
    waiter = EnterWaiter("")
    waiter.start()

    try:
        while True:
            if waiter.stop_requested():
                print("[GO HOME] Aborted by user. Returning to menu.")
                return

            t = time.time() - t0
            s = min(1.0, t / duration)  # linear blend factor
            q_des = [q_start[i] + s * (q_home[i] - q_start[i]) for i in range(len(motor_ids))]
            actuate_all(q_des)

            if s >= 1.0:
                break

            time.sleep(CTRL_DT)

        print("[GO HOME] Reached HOME. Holding position.\n")
        _hold_pose(q_home)

    except KeyboardInterrupt:
        print("\n[GO HOME] Interrupted. Returning to menu.\n")

def _hold_pose(q_target: List[float]):
    """
    Helper: hold a given pose with impedance until ENTER.
    """
    waiter = EnterWaiter("[HOLD @ HOME] Press ENTER to stop holding and return to menu...\n")
    waiter.start()
    try:
        while not waiter.stop_requested():
            actuate_all(q_target)
            time.sleep(CTRL_DT)
    except KeyboardInterrupt:
        pass
    print("[HOLD @ HOME] Done.\n")

# ----------------------------
# Main console loop
# ----------------------------
def main():
    # Initialize in passive state once
    print("Initializing motors in passive (zero-torque echo) state...")
    for m in motor_ids:
        read_motor(m)
    print("Ready.\n")

    while True:
        try:
            input("Press ENTER to open the mode menu...")
            print("\nSelect a mode:")
            print("  1) Position Hold (capture & hold current pose)")
            print("  2) No Torque (Free Drag)")
            print("  3) Go Home (Linear & Slow), then Hold")
            print("  0) Quit")
            choice = input("Enter number: ").strip()

            if choice == "1":
                mode_hold_position()
            elif choice == "2":
                mode_free_drag()
            elif choice == "3":
                mode_go_home_linear()
            elif choice == "0":
                print("Exiting. Bye!")
                break
            else:
                print("Invalid selection.\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break

if __name__ == "__main__":
    main()
