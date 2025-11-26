import time
import math
import sys
import threading
from queue import Queue
sys.path.append('../lib')  # Adjust if needed for unitree_actuator_sdk import
from unitree_actuator_sdk import *

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# ================== 1) DEFINE DH PARAMS & IK FUNCTION ==================
dh_params = {
    'd1': 0.4,
    'a2': 0.3,
    'a3': 0.25,
    'd6': 0.08  # might be unused if ignoring orientation
}

def compute_ik_3dof(x, y, z, dh):
    d1 = dh['d1']
    a2 = dh['a2']
    a3 = dh['a3']
    r = math.sqrt(x*x + y*y)
    s = z - d1

    theta1 = math.atan2(y, x)
    cosT3 = (r*r + s*s - a2*a2 - a3*a3) / (2.0 * a2 * a3)
    cosT3 = max(-1.0, min(1.0, cosT3))  # clamp
    theta3 = math.acos(cosT3)
    sinT3 = math.sin(theta3)
    theta2 = math.atan2(s, r) - math.atan2(a3*sinT3, a2 + a3*cosT3)

    return (theta1, theta2, theta3)

# ================== 2) SETUP MOTORS (3 DOF) ==================
serials = [SerialPort(f'/dev/ttyUSB{i}') for i in range(3)]
commands = [MotorCmd() for _ in range(3)]
data = [MotorData() for _ in range(3)]
gear_ratios = [queryGearRatio(MotorType.GO_M8010_6)] * 3  # 3 motors => 3 gear ratios

# Read current positions as the startup positions
startup_positions = [0.0] * 3
for i, serial in enumerate(serials):
    cmd = commands[i]
    dat = data[i]
    dat.motorType = MotorType.GO_M8010_6
    serial.sendRecv(cmd, dat)
    startup_positions[i] = dat.q

# We'll keep a local array for desired positions in motor space (rad*gear_ratio).
desired_positions = startup_positions[:]

# Logging arrays
log_time_steps = []
log_actual = [[], [], []]   # actual angles (deg)
log_desired = [[], [], []]  # desired angles (deg)
log_torque = [[], [], []]   # torque reading
time_step = 0

# ================== 3) MOTOR CONTROL ==================
kp_min = 0.5    # Minimum Kp
kp_max = 10.0   # Maximum Kp
kd = 0.01       # Derivative gain

def send_motor_commands():
    """
    Send motor commands based on 'desired_positions' array
    and log data (actual angle, torque, etc.) with variable Kp.
    """
    global time_step

    for i in range(3):
        serial = serials[i]
        cmd = commands[i]
        dat = data[i]

        # Read current motor position
        dat.motorType = MotorType.GO_M8010_6
        serial.sendRecv(cmd, dat)
        current_position = dat.q  # raw angle
        current_torque = dat.tau  # or dat.tauEst if your firmware differs

        # Compute position error
        position_error = abs(desired_positions[i] - current_position)

        # Dynamic Kp
        if position_error > 0.5:  # Large error => use kp_min
            kp = kp_min
        else:
            # Weighted interpolation from kp_min to kp_max
            kp = kp_max * (1 - position_error) + kp_min * position_error

        # Build command
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        cmd.q = desired_positions[i]   # desired motor position
        cmd.dq = 0.0
        cmd.kp = kp
        cmd.kd = kd
        cmd.tau = 0

        # Send command and update data
        serial.sendRecv(cmd, dat)

        # Log actual angle, desired angle, torque
        actual_deg = math.degrees(current_position / gear_ratios[i])
        desired_deg = math.degrees(desired_positions[i] / gear_ratios[i])
        log_actual[i].append(actual_deg)
        log_desired[i].append(desired_deg)
        log_torque[i].append(current_torque)

    log_time_steps.append(time_step)
    time_step += 1

def move_endeffector(x, y, z, dx, dy, dz, steps=10):
    """
    Move from (x,y,z) to (x+dx, y+dy, z+dz) in 'steps' increments,
    calling IK at each small step and sending motor commands.
    Returns final (x,y,z).
    """
    step_x = dx / steps
    step_y = dy / steps
    step_z = dz / steps

    for _ in range(steps):
        x += step_x
        y += step_y
        z += step_z
        # IK
        t1, t2, t3 = compute_ik_3dof(x, y, z, dh_params)

        # Map angles -> motor positions
        desired_positions[0] = t1 * gear_ratios[0]
        desired_positions[1] = t2 * gear_ratios[1]
        desired_positions[2] = t3 * gear_ratios[2]

        # Send commands & log
        send_motor_commands()
        time.sleep(0.05)

    return x, y, z

try:
    print("Starting preset motion with logging (angles, torque)...")

    # 1) Initialize end-effector at (0.1, 0.1, 0.1)
    x, y, z = 0.1, 0.1, 0.1
    t1, t2, t3 = compute_ik_3dof(x, y, z, dh_params)
    desired_positions[0] = t1 * gear_ratios[0]
    desired_positions[1] = t2 * gear_ratios[1]
    desired_positions[2] = t3 * gear_ratios[2]

    # Send a few commands to let it settle
    for _ in range(10):
        send_motor_commands()
        time.sleep(0.05)

    # 2) Perform a sample path: +0.1 in X, then -0.2, then +0.1
    x, y, z = move_endeffector(x, y, z, 0.1, 0, 0)
    x, y, z = move_endeffector(x, y, z, -0.2, 0, 0)
    x, y, z = move_endeffector(x, y, z, 0.1, 0, 0)

    print("Motion path complete.")

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")

finally:
    # Safely stop motors
    for i in range(3):
        cmd = MotorCmd()
        cmd.motorType = MotorType.GO_M8010_6
        cmd.q = data[i].q  # Hold the current position
        cmd.dq = 0.0
        cmd.tau = 0.0
        serials[i].sendRecv(cmd, data[i])
    print("All motors stopped.")
