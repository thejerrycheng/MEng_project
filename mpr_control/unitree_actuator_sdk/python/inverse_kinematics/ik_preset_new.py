import time
import math
import sys
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

sys.path.append('../lib')  # Adjust if needed for unitree_actuator_sdk import
from unitree_actuator_sdk import *

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
    r = math.sqrt(x * x + y * y)
    s = z - d1

    theta1 = math.atan2(y, x)
    cosT3 = (r * r + s * s - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
    cosT3 = max(-1.0, min(1.0, cosT3))  # clamp
    theta3 = math.acos(cosT3)
    sinT3 = math.sin(theta3)
    theta2 = math.atan2(s, r) - math.atan2(a3 * sinT3, a2 + a3 * cosT3)

    return theta1, theta2, theta3

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

# ================== 3) PRE-CALCULATE PATH ==================
def precompute_path(initial_position, movements, steps=50):
    x, y, z = initial_position
    joint_angle_path = []

    for dx, dy, dz in movements:
        step_x = dx / steps
        step_y = dy / steps
        step_z = dz / steps

        for _ in range(steps):
            x += step_x
            y += step_y
            z += step_z
            theta1, theta2, theta3 = compute_ik_3dof(x, y, z, dh_params)
            joint_angle_path.append((theta1, theta2, theta3))

    return joint_angle_path

motion_path = [
    (0.1, 0, 0), (-0.2, 0, 0), (0.1, 0, 0),
    (0, 0.1, 0), (0, -0.2, 0), (0, 0.1, 0),
    (0, 0, 0.1), (0, 0, -0.2), (0, 0, 0.1)
]

initial_position = (0.1, 0.1, 0.1)
joint_angle_path = precompute_path(initial_position, motion_path)

# ================== 4) MOTOR CONTROL & LOGGING ==================
kp_min = 0.5
kp_max = 1.0
kd = 0.01

log_time_steps = []
log_actual = [[], [], []]
log_desired = [[], [], []]
log_torque = [[], [], []]
time_step = 0

def send_motor_command(motor_id, desired_angle, current_position, serial, command, data):
    global log_time_steps, time_step
    position_error = abs(desired_angle - current_position)
    kp = kp_min if position_error > 0.5 else kp_max * (1 - position_error) + kp_min * position_error

    command.motorType = MotorType.GO_M8010_6
    command.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
    command.id = 0
    command.q = desired_angle
    command.dq = 0.0
    command.kp = kp
    command.kd = kd
    command.tau = 0
    serial.sendRecv(command, data)

    actual_deg = math.degrees(current_position / gear_ratios[motor_id])
    desired_deg = math.degrees(desired_angle / gear_ratios[motor_id])
    log_actual[motor_id].append(actual_deg)
    log_desired[motor_id].append(desired_deg)
    log_torque[motor_id].append(data.tau)

def execute_motion(joint_angle_path, delay=0.05):
    global time_step
    for joint_angles in joint_angle_path:
        for i in range(3):
            serial = serials[i]
            cmd = commands[i]
            dat = data[i]

            dat.motorType = MotorType.GO_M8010_6
            serial.sendRecv(cmd, dat)
            current_position = dat.q

            desired_angle = joint_angles[i] * gear_ratios[i]
            send_motor_command(i, desired_angle, current_position, serial, cmd, dat)

        log_time_steps.append(time_step)
        time_step += 1
        time.sleep(delay)

# ================== 5) MAIN EXECUTION ==================
try:
    print("Starting pre-computed motion...")
    execute_motion(joint_angle_path)
    print("Motion complete.")

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")

finally:
    for i, serial in enumerate(serials):
        cmd = MotorCmd()
        cmd.motorType = MotorType.GO_M8010_6
        cmd.q = data[i].q
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, data[i])
    print("All motors stopped.")

# ================== 6) PLOTTING ==================
print("Plotting logs (Angles, Torque, Error)...")

# Ignore the first 10 timesteps
start_index = 10

# Joint Angles vs Time
fig1, ax1 = plt.subplots()
colors = ['red', 'green', 'blue']
for i in range(3):
    ax1.plot(
        log_time_steps[start_index:], 
        log_actual[i][start_index:], 
        label=f'Motor{i+1} Actual', 
        color=colors[i]
    )
    ax1.plot(
        log_time_steps[start_index:], 
        log_desired[i][start_index:], 
        label=f'Motor{i+1} Desired', 
        linestyle='--', 
        color=colors[i]
    )
ax1.set_title('Joint Angles vs Time (Excluding First 10 Timesteps)')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Angle (deg)')
ax1.legend()
ax1.grid()

# Torque vs Time
fig2, ax2 = plt.subplots()
for i in range(3):
    ax2.plot(
        log_time_steps[start_index:], 
        log_torque[i][start_index:], 
        label=f'Motor{i+1} Torque', 
        color=colors[i]
    )
ax2.set_title('Torque vs Time (Excluding First 10 Timesteps)')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Torque (Nm)')
ax2.legend()
ax2.grid()

# Error vs Time
fig3, ax3 = plt.subplots()
for i in range(3):
    errors = [
        log_desired[i][j] - log_actual[i][j] 
        for j in range(start_index, len(log_time_steps))
    ]
    ax3.plot(
        log_time_steps[start_index:], 
        errors, 
        label=f'Motor{i+1} Error', 
        color=colors[i]
    )
ax3.set_title('Error vs Time (Excluding First 10 Timesteps)')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Angle Error (deg)')
ax3.legend()
ax3.grid()

plt.show()
