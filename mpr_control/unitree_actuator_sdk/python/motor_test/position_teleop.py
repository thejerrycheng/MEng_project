#!/usr/bin/env python
import time
import sys
import os
import math
from pynput import keyboard

# -------------------------------------------------
# SDK import
# -------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'lib'))
sys.path.insert(0, LIB_DIR)

from unitree_actuator_sdk import *

# ---------------- Configuration ----------------
PORT = '/dev/ttyUSB0'
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_ID = 0

KP = 1.0
KD = 0.6                 # stronger damping
DT = 0.002               # 500 Hz

VEL_FAST = 0.8           # rad/s
VEL_SLOW = 0.25          # rad/s
ACC_LIMIT = 2.0          # rad/s^2
POS_LIMIT = math.pi

# ---------------- Init ----------------
serial = SerialPort(PORT)
cmd = MotorCmd()
data = MotorData()

cmd.motorType = MOTOR_TYPE
data.motorType = MOTOR_TYPE
cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
cmd.id = MOTOR_ID

# Read initial state
cmd.q = 0.0
cmd.dq = 0.0
cmd.kp = 0.0
cmd.kd = 0.0
cmd.tau = 0.0
serial.sendRecv(cmd, data)

q_cmd = data.q
dq_cmd = 0.0
dq_target = 0.0
vel_step = VEL_FAST
running = True

print("\nSafe Smooth Keyboard Teleop")
print("Initial q: {:.3f} rad".format(q_cmd))
print("UP/DOWN: move | LEFT/RIGHT: slow/fast | ESC: quit\n")

# ---------------- Keyboard ----------------
def on_press(key):
    global dq_target, vel_step, running
    try:
        if key == keyboard.Key.up:
            dq_target = vel_step
        elif key == keyboard.Key.down:
            dq_target = -vel_step
        elif key == keyboard.Key.left:
            vel_step = VEL_SLOW
        elif key == keyboard.Key.right:
            vel_step = VEL_FAST
        elif key == keyboard.Key.esc:
            running = False
            return False
    except:
        pass

def on_release(key):
    global dq_target
    if key in [keyboard.Key.up, keyboard.Key.down]:
        dq_target = 0.0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ---------------- Control Loop ----------------
while running:
    # Acceleration-limited velocity tracking
    dq_err = dq_target - dq_cmd
    dq_step = max(-ACC_LIMIT * DT, min(ACC_LIMIT * DT, dq_err))
    dq_cmd += dq_step

    # Integrate position
    q_cmd += dq_cmd * DT
    q_cmd = max(-POS_LIMIT, min(POS_LIMIT, q_cmd))

    # Hybrid PD command (CONSISTENT q, dq)
    cmd.q = q_cmd
    cmd.dq = dq_cmd
    cmd.kp = KP
    cmd.kd = KD
    cmd.tau = 0.0

    serial.sendRecv(cmd, data)

    if data.merror != 0:
        print("Motor error:", data.merror)
        break

    print(
        "q_cmd: {:+.3f} | q: {:+.3f} | dq_cmd: {:+.2f} | temp: {}".format(
            q_cmd, data.q, dq_cmd, data.temp
        )
    )

    time.sleep(DT)

# ---------------- Safe Exit ----------------
print("\nStopping motor safely...")
cmd.q = data.q
cmd.dq = 0.0
cmd.kp = 0.0
cmd.kd = 0.0
cmd.tau = 0.0
serial.sendRecv(cmd, data)

listener.stop()
print("Exited cleanly.")
