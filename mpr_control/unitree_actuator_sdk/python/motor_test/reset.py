import time
import sys
import os

# -------------------------------------------------
# Correct SDK import for your tree:
# unitree_actuator_sdk/python/motor_test/reset.py
# SDK lives in: ../../lib
# -------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'lib'))
sys.path.insert(0, LIB_DIR)

from unitree_actuator_sdk import *

# ---------------- Configuration ----------------
PORT = '/dev/ttyUSB0'
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_ID = 0

serial = SerialPort(PORT)
cmd = MotorCmd()
data = MotorData()

print("Connecting to motor...")

# ---------------- Step 1: Read current state ----------------
cmd.motorType = MOTOR_TYPE
data.motorType = MOTOR_TYPE
cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
cmd.id = MOTOR_ID

cmd.q = 0.0
cmd.dq = 0.0
cmd.kp = 0.0
cmd.kd = 0.0
cmd.tau = 0.0

serial.sendRecv(cmd, data)

print("Initial state:")
print("  q =", data.q)
print("  dq =", data.dq)
print("  temp =", data.temp)
print("  merror =", data.merror)

# ---------------- Step 2: Neutral command (clear fault) ----------------
print("Sending neutral reset command...")

for _ in range(20):
    cmd.q = data.q      # hold current position
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    time.sleep(0.01)

# ---------------- Step 3: Re-check ----------------
serial.sendRecv(cmd, data)

print("After reset:")
print("  q =", data.q)
print("  dq =", data.dq)
print("  temp =", data.temp)
print("  merror =", data.merror)

if data.merror == 0:
    print("✅ Motor reset successful.")
else:
    print("⚠️  Motor still in error state.")
    print("    -> Power-cycle motor supply to fully clear latch.")
