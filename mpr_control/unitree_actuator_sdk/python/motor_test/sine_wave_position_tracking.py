#!/usr/bin/env python
import time
import math
import sys
import threading
import csv
import time
import sys
import os

import time
import sys
sys.path.append('../lib')
from unitree_actuator_sdk import *


# ---------------- Configuration ----------------
PORT = '/dev/ttyUSB0'
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_ID = 0

AMP = math.pi / 2        # +/- 90 deg (rad)
FREQ = 0.3               # Hz
KP = 5.0
KD = 1
DT = 0.002               # 500 Hz

# ---------------- Init ----------------
serial = SerialPort(PORT)
cmd = MotorCmd()
data = MotorData()

t0 = time.time()

print("Starting sinusoidal position tracking (+/-90 deg)")

# ---------------- Control Loop ----------------
while True:
    t = time.time() - t0

    q_des = AMP * math.sin(2 * math.pi * FREQ * t)
    dq_des = AMP * 2 * math.pi * FREQ * math.cos(2 * math.pi * FREQ * t)

    cmd.motorType = MOTOR_TYPE
    data.motorType = MOTOR_TYPE

    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = MOTOR_ID

    cmd.q = q_des
    cmd.dq = dq_des
    cmd.kp = KP
    cmd.kd = KD
    cmd.tau = 0.0

    serial.sendRecv(cmd, data)

    print(
        "q_des: {:+.3f} | q: {:+.3f} | dq: {:+.3f} | temp: {} | err: {}".format(
            q_des, data.q, data.dq, data.temp, data.merror
        )
    )

    time.sleep(DT)
