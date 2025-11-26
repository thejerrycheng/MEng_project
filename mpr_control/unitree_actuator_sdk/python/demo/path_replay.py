#!/usr/bin/env python3
import time, sys, csv, math, argparse

# Match your repo layout (same as your demo):
sys.path.append('../lib')
from unitree_actuator_sdk import *  # SerialPort, MotorCmd, MotorData, MotorType, MotorMode, queryMotorMode

DEFAULT_IDS = [0, 1, 2, 3, 4, 5]
PORT_DEFAULT = '/dev/ttyUSB0'

# Per-joint impedance (kp, kd) — from your example
impedance_settings = {
    0: {"kp": 1.0, "kd": 0.10},
    1: {"kp": 3.7, "kd": 0.200},
    2: {"kp": 2.5, "kd": 0.150},
    3: {"kp": 0.7, "kd": 0.035},
    4: {"kp": 0.3, "kd": 0.025},
    5: {"kp": 0.3, "kd": 0.025},
}

# ----------------------------
# Low-level helpers (SDK pattern same as your script)
# ----------------------------
def foc_mode_cmd(cmd):
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)

def read_motor(serial, cmd, data, motor_id):
    """
    Passive read with zero gains (same idea as your read_motor()).
    """
    data.motorType = MotorType.GO_M8010_6
    foc_mode_cmd(cmd)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)          # read sensors into 'data'
    current_position = data.q
    # re-command same pose with zero gains
    cmd.q = current_position
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

def read_all(serial, cmd, data, motor_ids):
    return [read_motor(serial, cmd, data, mid) for mid in motor_ids]

def actuate_motor(serial, cmd, data, motor_id, q_des):
    """
    Active actuation with your kp/kd per joint.
    """
    data.motorType = MotorType.GO_M8010_6
    foc_mode_cmd(cmd)
    cmd.id = motor_id
    cmd.q = q_des
    cmd.dq = 0.0
    gains = impedance_settings.get(motor_id, {"kp": 0.5, "kd": 0.02})
    cmd.kp = gains["kp"]
    cmd.kd = gains["kd"]
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

def move_toward(serial, cmd, data, motor_ids, q_target, seconds=2.0):
    """
    Gentle linear move from current pose to q_target within 'seconds'.
    """
    q_now = read_all(serial, cmd, data, motor_ids)
    steps = max(1, int(seconds / 0.05))  # ~20 Hz stepping
    for s in range(1, steps + 1):
        a = s / steps
        q_cmd = [(1 - a) * q_now[i] + a * q_target[i] for i in range(len(motor_ids))]
        for i, mid in enumerate(motor_ids):
            actuate_motor(serial, cmd, data, mid, q_cmd[i])
        time.sleep(0.05)

# ----------------------------
# CSV handling
# ----------------------------
def load_path_csv(path, expect_joints):
    """
    Accepts header from your recorder: 'timestamp, motor1..motor6'
    Also tolerates 'time' and 'q0..q5' variants.
    Returns (times, qs_list), where times start at 0.
    """
    with open(path, 'r') as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]

    # find time column
    t_idx = None
    time_names = {'timestamp', 'time', 't', 't_ros'}
    for i, name in enumerate(header):
        if name.strip().lower() in time_names:
            t_idx = i; break
    if t_idx is None:
        raise ValueError("No time column found in CSV header.")

    # find joint columns
    q_indices = []
    # support 'motor1..motor6' (your example) or 'q0..q5'
    name_to_idx = {name.strip().lower(): i for i, name in enumerate(header)}
    # try motor1..motorN first
    for j in range(1, expect_joints + 1):
        key = f"motor{j}"
        if key in name_to_idx:
            q_indices.append(name_to_idx[key])
    if len(q_indices) != expect_joints:
        q_indices = []
        for j in range(expect_joints):
            key = f"q{j}"
            if key in name_to_idx:
                q_indices.append(name_to_idx[key])
    if len(q_indices) != expect_joints:
        raise ValueError("Could not find all joint columns (motor1.. or q0..).")

    ts = []
    qs = []
    for row in rows:
        t = float(row[t_idx])
        qrow = [float(row[idx]) for idx in q_indices]
        ts.append(t); qs.append(qrow)

    # normalize time to start at 0
    t0 = ts[0]
    ts = [t - t0 for t in ts]
    return ts, qs

# ----------------------------
# Playback core
# ----------------------------
def playback_once(serial, cmd, data, motor_ids, ts, qs):
    if not qs:
        return
    # smoothly move to the first waypoint
    print("[PLAY] Moving to start...")
    move_toward(serial, cmd, data, motor_ids, qs[0], seconds=2.0)
    time.sleep(0.3)

    print("[PLAY] Following recorded timing...")
    t0 = time.time()
    for i, q_des in enumerate(qs):
        desired_t = ts[i]
        # wait until it's time for this sample
        while True:
            dt = time.time() - t0
            if dt >= desired_t - 1e-4:
                break
            time.sleep(0.001)
        # command all joints for this sample
        for j, mid in enumerate(motor_ids):
            actuate_motor(serial, cmd, data, mid, q_des[j])
        time.sleep(0.005)  # light pacing

def parse_args():
    ap = argparse.ArgumentParser(description="Replay a recorded joint path (loop with return-to-home).")
    ap.add_argument('--input', required=True, help='CSV path from path_collect.py / your demo')
    ap.add_argument('--port', default=PORT_DEFAULT)
    ap.add_argument('--ids', nargs='+', type=int, default=DEFAULT_IDS)
    ap.add_argument('--loops', type=int, default=0, help='0 = loop forever; N = run N times')
    ap.add_argument('--home', nargs='+', type=float, help='Optional home joint (6 values, radians)')
    ap.add_argument('--home-time', type=float, default=4.0, help='Seconds to go home between loops')
    return ap.parse_args()

def main():
    args = parse_args()
    serial = SerialPort(args.port)
    cmd = MotorCmd()
    data = MotorData()

    print("[INIT] Zero-gain passive read to initialize...")
    for mid in args.ids:
        read_motor(serial, cmd, data, mid)
    print("[READY] Loaded motors; reading CSV...")

    ts, qs = load_path_csv(args.input, expect_joints=len(args.ids))
    home = args.home if args.home is not None else qs[0]

    loop_idx = 0
    try:
        while True:
            loop_idx += 1
            print(f"\n=== LOOP {loop_idx} ===")
            playback_once(serial, cmd, data, args.ids, ts, qs)
            print("[PLAY] Done. Returning HOME...")
            move_toward(serial, cmd, data, args.ids, home, seconds=args.home_time)
            time.sleep(0.3)

            if args.loops > 0 and loop_idx >= args.loops:
                print("[DONE] Completed requested loops.")
                break
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
    finally:
        # Soft hold current pose (optional): command current q with small kp
        current_q = read_all(serial, cmd, data, args.ids)
        print("[HOLD] Holding current pose (Ctrl+C again to exit).")
        try:
            while True:
                for i, mid in enumerate(args.ids):
                    # gentle hold
                    data.motorType = MotorType.GO_M8010_6
                    foc_mode_cmd(cmd)
                    cmd.id = mid
                    cmd.q = current_q[i]
                    cmd.dq = 0.0
                    cmd.kp = 0.3
                    cmd.kd = 0.02
                    cmd.tau = 0.0
                    serial.sendRecv(cmd, data)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[EXIT] Bye.")

if __name__ == '__main__':
    main()
