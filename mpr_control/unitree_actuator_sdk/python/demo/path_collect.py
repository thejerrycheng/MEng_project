#!/usr/bin/env python3
import time, sys, csv, threading, argparse

# Match your repo layout (same as your demo):
sys.path.append('../lib')
from unitree_actuator_sdk import *  # SerialPort, MotorCmd, MotorData, MotorType, MotorMode, queryMotorMode

# ----------------------------
# Config
# ----------------------------
DEFAULT_IDS = [0, 1, 2, 3, 4, 5]
PORT_DEFAULT = '/dev/ttyUSB0'

# ----------------------------
# Low-level helpers (SDK pattern same as your script)
# ----------------------------
def foc_mode_cmd(cmd):
    cmd.motorType = MotorType.GO_M8010_6
    cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)

def read_motor_passive(serial, cmd, data, motor_id):
    """
    Passive read: read sensors, then re-command the same q with zero gains (no torque).
    """
    data.motorType = MotorType.GO_M8010_6
    foc_mode_cmd(cmd)
    cmd.id = motor_id
    serial.sendRecv(cmd, data)
    current_q = data.q

    # Re-command current pose with zero gains to keep it passive
    cmd.q = current_q
    cmd.dq = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    cmd.tau = 0.0
    serial.sendRecv(cmd, data)
    return data.q

def read_all_positions_passive(serial, cmd, data, motor_ids):
    positions = []
    for mid in motor_ids:
        q = read_motor_passive(serial, cmd, data, mid)
        positions.append(q)
    return positions

# ----------------------------
# Recorder
# ----------------------------
class PathCollector:
    def __init__(self, port, motor_ids, sample_dt, out_csv):
        self.serial = SerialPort(port)
        self.cmd = MotorCmd()
        self.data = MotorData()
        self.motor_ids = motor_ids
        self.sample_dt = sample_dt
        self.out_csv = out_csv

        self._run = True
        self._recording = False
        self._buf = []  # dicts with "time" and "positions"
        self._t0 = None

    def _sample_once(self):
        qs = read_all_positions_passive(self.serial, self.cmd, self.data, self.motor_ids)
        if self._recording:
            t_rel = time.time() - self._t0
            self._buf.append({"time": t_rel, "positions": qs})
        return qs

    def sampler_thread(self):
        print(f"[INFO] Passive mode enabled. ENTER to start/stop recording, 'q'+ENTER to quit.")
        t_next = time.time()
        last_print = 0.0
        while self._run:
            qs = self._sample_once()
            now = time.time()
            if now - last_print >= 0.1:  # readable terminal update
                print("q(rad): " + "  ".join(f"{v:+.4f}" for v in qs))
                last_print = now
            t_next += self.sample_dt
            time.sleep(max(0.0, t_next - time.time()))
            if t_next < time.time():
                t_next = time.time()

    def toggle(self):
        if not self._recording:
            self._buf.clear()
            self._t0 = time.time()
            self._recording = True
            print("[REC] START")
        else:
            self._recording = False
            print(f"[REC] STOP — {len(self._buf)} samples captured")

    def save(self):
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["timestamp"] + [f"motor{i+1}" for i in range(len(self.motor_ids))]
            w.writerow(header)
            for sample in self._buf:
                row = [sample["time"]] + sample["positions"]
                w.writerow(row)
        print(f"[SAVE] Wrote {len(self._buf)} samples → {self.out_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Drag-to-record joint path (freedrive) to CSV.")
    ap.add_argument('--out', required=True, help='Output CSV file, e.g., path1.csv')
    ap.add_argument('--port', default=PORT_DEFAULT)
    ap.add_argument('--ids', nargs='+', type=int, default=DEFAULT_IDS)
    ap.add_argument('--rate', type=float, default=100.0, help='Sampling rate (Hz)')
    return ap.parse_args()

def main():
    args = parse_args()
    collector = PathCollector(args.port, args.ids, 1.0/max(args.rate,1e-6), args.out)

    # Initialize passive (zero-gain) by one read per motor
    print("[INIT] Setting passive/zero-gain state for all motors...")
    for mid in args.ids:
        read_motor_passive(collector.serial, collector.cmd, collector.data, mid)
    print("[READY] You can now backdrive the arm safely.")

    t = threading.Thread(target=collector.sampler_thread, daemon=True)
    t.start()

    try:
        while True:
            s = input()
            if s.strip().lower() == 'q':
                break
            collector.toggle()
    except KeyboardInterrupt:
        pass
    finally:
        collector._run = False
        t.join(timeout=1.0)
        if collector._buf:
            collector.save()
        print("[EXIT] Bye.")

if __name__ == '__main__':
    main()
