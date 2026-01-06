# Teach & Repeat — Path Collection & Replay (Unitree SDK)

Two scripts:

* `path_collect.py` — **drag-to-record** a joint path (freedrive, zero gains) → CSV
* `path_replay.py` — **replay** the CSV on loop with impedance control, returning to **home** each run

> Assumes Unitree SDK is at `../lib` and you use `serial.sendRecv(cmd, data)` with FOC via `queryMotorMode`.

---

## Prereqs

* Python 3.8+
* Unitree actuator SDK on `PYTHONPATH`

  ```python
  sys.path.append('../lib')
  from unitree_actuator_sdk import *
  ```
* 6 motors on `/dev/ttyUSB0` (default), IDs: `0 1 2 3 4 5`
* **Safety:** clear workspace; start with conservative gains.

---

## Quick Start Demo Scripts Under `python/demo/`

### 1) Collect a Path

```bash
mkdir -p records
python path_collect.py --out records/path1.csv --port /dev/ttyUSB0 --ids 0 1 2 3 4 5 --rate 100
# ENTER → start; move arm; ENTER → stop; 'q'+ENTER or Ctrl+C → quit & save
```

**Output CSV** (`--out` path):

```
timestamp,motor1,motor2,motor3,motor4,motor5,motor6
0.000123,0.1234,...
...
```

Angles in **radians**. (Ensure the parent folder exists.)

### 2) Replay the Path

```bash
python path_replay.py --input records/path1.csv --port /dev/ttyUSB0 --ids 0 1 2 3 4 5 --loops 0
# loops=0 → repeat forever; returns to HOME (first row) after each run
```

Optional:

```bash
--home q0 q1 q2 q3 q4 q5   # custom home (rad)
--home-time 5              # seconds to go home
```

---

## Common Flags

* `--port /dev/ttyUSB0` – serial device
* `--ids 0 1 2 3 4 5` – joint IDs (keep consistent)
* `--rate 100` – collector sampling Hz
* `--loops 0|N` – replay forever or N times
* `--home ...` / `--home-time ...` – replay homing

---

## 3) Teach & Repeat (`teach_and_repeat.py`)

Single run: **passive init → record → optional single replay → hold**.

```bash
python teach_and_repeat.py
```

* Saves `demonstration_data.csv` in CWD
* Uses your impedance table (edit in file)
* Press Ctrl+C to exit from hold

---

## Troubleshooting (quick)

* **SDK not found:** update `sys.path.append('/path/to/sdk/python')`
* **Wrong port:** use the actual device (`/dev/ttyUSB1`, `/dev/ttyACM0`, …)
* **Stiff during collect:** re-run collector (it enforces zero gains while recording)
* **Jerky replay:** lower gains, ensure reasonable collect `--rate`, clear workspace

---

## Notes

* Angles are **radians**.
* Scripts default to 6-DoF; change `--ids` if needed (match both phases).
* Tune gains cautiously for your hardware.
