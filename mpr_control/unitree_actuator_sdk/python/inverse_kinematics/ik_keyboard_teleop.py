import time
import math
import sys
from pynput import keyboard
sys.path.append('../lib')
from unitree_actuator_sdk import *

# ================== 1) DEFINE DH PARAMS & IK FUNCTION ==================
dh_params = {
    'd1': 0.4,
    'a2': 0.3,
    'a3': 0.25,
    'd6': 0.08  # might be unused if we ignore orientation
}

def compute_ik_3dof(x, y, z, dh):
    d1 = dh['d1']
    a2 = dh['a2']
    a3 = dh['a3']
    r = math.sqrt(x*x + y*y)
    s = z - d1
    theta1 = math.atan2(y, x)
    cosT3 = (r*r + s*s - a2*a2 - a3*a3) / (2.0 * a2 * a3)
    cosT3 = max(-1.0, min(1.0, cosT3))
    theta3 = math.acos(cosT3)
    sinT3 = math.sin(theta3)
    theta2 = math.atan2(s, r) - math.atan2(a3*sinT3, a2 + a3*cosT3)
    return (theta1, theta2, theta3)

# ================== 2) SETUP MOTORS ==================
serials = [SerialPort(f'/dev/ttyUSB{i}') for i in range(3)]
commands = [MotorCmd() for _ in range(3)]
data = [MotorData() for _ in range(3)]
gear_ratios = [queryGearRatio(MotorType.GO_M8010_6)] * 6

# Read current positions as the startup positions
startup_positions = [0.0] * 6
for i, serial in enumerate(serials):
    cmd = commands[i]
    dat = data[i]
    dat.motorType = MotorType.GO_M8010_6
    serial.sendRecv(cmd, dat)
    startup_positions[i] = dat.q

# ================== 3) END-EFFECTOR CARTESIAN CONTROL ==================
end_effector_pos = [0.5, 0.5, 0.1]  # (x, y, z) start at (0.5, 0.5, 0.1)
cart_step = 0.02  # 2 cm step each arrow press or W/S

# We'll keep orientation fixed => J4,J5,J6 = 0 rad
def set_j456_zero(des_pos):
    # des_pos[3], des_pos[4], des_pos[5] = 0
    des_pos[3] = 0.0
    des_pos[4] = 0.0
    des_pos[5] = 0.0

# Initialize desired positions (motor space) to the startup
desired_positions = startup_positions[:]

# We'll define 6 DOF, but only motors 1..3 use IK. Motors 4..6 remain at 0 rad
# once we compute the IK angles, we map them to motor space
def updateMotorsFromCartesian(x, y, z):
    # Solve for (theta1, theta2, theta3) in radians
    t1, t2, t3 = compute_ik_3dof(x, y, z, dh_params)

    # Store in desired_positions. J4..J6 = 0 rad
    desired_positions[0] = t1 * gear_ratios[0]
    desired_positions[1] = t2 * gear_ratios[1]
    desired_positions[2] = t3 * gear_ratios[2]
    desired_positions[3] = 0.0
    desired_positions[4] = 0.0
    desired_positions[5] = 0.0

    print(f"\nNew IK solution => J1={math.degrees(t1):.2f}°, J2={math.degrees(t2):.2f}°, J3={math.degrees(t3):.2f}°")


# ================== 4) KEYBOARD HANDLING ==================
key_states = {
    "up": False, "down": False,
    "left": False, "right": False,
    "w": False, "s": False
}

key_map = {
    "up":    keyboard.Key.up,
    "down":  keyboard.Key.down,
    "left":  keyboard.Key.left,
    "right": keyboard.Key.right,
    "w":     keyboard.KeyCode(char='w'),
    "s":     keyboard.KeyCode(char='s')
}

def on_press(key):
    for action, mapped_key in key_map.items():
        if key == mapped_key:
            key_states[action] = True

def on_release(key):
    for action, mapped_key in key_map.items():
        if key == mapped_key:
            key_states[action] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Use arrow keys to move end-effector in X/Y plane:")
print("  UP   => +X")
print("  DOWN => -X")
print("  LEFT => +Y")
print("  RIGHT=> -Y")
print("Use W/S to move end-effector along Z axis:")
print("  W => +Z")
print("  S => -Z")
print("Press Ctrl+C to exit.\n")

# Move to initial position via IK
updateMotorsFromCartesian(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2])

kp_min, kp_max, kd = 0.1, 1.0, 0.01

# ================== 5) MAIN LOOP ==================
try:
    while True:
        # 1) Check key states => adjust (x,y,z)
        if key_states["up"]:
            end_effector_pos[0] += cart_step  # +X
        if key_states["down"]:
            end_effector_pos[0] -= cart_step  # -X
        if key_states["left"]:
            end_effector_pos[1] += cart_step  # +Y
        if key_states["right"]:
            end_effector_pos[1] -= cart_step  # -Y
        if key_states["w"]:
            end_effector_pos[2] += cart_step  # +Z
        if key_states["s"]:
            end_effector_pos[2] -= cart_step  # -Z

        # 2) Recompute IK => update desired_positions
        updateMotorsFromCartesian(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2])

        # 3) Send commands to each motor
        for i in range(3):
            serial = serials[i]
            cmd = commands[i]
            dat = data[i]

            # read current motor pos
            dat.motorType = MotorType.GO_M8010_6
            serial.sendRecv(cmd, dat)
            current_position = dat.q

            # compute position error => dynamic kp
            position_error = abs(desired_positions[i] - current_position)
            kp = kp_min if position_error > 1.0 else (kp_max * (1 - position_error) + kp_min * position_error)

            # build command
            cmd.motorType = MotorType.GO_M8010_6
            cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
            cmd.id = 0
            cmd.q = desired_positions[i]
            cmd.dq = 0.0
            cmd.kp = kp
            cmd.kd = kd
            cmd.tau = 0

            # send
            serial.sendRecv(cmd, dat)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")
finally:
    # stop all motors
    for i, serial in enumerate(serials):
        cmd = commands[i]
        dat = data[i]
        cmd.q = dat.q  # hold current pos
        cmd.dq = 0.0
        cmd.tau = 0.0
        serial.sendRecv(cmd, dat)
    print("All motors stopped.")
