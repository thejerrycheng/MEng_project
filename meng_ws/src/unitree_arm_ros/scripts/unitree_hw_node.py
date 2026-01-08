#!/usr/bin/env python3
import sys
import rospy
from sensor_msgs.msg import JointState

# --------------------------------------------------
# Unitree SDK path
# --------------------------------------------------
UNITREE_SDK_LIB = "/home/jerry/Desktop/MEng_project/mpr_control/unitree_actuator_sdk/lib"
if UNITREE_SDK_LIB not in sys.path:
    sys.path.append(UNITREE_SDK_LIB)

from unitree_actuator_sdk import *

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_IDS = [0, 1, 2, 3, 4, 5]

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

SERIAL_PORT = "/dev/ttyUSB0"

# IMPORTANT: 6 motors * 500Hz = 3000 sendRecv/sec (often jittery)
CONTROL_RATE = 200  # Hz (start here; increase later carefully)
DT = 1.0 / CONTROL_RATE

COMMAND_TIMEOUT = 0.25  # seconds

# Smoothness knobs
# Max joint velocity (rad/s). Start conservative.
MAX_VEL = [0.6, 0.6, 0.6, 0.8, 1.0, 1.0]

# Low-pass filter for target (0..1). Smaller = smoother/slower.
# alpha ~ 0.08 @200Hz gives ~1–2 Hz-ish smoothing feel.
TARGET_LPF_ALPHA = 0.08

# Impedance gains (start lower; raise after stable)
IMPEDANCE = {
    0: {"kp": 0.4, "kd": 0.03},
    1: {"kp": 1.2, "kd": 0.06},
    2: {"kp": 1.2, "kd": 0.06},
    3: {"kp": 0.8, "kd": 0.05},
    4: {"kp": 0.6, "kd": 0.03},
    5: {"kp": 0.6, "kd": 0.03},
}

# --------------------------------------------------
class UnitreeArmHW:
    """
    Smooth + safe hardware interface:

    - Publishes:  /joint_states
    - Subscribes: /arm/command

    Behavior:
    - No fresh ROS command => READ-ONLY (no actuation commands)
    - Fresh ROS command    => smooth position impedance control (soft-start + ramp + LPF)
    """
    def __init__(self):
        rospy.init_node("unitree_hw_node")

        self.serial = SerialPort(SERIAL_PORT)
        self.cmd = MotorCmd()
        self.data = MotorData()
        self.gear_ratio = queryGearRatio(MOTOR_TYPE)

        # Measured state
        self.q = [0.0] * 6
        self.dq = [0.0] * 6

        # Command state
        self.q_des = None          # raw desired from ROS
        self.q_des_filt = None     # low-pass filtered target
        self.q_cmd = None          # ramp-limited command actually sent
        self.last_cmd_time = None
        self.was_active = False    # detect rising edge into control

        # ROS
        self.state_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        self.cmd_sub = rospy.Subscriber("/arm/command", JointState, self.command_cb, queue_size=1)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Unitree HW Node (SMOOTH + COMMAND-GATED)")
        rospy.loginfo(" CONTROL_RATE: %d Hz", CONTROL_RATE)
        rospy.loginfo(" COMMAND_TIMEOUT: %.2f s", COMMAND_TIMEOUT)
        rospy.loginfo(" No command => READ-ONLY (no actuation)")
        rospy.loginfo("==============================================")

    # --------------------------------------------------
    def command_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        if self.q_des is None:
            self.q_des = [0.0] * 6

        for i, j in enumerate(JOINT_NAMES):
            if j in name_to_idx:
                self.q_des[i] = msg.position[name_to_idx[j]]

        self.last_cmd_time = rospy.Time.now().to_sec()

    # --------------------------------------------------
    def read_motor_passive(self, motor_id):
        """
        Minimal passive read (same style as your stable read-only node).
        NOTE: sendRecv is required to get data back.
        """
        self.data.motorType = MOTOR_TYPE
        self.cmd.motorType = MOTOR_TYPE
        self.cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        self.cmd.id = motor_id

        # read
        self.serial.sendRecv(self.cmd, self.data)
        q_motor = self.data.q
        dq_motor = self.data.dq

        # re-send same position with zero gains (prevents injection)
        self.cmd.q = q_motor
        self.cmd.dq = 0.0
        self.cmd.kp = 0.0
        self.cmd.kd = 0.0
        self.cmd.tau = 0.0
        self.serial.sendRecv(self.cmd, self.data)

        return q_motor / self.gear_ratio, dq_motor / self.gear_ratio

    # --------------------------------------------------
    def send_motor_command(self, motor_index, q_cmd):
        """
        Smooth impedance command. Uses q_cmd already ramped/filtered.
        """
        motor_id = MOTOR_IDS[motor_index]

        self.data.motorType = MOTOR_TYPE
        self.cmd.motorType = MOTOR_TYPE
        self.cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        self.cmd.id = motor_id

        self.cmd.q = q_cmd * self.gear_ratio
        self.cmd.dq = 0.0
        self.cmd.kp = IMPEDANCE[motor_index]["kp"]
        self.cmd.kd = IMPEDANCE[motor_index]["kd"]
        self.cmd.tau = 0.0

        self.serial.sendRecv(self.cmd, self.data)
        return self.data.q / self.gear_ratio, self.data.dq / self.gear_ratio

    # --------------------------------------------------
    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = self.q
        msg.velocity = self.dq
        self.state_pub.publish(msg)

    # --------------------------------------------------
    def run(self):
        rate = rospy.Rate(CONTROL_RATE)

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            active = (
                self.q_des is not None
                and self.last_cmd_time is not None
                and (now - self.last_cmd_time) < COMMAND_TIMEOUT
            )

            # Always update measured state
            # If inactive => passive read only
            if not active:
                self.was_active = False
                for i, motor_id in enumerate(MOTOR_IDS):
                    q, dq = self.read_motor_passive(motor_id)
                    self.q[i] = q
                    self.dq[i] = dq
                self.publish_joint_states()
                rate.sleep()
                continue

            # Rising edge into active: soft-start from current measured q
            if active and not self.was_active:
                # ensure we have current q
                for i, motor_id in enumerate(MOTOR_IDS):
                    q, dq = self.read_motor_passive(motor_id)
                    self.q[i] = q
                    self.dq[i] = dq

                self.q_des_filt = self.q[:]   # start filtered target at current
                self.q_cmd = self.q[:]        # start command at current
                self.was_active = True

            # Active control: filter target + ramp command
            for i in range(6):
                # LPF the incoming desired target
                self.q_des_filt[i] = (1.0 - TARGET_LPF_ALPHA) * self.q_des_filt[i] + TARGET_LPF_ALPHA * self.q_des[i]

                # Velocity-limited ramp from q_cmd toward q_des_filt
                err = self.q_des_filt[i] - self.q_cmd[i]
                max_step = MAX_VEL[i] * DT
                step = max(min(err, max_step), -max_step)
                self.q_cmd[i] += step

                # Send command
                q, dq = self.send_motor_command(i, self.q_cmd[i])
                self.q[i] = q
                self.dq[i] = dq

            self.publish_joint_states()
            rate.sleep()

# --------------------------------------------------
if __name__ == "__main__":
    try:
        UnitreeArmHW().run()
    except rospy.ROSInterruptException:
        pass
