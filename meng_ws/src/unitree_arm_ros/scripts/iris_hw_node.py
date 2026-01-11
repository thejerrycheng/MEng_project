#!/usr/bin/env python3
import sys
import time
import rospy
from sensor_msgs.msg import JointState

# --------------------------------------------------
# Unitree SDK import
# --------------------------------------------------
UNITREE_SDK_LIB = "/home/jerry/Desktop/MEng_project/mpr_control/unitree_actuator_sdk/lib"
if UNITREE_SDK_LIB not in sys.path:
    sys.path.append(UNITREE_SDK_LIB)

from unitree_actuator_sdk import *

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SERIAL_PORT = "/dev/ttyUSB0"
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_IDS = [0,1,2,3,4,5]

JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","joint_5","joint_6"
]

NUM_MOTORS = 6
GEAR = queryGearRatio(MOTOR_TYPE)

CONTROL_RATE = 200       # Hz
DT = 1.0 / CONTROL_RATE
USB_GUARD_SLEEP = 0.0005

COMMAND_TIMEOUT = 0.25   # seconds

# Max joint velocity (rad/s)
MAX_VEL = [0.6,0.6,0.6,0.8,1.0,1.0]

# Target low-pass filter
TARGET_LPF_ALPHA = 0.08

# Joint impedance gains
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
    def __init__(self):
        rospy.init_node("unitree_arm_hw_node")

        self.serial = SerialPort(SERIAL_PORT)
        self.cmd  = MotorCmd()
        self.data = MotorData()

        # Joint state buffers
        self.q  = [0.0]*NUM_MOTORS
        self.dq = [0.0]*NUM_MOTORS

        # Command buffers
        self.q_des = None
        self.q_des_filt = None
        self.q_cmd = None
        self.last_cmd_time = None

        # ROS I/O
        self.state_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        rospy.Subscriber("/arm/command", JointState, self.command_cb, queue_size=1)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Unitree Arm Hardware Node")
        rospy.loginfo(" Publishes : /joint_states")
        rospy.loginfo(" Subscribes: /arm/command")
        rospy.loginfo(" Control Rate: %d Hz", CONTROL_RATE)
        rospy.loginfo("==============================================")

        # --------------------------------------------------
        # SDK Warm-up Phase
        # --------------------------------------------------
        rospy.loginfo("Connecting to Unitree SDK...")
        rospy.loginfo("Performing passive warm-up stream...")

        t0 = time.time()
        while time.time() - t0 < 1.0:   # 1 second warmup
            for mid in MOTOR_IDS:
                self.sdk_cycle(mid, passive=True)
            time.sleep(0.002)

        rospy.loginfo("SDK stream synchronized.")
        rospy.loginfo("Motors ready for control.")

    # --------------------------------------------------
    def command_cb(self, msg):
        name_to_idx = {n:i for i,n in enumerate(msg.name)}

        if self.q_des is None:
            self.q_des = [0.0]*NUM_MOTORS

        for i,n in enumerate(JOINT_NAMES):
            if n in name_to_idx:
                self.q_des[i] = msg.position[name_to_idx[n]]

        self.last_cmd_time = rospy.Time.now().to_sec()

    # --------------------------------------------------
    def sdk_cycle(self, motor_id, passive=False, q_cmd=None, motor_index=None):
        """
        Exactly one Unitree sendRecv transaction.
        """

        self.data.motorType = MOTOR_TYPE
        self.cmd.motorType  = MOTOR_TYPE
        self.cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        self.cmd.id   = motor_id

        if passive or q_cmd is None:
            # Transparent passive frame
            self.cmd.q   = self.data.q
            self.cmd.dq  = 0.0
            self.cmd.kp  = 0.0
            self.cmd.kd  = 0.0
        else:
            # Active impedance command
            self.cmd.q   = q_cmd * GEAR
            self.cmd.dq  = 0.0
            self.cmd.kp  = IMPEDANCE[motor_index]["kp"]
            self.cmd.kd  = IMPEDANCE[motor_index]["kd"]

        self.cmd.tau = 0.0

        # ---- single clean transaction ----
        self.serial.sendRecv(self.cmd, self.data)

        # Return joint-side state
        return self.data.q / GEAR, self.data.dq / GEAR, self.data.merror

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
                self.q_des is not None and
                self.last_cmd_time is not None and
                (now - self.last_cmd_time) < COMMAND_TIMEOUT
            )

            # First time entering active → initialize filters
            if active and self.q_cmd is None:
                self.q_des_filt = self.q[:]
                self.q_cmd = self.q[:]
                rospy.loginfo("Received first command → entering active control mode.")

            # --------------------------------------------------
            # Main SDK streaming
            # --------------------------------------------------
            for i, mid in enumerate(MOTOR_IDS):

                # ----- Passive mode -----
                if not active:
                    q, dq, err = self.sdk_cycle(mid, passive=True)

                # ----- Active mode -----
                else:
                    # Low-pass filter target
                    self.q_des_filt[i] = (1.0 - TARGET_LPF_ALPHA)*self.q_des_filt[i] + TARGET_LPF_ALPHA*self.q_des[i]

                    # Velocity-limited ramp
                    err_q = self.q_des_filt[i] - self.q_cmd[i]
                    max_step = MAX_VEL[i]*DT
                    step = max(min(err_q, max_step), -max_step)
                    self.q_cmd[i] += step

                    q, dq, err = self.sdk_cycle(mid, passive=False, q_cmd=self.q_cmd[i], motor_index=i)

                # Store state
                self.q[i] = q
                self.dq[i] = dq

                # Error check
                if err != 0:
                    rospy.logerr("Motor %d error %d → power-cycle required", mid, err)
                    rospy.signal_shutdown("Motor fault")

            # Publish ROS joint states
            self.publish_joint_states()

            time.sleep(USB_GUARD_SLEEP)
            rate.sleep()

# --------------------------------------------------
if __name__ == "__main__":
    try:
        UnitreeArmHW().run()
    except rospy.ROSInterruptException:
        pass
