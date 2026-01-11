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
PORT = "/dev/ttyUSB0"
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_ID   = 0
JOINT_NAME = "joint_1"

GEAR = queryGearRatio(MOTOR_TYPE)

# Motor-side PD gains
KP = 1.5
KD = 0.05

CONTROL_RATE = 200        # Hz
USB_GUARD_SLEEP = 0.0005  # seconds
COMMAND_TIMEOUT = 0.3    # seconds

# --------------------------------------------------
class SingleMotorHW:
    def __init__(self):
        rospy.init_node("single_motor_hw_node")

        self.serial = SerialPort(PORT)
        self.cmd  = MotorCmd()
        self.data = MotorData()

        # Joint-side measured state
        self.q_meas = 0.0
        self.dq_meas = 0.0

        # Desired joint target
        self.q_des = None
        self.last_cmd_time = None

        # ROS I/O
        self.state_pub = rospy.Publisher("/motor_state", JointState, queue_size=1)
        rospy.Subscriber("/motor_command", JointState, self.command_cb, queue_size=1)

        rospy.loginfo("===================================")
        rospy.loginfo(" Single Motor HW Node Running")
        rospy.loginfo(" Publishes : /motor_state")
        rospy.loginfo(" Subscribes: /motor_command")
        rospy.loginfo("===================================")

        # --------------------------------------------------
        # Mandatory SDK warm-up stream
        # --------------------------------------------------
        rospy.loginfo("Performing SDK warm-up stream...")
        t0 = time.time()
        while time.time() - t0 < 0.5:   # 0.5s passive frames
            self.sdk_cycle(passive=True)
            time.sleep(0.002)
        rospy.loginfo("Warm-up complete. Motor ready.")

    # --------------------------------------------------
    def command_cb(self, msg):
        if JOINT_NAME in msg.name:
            idx = msg.name.index(JOINT_NAME)
            if len(msg.position) > idx:
                self.q_des = msg.position[idx]
                self.last_cmd_time = rospy.Time.now().to_sec()

    # --------------------------------------------------
    def sdk_cycle(self, passive=False):
        """One correct Unitree SDK send/recv cycle"""

        self.data.motorType = MOTOR_TYPE
        self.cmd.motorType  = MOTOR_TYPE
        self.cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        self.cmd.id   = MOTOR_ID

        if passive or self.q_des is None:
            # Transparent mode (no torque)
            self.cmd.q   = self.data.q
            self.cmd.dq  = 0.0
            self.cmd.kp  = 0.0
            self.cmd.kd  = 0.0
        else:
            # Active position control
            self.cmd.q   = self.q_des * GEAR
            self.cmd.dq  = 0.0
            self.cmd.kp  = KP
            self.cmd.kd  = KD

        self.cmd.tau = 0.0

        # --- Send & Receive ---
        self.serial.sendRecv(self.cmd, self.data)

        # --- Convert motor-side → joint-side ---
        self.q_meas  = self.data.q / GEAR
        self.dq_meas = self.data.dq / GEAR

    # --------------------------------------------------
    def publish_state(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [JOINT_NAME]
        msg.position = [self.q_meas]
        msg.velocity = [self.dq_meas]
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

            # Maintain continuous SDK stream
            if active:
                self.sdk_cycle(passive=False)
            else:
                self.sdk_cycle(passive=True)

            # Error check
            if self.data.merror != 0:
                rospy.logerr("Motor error %d → Power-cycle required", self.data.merror)
                break

            self.publish_state()

            # USB guard
            time.sleep(USB_GUARD_SLEEP)
            rate.sleep()

        self.safe_exit()

    # --------------------------------------------------
    def safe_exit(self):
        rospy.loginfo("Stopping motor safely...")

        self.data.motorType = MOTOR_TYPE
        self.cmd.motorType  = MOTOR_TYPE
        self.cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        self.cmd.id   = MOTOR_ID

        self.cmd.q   = self.data.q
        self.cmd.dq  = 0.0
        self.cmd.kp  = 0.0
        self.cmd.kd  = 0.0
        self.cmd.tau = 0.0

        self.serial.sendRecv(self.cmd, self.data)
        rospy.loginfo("Exited cleanly.")

# --------------------------------------------------
if __name__ == "__main__":
    try:
        SingleMotorHW().run()
    except rospy.ROSInterruptException:
        pass
