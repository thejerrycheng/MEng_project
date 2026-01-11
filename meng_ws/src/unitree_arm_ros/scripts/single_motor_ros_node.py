#!/usr/bin/env python3
import sys
import time
import rospy
from sensor_msgs.msg import JointState

# --------------------------------------------------
# Unitree SDK import (same as your working setup)
# --------------------------------------------------
UNITREE_SDK_LIB = "/home/jerry/Desktop/MEng_project/mpr_control/unitree_actuator_sdk/lib"
if UNITREE_SDK_LIB not in sys.path:
    sys.path.append(UNITREE_SDK_LIB)

from unitree_actuator_sdk import *

# --------------------------------------------------
PORT = "/dev/ttyUSB0"
MOTOR_TYPE = MotorType.GO_M8010_6
MOTOR_ID   = 0
JOINT_NAME = "joint_1"

# Same parameters as your proven minimal script
KP = 0.0
KD = 0.05
DQ_TARGET = -1.28                      # motor-side rad/s
GEAR = queryGearRatio(MOTOR_TYPE)

# Stable communication rate
CONTROL_RATE = 200                     # Hz
DT = 1.0 / CONTROL_RATE
USB_GUARD_SLEEP = 0.0005               # 0.5 ms

# --------------------------------------------------
class SingleMotorNode:
    def __init__(self):
        rospy.init_node("single_motor_node")

        self.serial = SerialPort(PORT)
        self.cmd  = MotorCmd()
        self.data = MotorData()

        self.state_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)

        rospy.loginfo("===================================")
        rospy.loginfo(" Single Motor ROS Node (Stable SDK)")
        rospy.loginfo("===================================")
        rospy.loginfo("Running at %d Hz", CONTROL_RATE)

    # --------------------------------------------------
    def publish_state(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [JOINT_NAME]
        msg.position = [self.data.q]
        msg.velocity = [self.data.dq]
        self.state_pub.publish(msg)

    # --------------------------------------------------
    def run(self):
        rate = rospy.Rate(CONTROL_RATE)

        while not rospy.is_shutdown():

            # --- EXACT minimal working SDK sequence ---
            self.data.motorType = MOTOR_TYPE
            self.cmd.motorType  = MOTOR_TYPE
            self.cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
            self.cmd.id   = MOTOR_ID

            self.cmd.q    = 0.0
            self.cmd.dq   = DQ_TARGET * GEAR
            self.cmd.kp   = KP
            self.cmd.kd   = KD
            self.cmd.tau  = 0.0

            # --- Send / Receive ---
            self.serial.sendRecv(self.cmd, self.data)

            # --- Publish ---
            if self.data.merror != 0:
                rospy.logerr("Motor error: %d  → Power-cycle required", self.data.merror)
                break

            self.publish_state()

            # --- Guard time for USB stability ---
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

        self.cmd.q   = 0.0
        self.cmd.dq  = 0.0
        self.cmd.kp  = 0.0
        self.cmd.kd  = 0.0
        self.cmd.tau = 0.0

        self.serial.sendRecv(self.cmd, self.data)
        rospy.loginfo("Exited cleanly.")

# --------------------------------------------------
if __name__ == "__main__":
    try:
        SingleMotorNode().run()
    except rospy.ROSInterruptException:
        pass
