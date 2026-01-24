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

COMMAND_TIMEOUT = 0.25   # seconds: Time before entering "Hold" mode
PASSIVE_TIMEOUT = 5.0    # seconds: Time in "Hold" before going "Passive"

# Max joint velocity (rad/s)
MAX_VEL = [0.6,0.6,0.6,0.8,1.0,1.0]

# Target low-pass filter
TARGET_LPF_ALPHA = 0.08

# Joint impedance gains
IMPEDANCE = {
    0: {"kp": 8.0, "kd": 0.03},
    1: {"kp": 8.0, "kd": 0.15},
    2: {"kp": 8.0, "kd": 0.06},
    3: {"kp": 6.0, "kd": 0.05},
    4: {"kp": 6.0, "kd": 0.03},
    5: {"kp": 6.0, "kd": 0.03},
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
        
        # State tracking for logging
        self.current_mode = "PASSIVE" # PASSIVE, ACTIVE, HOLD

        # ROS I/O
        self.state_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        rospy.Subscriber("/joint_commands", JointState, self.command_cb, queue_size=1)

        rospy.loginfo("==============================================")
        rospy.loginfo(" Unitree Arm Hardware Node")
        rospy.loginfo(" Publishes : /joint_states")
        rospy.loginfo(" Subscribes: /joint_commands")
        rospy.loginfo(" Control Rate: %d Hz", CONTROL_RATE)
        rospy.loginfo(" Auto-Passive Delay: %.1f s", PASSIVE_TIMEOUT)
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
            
            # Determine time since last command
            if self.last_cmd_time is None:
                dt_last_cmd = 99999.0 # Infinite if never received
            else:
                dt_last_cmd = now - self.last_cmd_time

            # --------------------------------------------------
            # STATE MACHINE LOGIC
            # --------------------------------------------------
            
            # 1. ACTIVE: Receiving commands recently
            if dt_last_cmd < COMMAND_TIMEOUT:
                state = "ACTIVE"
                
                # First time entering active -> initialize
                if self.q_cmd is None:
                    self.q_des_filt = self.q[:]
                    self.q_cmd = self.q[:]
                    rospy.loginfo("State -> ACTIVE (Control Engaged)")

            # 2. HOLD: Lost command, but within 5s grace period
            elif dt_last_cmd < PASSIVE_TIMEOUT:
                state = "HOLD"
                
                # If we just transitioned into HOLD, log it
                if self.current_mode != "HOLD" and self.current_mode == "ACTIVE":
                    rospy.logwarn(f"Command timeout! Holding position for {PASSIVE_TIMEOUT}s...")

            # 3. PASSIVE: No command for > 5s
            else:
                state = "PASSIVE"
                
                # Reset command buffer so next engagement is smooth
                if self.current_mode != "PASSIVE":
                    rospy.logwarn("Timeout exceeded -> Switching to PASSIVE mode.")
                    self.q_cmd = None 
                    self.q_des_filt = None

            self.current_mode = state

            # --------------------------------------------------
            # Main SDK streaming
            # --------------------------------------------------
            for i, mid in enumerate(MOTOR_IDS):

                # CASE: PASSIVE
                if state == "PASSIVE":
                    q, dq, err = self.sdk_cycle(mid, passive=True)

                # CASE: ACTIVE or HOLD
                else:
                    # Only update trajectory if ACTIVE. 
                    # If HOLD, we skip this block, effectively freezing q_cmd at last value.
                    if state == "ACTIVE":
                        # Low-pass filter target
                        self.q_des_filt[i] = (1.0 - TARGET_LPF_ALPHA)*self.q_des_filt[i] + TARGET_LPF_ALPHA*self.q_des[i]

                        # Velocity-limited ramp
                        err_q = self.q_des_filt[i] - self.q_cmd[i]
                        max_step = MAX_VEL[i]*DT
                        step = max(min(err_q, max_step), -max_step)
                        self.q_cmd[i] += step

                    # Send command (In HOLD, q_cmd stays constant, keeping motors stiff)
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