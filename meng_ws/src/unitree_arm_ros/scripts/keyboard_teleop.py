#!/usr/bin/env python3
import rospy
import math
from pynput import keyboard
from sensor_msgs.msg import JointState

JOINT_NAME = "joint_1"

POS_LIMIT = math.pi

# Duration range limits (rad)
DURATION_MIN = 0.5
DURATION_MAX = 3.0
DURATION_STEP = 0.2   # change per left/right key press

# Publish rate
PUBLISH_HZ = 50

class KeyboardTeleop:
    def __init__(self):
        rospy.init_node("keyboard_teleop")

        # Publisher: target command
        self.cmd_pub = rospy.Publisher("/motor_command", JointState, queue_size=1)

        # Subscriber: motor feedback
        rospy.Subscriber("/motor_state", JointState, self.state_cb, queue_size=1)

        # Internal state
        self.q_target = 0.0
        self.q_meas = None
        self.dq_meas = 0.0

        # Duration range (0 → 3 rad)
        self.duration_range = 0.0

        # Key hold states
        self.key_up_pressed = False
        self.key_down_pressed = False

        rospy.loginfo("===================================")
        rospy.loginfo(" Keyboard Teleop Running")
        rospy.loginfo(" Waiting for motor state...")
        rospy.loginfo("===================================")

        # Wait for motor feedback
        while not rospy.is_shutdown() and self.q_meas is None:
            rospy.sleep(0.05)

        # Initialize target to current motor position
        self.q_target = self.q_meas

        rospy.loginfo("Motor state received.")
        rospy.loginfo("Starting teleop at q = %.3f rad", self.q_target)
        rospy.loginfo("Controls:")
        rospy.loginfo(" ↑ / ↓ : Move motor")
        rospy.loginfo(" ← / → : Change duration range (0 → 3 rad)")
        rospy.loginfo(" ESC : Quit")
        rospy.loginfo("===================================")
        rospy.loginfo("Current Duration Range: %.2f rad", self.duration_range)

        # Keyboard listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.daemon = True
        listener.start()

        # Main loop
        self.loop()

    # --------------------------------------------------
    def state_cb(self, msg):
        if JOINT_NAME in msg.name:
            idx = msg.name.index(JOINT_NAME)
            if len(msg.position) > idx:
                self.q_meas = msg.position[idx]
            if len(msg.velocity) > idx:
                self.dq_meas = msg.velocity[idx]

    # --------------------------------------------------
    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.key_up_pressed = True

            elif key == keyboard.Key.down:
                self.key_down_pressed = True

            elif key == keyboard.Key.left:
                self.duration_range = max(DURATION_MIN, self.duration_range - DURATION_STEP)
                rospy.loginfo("Duration Range → %.2f rad", self.duration_range)

            elif key == keyboard.Key.right:
                self.duration_range = min(DURATION_MAX, self.duration_range + DURATION_STEP)
                rospy.loginfo("Duration Range → %.2f rad", self.duration_range)

            elif key == keyboard.Key.esc:
                rospy.signal_shutdown("User Exit")

        except:
            pass

    # --------------------------------------------------
    def on_release(self, key):
        if key == keyboard.Key.up:
            self.key_up_pressed = False
        elif key == keyboard.Key.down:
            self.key_down_pressed = False

    # --------------------------------------------------
    def loop(self):
        rate = rospy.Rate(PUBLISH_HZ)
        msg = JointState()
        msg.name = [JOINT_NAME]

        while not rospy.is_shutdown():

            # Compute incremental step from duration range
            # Larger duration_range → faster motion
            step = self.duration_range / PUBLISH_HZ

            if self.key_up_pressed:
                self.q_target += step

            if self.key_down_pressed:
                self.q_target -= step

            # Clamp to joint limits
            self.q_target = max(-POS_LIMIT, min(POS_LIMIT, self.q_target))

            # Publish command
            msg.header.stamp = rospy.Time.now()
            msg.position = [self.q_target]
            self.cmd_pub.publish(msg)

            # Log occasionally (5 Hz) to avoid spam
            if int(rospy.Time.now().to_sec() * 5) % 5 == 0:
                rospy.loginfo(
                    "Duration: %.2f rad | Target: %.3f | Measured: %.3f | Vel: %.3f",
                    self.duration_range, self.q_target, self.q_meas, self.dq_meas
                )

            rate.sleep()

# --------------------------------------------------
if __name__ == "__main__":
    KeyboardTeleop()
