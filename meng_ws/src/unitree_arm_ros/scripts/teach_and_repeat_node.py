#!/usr/bin/env python3
import rospy
import threading
import time
import csv
import math   # <-- FIX: use math for trig

from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "joint_1","joint_2","joint_3",
    "joint_4","joint_5","joint_6"
]

RECORD_HZ = 200
PLAYBACK_HZ = 200

class TeachRepeatNode:
    def __init__(self):
        rospy.init_node("teach_repeat_node")

        # Buffers
        self.current_q = [0.0]*6
        self.received_state = False

        # Demonstration storage
        self.demo = []
        self.recording = False

        # ROS I/O
        rospy.Subscriber("/joint_states", JointState, self.joint_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher("/arm/command", JointState, queue_size=1)

        rospy.loginfo("======================================")
        rospy.loginfo("  Teach & Repeat ROS Node Ready")
        rospy.loginfo("======================================")

    # -------------------------
    def joint_cb(self, msg):
        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        for i,n in enumerate(JOINT_NAMES):
            if n in name_to_idx:
                self.current_q[i] = msg.position[name_to_idx[n]]
        self.received_state = True

    # -------------------------
    def record_loop(self):
        rate = rospy.Rate(RECORD_HZ)
        start_t = time.time()

        while self.recording and not rospy.is_shutdown():
            if not self.received_state:
                rate.sleep()
                continue

            self.demo.append({
                "time": time.time() - start_t,
                "q": self.current_q.copy()
            })
            rate.sleep()

    # -------------------------
    def publish_command(self, q_cmd):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_cmd
        self.cmd_pub.publish(msg)

    # -------------------------
    def smooth_move_to(self, q_target, duration=3.0):
        rospy.loginfo("Moving smoothly to start pose...")

        q_start = self.current_q.copy()
        t0 = time.time()
        rate = rospy.Rate(PLAYBACK_HZ)

        while not rospy.is_shutdown():
            t = time.time() - t0
            if t > duration:
                break

            alpha = t / duration
            # FIXED LINE
            alpha_smooth = 0.5 * (1 - math.cos(math.pi * alpha))

            q_cmd = [
                q_start[i] + (q_target[i]-q_start[i]) * alpha_smooth
                for i in range(6)
            ]
            self.publish_command(q_cmd)
            rate.sleep()

        self.publish_command(q_target)
        rospy.loginfo("Reached start pose.")

    # -------------------------
    def playback(self):
        if not self.demo:
            rospy.logwarn("No demo to play.")
            return

        rospy.loginfo("Playing back demonstration...")

        t0 = self.demo[0]["time"]
        start_wall = time.time()

        for sample in self.demo:
            desired_t = sample["time"] - t0
            while (time.time() - start_wall) < desired_t and not rospy.is_shutdown():
                time.sleep(0.0005)
            self.publish_command(sample["q"])

        rospy.loginfo("Playback finished.")

    # -------------------------
    def save_csv(self, filename="demonstration_data.csv"):
        with open(filename,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + JOINT_NAMES)
            for s in self.demo:
                writer.writerow([s["time"]] + s["q"])
        rospy.loginfo(f"Saved: {filename}")

    # -------------------------
    def run(self):
        rospy.loginfo("Waiting for /joint_states...")
        while not self.received_state and not rospy.is_shutdown():
            time.sleep(0.1)

        input("Start demonstration? Press ENTER to begin...")

        rospy.loginfo("Recording. Move robot by hand. Press ENTER to stop.")
        self.recording = True
        th = threading.Thread(target=self.record_loop)
        th.start()
        input()
        self.recording = False
        th.join()

        rospy.loginfo("Recording finished. Samples: %d", len(self.demo))
        self.save_csv()

        if input("Repeat path? (y/n): ").lower() == "y":
            self.smooth_move_to(self.demo[0]["q"], duration=4.0)
            self.playback()

        rospy.loginfo("Holding final pose. Ctrl+C to exit.")
        rate = rospy.Rate(50)
        final = self.demo[-1]["q"]
        while not rospy.is_shutdown():
            self.publish_command(final)
            rate.sleep()

# -------------------------
if __name__ == "__main__":
    TeachRepeatNode().run()
