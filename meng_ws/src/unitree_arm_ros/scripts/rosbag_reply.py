#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import rospy
import rosbag
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
# Topics
TOPIC_SUB = "/joint_states_calibrated"   # Live robot state
TOPIC_PUB = "/joint_commands_calibrated" # Command topic

# Robot Constraints for Homing (Approach)
HOMING_VELOCITY = 0.2  # rad/s (speed to move to start of bag)
RATE_HZ = 200          # Control loop rate
DT = 1.0 / RATE_HZ

JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6

# ==================================================
# Helpers
# ==================================================
def read_bag_trajectory(bag_path):
    """
    Reads joint states from the bag file and returns a list of 
    (relative_time, joint_positions) tuples.
    """
    rospy.loginfo(f"Loading bag: {bag_path}")
    trajectory = []
    start_time = None

    try:
        bag = rosbag.Bag(bag_path)
        # We look for the same topic we usually listen to, or the recorded version
        # Adjust topic name if your bag saved it differently (e.g. /joint_states)
        topic_to_read = TOPIC_SUB 
        
        for topic, msg, t in bag.read_messages(topics=[topic_to_read]):
            # Order the joints correctly
            q_pos = np.zeros(NUM_JOINTS)
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            
            valid_msg = True
            for i, name in enumerate(JOINT_NAMES):
                if name in name_to_idx:
                    q_pos[i] = msg.position[name_to_idx[name]]
                else:
                    valid_msg = False
                    break
            
            if not valid_msg: 
                continue

            if start_time is None:
                start_time = t.to_sec()

            rel_time = t.to_sec() - start_time
            trajectory.append((rel_time, q_pos))

        bag.close()
    except Exception as e:
        rospy.logerr(f"Failed to read bag: {e}")
        sys.exit(1)

    if not trajectory:
        rospy.logerr("No valid joint states found in bag!")
        sys.exit(1)

    rospy.loginfo(f"Loaded {len(trajectory)} points. Duration: {trajectory[-1][0]:.2f}s")
    return trajectory

# ==================================================
# Main Node
# ==================================================
class TeachRepeatNode:
    def __init__(self, bag_path):
        rospy.init_node("rosbag_replay_node")

        # 1. Load Trajectory from Bag
        self.recorded_traj = read_bag_trajectory(bag_path)

        # 2. Setup ROS
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        self.state_sub = rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)

        # State
        self.live_joints = None
        self.state_received = False 
        
        rospy.loginfo("Waiting for live /joint_states_calibrated to init homing...")

    def state_cb(self, msg):
        """Update live robot state constantly"""
        q_temp = np.zeros(NUM_JOINTS, dtype=float)
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        
        complete = True
        for i, name in enumerate(JOINT_NAMES):
            if name in name_to_idx:
                q_temp[i] = msg.position[name_to_idx[name]]
            else:
                complete = False
        
        if complete:
            self.live_joints = q_temp
            self.state_received = True

    def publish_cmd(self, q_target):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_target.tolist()
        msg.velocity = [] # Optional: Calculate vel if needed
        msg.effort = []
        self.cmd_pub.publish(msg)

    def run_homing(self):
        """
        Smoothly interpolate from CURRENT LIVE position to RECORDING START position.
        """
        while not self.state_received and not rospy.is_shutdown():
            rospy.sleep(0.1)

        start_joints = self.live_joints.copy()
        target_joints = self.recorded_traj[0][1] # First frame of bag
        
        # Calculate max displacement to determine duration
        max_diff = np.max(np.abs(start_joints - target_joints))
        duration = max_diff / HOMING_VELOCITY
        # Ensure at least 1 second for safety if diff is tiny
        duration = max(duration, 1.0) 

        rospy.loginfo(f"Homing to start position... (Duration: {duration:.2f}s)")
        
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(RATE_HZ)

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now - start_time
            t = min(elapsed / duration, 1.0)
            
            # Cubic interpolation (smooth start/stop)
            # s(t) = 3t^2 - 2t^3
            smooth_t = 3*t**2 - 2*t**3

            current_cmd = start_joints + (target_joints - start_joints) * smooth_t
            self.publish_cmd(current_cmd)

            if t >= 1.0:
                break
            rate.sleep()
            
        rospy.loginfo("Homing Complete. Starting Playback...")
        rospy.sleep(0.5) # Short pause before playback

    def run_playback(self):
        """
        Replay the recorded trajectory
        """
        rate = rospy.Rate(RATE_HZ)
        playback_start_time = rospy.Time.now().to_sec()
        
        # We iterate through the recorded points
        # To handle playback speed correctly, we look up the point 
        # corresponding to the current elapsed time.
        
        traj_idx = 0
        total_points = len(self.recorded_traj)
        
        while not rospy.is_shutdown() and traj_idx < total_points:
            now = rospy.Time.now().to_sec()
            time_since_start = now - playback_start_time
            
            # Advance index until we find the frame matching current time
            while (traj_idx < total_points - 1 and 
                   self.recorded_traj[traj_idx+1][0] < time_since_start):
                traj_idx += 1
            
            # Get current target
            target_q = self.recorded_traj[traj_idx][1]
            self.publish_cmd(target_q)
            
            # Progress bar
            if traj_idx % 20 == 0:
                progress = (traj_idx / total_points) * 100
                sys.stdout.write(f"\rPlayback: {progress:.1f}% | Time: {time_since_start:.2f}s")
                sys.stdout.flush()

            rate.sleep()
        
        print("\nPlayback Finished.")

    def run(self):
        try:
            self.run_homing()
            self.run_playback()
        except rospy.ROSInterruptException:
            pass

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="ROSbag replay Node")
    parser.add_argument("bag_path", type=str, help="Path to the .bag file")
    args = parser.parse_args()

    if not os.path.exists(args.bag_path):
        print(f"Error: File not found at {args.bag_path}")
        sys.exit(1)

    node = TeachRepeatNode(args.bag_path)
    node.run()