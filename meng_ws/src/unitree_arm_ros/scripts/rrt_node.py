#!/usr/bin/env python3
import os
import sys
import time
import argparse
import csv
import numpy as np
import rospy
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
TOPIC_SUB = "/joint_states_calibrated"
TOPIC_PUB = "/joint_commands_calibrated"

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
NUM_JOINTS = 6

RATE_HZ = 100            # Control loop frequency
TRANSITION_SPEED = 0.5   # rad/s (Speed for Home->Start and Start->Home)
PATH_PLAYBACK_SPEED = 1.0 # Multiplier for RRT path execution (1.0 = real time)

# ==================================================
# Data Loader
# ==================================================
def load_all_paths(dataset_root, max_episodes=None):
    """
    Scans the dataset folder and loads 'q1'...'q6' from path.csv files.
    Returns a list of numpy arrays (N x 6).
    """
    paths = []
    
    if not os.path.exists(dataset_root):
        rospy.logerr(f"Dataset root not found: {dataset_root}")
        return []

    # Sort folders to ensure deterministic order (episode_000, episode_001...)
    ep_dirs = sorted([d for d in os.listdir(dataset_root) if d.startswith("episode_")])

    if max_episodes:
        ep_dirs = ep_dirs[:max_episodes]

    rospy.loginfo(f"Found {len(ep_dirs)} episodes in {dataset_root}")

    for ep in ep_dirs:
        csv_path = os.path.join(dataset_root, ep, "path.csv")
        if not os.path.exists(csv_path):
            continue
            
        traj_points = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = np.array([
                    float(row["q1"]), float(row["q2"]), float(row["q3"]),
                    float(row["q4"]), float(row["q5"]), float(row["q6"])
                ])
                traj_points.append(q)
        
        if traj_points:
            paths.append(np.array(traj_points))
            
    return paths

# ==================================================
# Executor Node
# ==================================================
class RRTExecutorNode:
    def __init__(self, dataset_path, episodes):
        rospy.init_node("rrt_path_executor")

        # 1. Load Data
        rospy.loginfo(f"Loading paths from: {dataset_path}")
        self.paths = load_all_paths(dataset_path, episodes)
        if not self.paths:
            rospy.logerr("No paths loaded. Please check the path and try again.")
            sys.exit(1)

        # 2. Setup ROS
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        self.state_sub = rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)

        # State
        self.live_joints = None
        self.home_pose = None 
        self.state_received = False

        rospy.loginfo("Waiting for robot state...")
        while not self.state_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        # Save the starting position as "Home"
        self.home_pose = self.live_joints.copy()
        rospy.loginfo("Robot state received. Home pose recorded.")

    def state_cb(self, msg):
        """Parse current joint states"""
        q_temp = np.zeros(NUM_JOINTS)
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        
        valid = True
        for i, name in enumerate(JOINT_NAMES):
            if name in name_to_idx:
                q_temp[i] = msg.position[name_to_idx[name]]
            else:
                valid = False
                break
        
        if valid:
            self.live_joints = q_temp
            self.state_received = True

    def publish_cmd(self, q_target):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_target.tolist()
        self.cmd_pub.publish(msg)

    def move_segment_interpolated(self, target_q, speed_limit):
        """
        Moves from current live position to target_q using cubic interpolation.
        Used for the "Home -> Start" and "Start -> Home" transitions.
        """
        start_q = self.live_joints.copy()
        
        # Calculate duration based on max joint displacement
        max_diff = np.max(np.abs(target_q - start_q))
        duration = max_diff / speed_limit
        duration = max(duration, 2.0) # Minimum 2 seconds for safety on large moves

        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(RATE_HZ)

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            t = (now - start_time) / duration
            
            if t >= 1.0:
                self.publish_cmd(target_q)
                break

            # Cubic easing (smooth start/stop)
            # s(t) = 3t^2 - 2t^3
            smooth_t = 3*t**2 - 2*t**3
            
            cmd_q = start_q + (target_q - start_q) * smooth_t
            self.publish_cmd(cmd_q)
            rate.sleep()

    def execute_rrt_path(self, path_array):
        """
        Streams the dense RRT path points.
        """
        rate = rospy.Rate(RATE_HZ * PATH_PLAYBACK_SPEED)
        
        for q in path_array:
            if rospy.is_shutdown(): break
            self.publish_cmd(q)
            rate.sleep()

    def run(self):
        rospy.sleep(1.0) # Safety wait
        
        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if rospy.is_shutdown(): break
            
            rospy.loginfo(f"--- Executing Path {i+1}/{total} ---")
            
            path_start_q = path[0]

            # 1. Interpolate: Current -> Path Start
            rospy.loginfo("Moving to Path Start...")
            self.move_segment_interpolated(path_start_q, TRANSITION_SPEED)
            rospy.sleep(0.5)

            # 2. Execute: Path Start -> Path End (Forward)
            rospy.loginfo("Executing RRT Path (Forward)...")
            self.execute_rrt_path(path)
            rospy.sleep(0.5)

            # 3. Execute: Path End -> Path Start (Reverse/Backtrack)
            rospy.loginfo("Backtracking RRT Path (Reverse)...")
            self.execute_rrt_path(path[::-1]) # Reverse numpy array
            rospy.sleep(0.5)

        # 4. Final Return: Current -> Home
        rospy.loginfo("All paths done. Returning Home...")
        self.move_segment_interpolated(self.home_pose, TRANSITION_SPEED)
        
        rospy.loginfo("Mission Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # UPDATED DEFAULT PATH: Points to the folder in 'classical_planner' relative to your project root
    # Adjust this string if your user name or folder structure differs slightly
    default_dataset = os.path.expanduser("~/Desktop/MEng_project/classical_planner/random_rrt_dataset3")
    
    parser.add_argument("--path", type=str, default=default_dataset, help="Path to RRT dataset")
    parser.add_argument("--episodes", type=int, default=None, help="Limit number of episodes to run")
    args = parser.parse_args()

    try:
        node = RRTExecutorNode(args.path, args.episodes)
        node.run()
    except rospy.ROSInterruptException:
        pass