#!/usr/bin/env python3
import os
import sys
import time
import argparse
import csv
import numpy as np
import rospy
import mujoco
import mujoco.viewer
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
TOPIC_SUB = "/joint_states_calibrated"
TOPIC_PUB = "/joint_commands_calibrated"

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
NUM_JOINTS = 6

RATE_HZ = 200               # Execution Rate
DT = 1.0 / RATE_HZ

# Constraints
TRANSITION_SPEED = 0.3      # rad/s (Home <-> Start)
PATH_MAX_VEL     = 0.15      # rad/s (Path Execution)
LPF_ALPHA        = 0.2      # Smoothing Factor (Lower = Smoother)

# MuJoCo
USER_HOME = os.path.expanduser("~")
XML_PATH = os.path.join(USER_HOME, "Desktop/MEng_project/mujoco_sim/assets/scene2.xml")

# ==================================================
# Data Loader
# ==================================================
def load_all_paths(dataset_root, max_episodes=None):
    paths = []
    if not os.path.exists(dataset_root):
        rospy.logerr(f"Dataset root not found: {dataset_root}")
        return []

    ep_dirs = sorted([d for d in os.listdir(dataset_root) if d.startswith("episode_")])
    if max_episodes: ep_dirs = ep_dirs[:max_episodes]

    rospy.loginfo(f"Found {len(ep_dirs)} episodes.")

    for ep in ep_dirs:
        csv_path = os.path.join(dataset_root, ep, "path.csv")
        if not os.path.exists(csv_path): continue
        traj = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = np.array([float(row[f"q{i}"]) for i in range(1, 7)])
                traj.append(q)
        if traj: paths.append(np.array(traj))
    return paths

# ==================================================
# Trajectory Generator (The Math Engine)
# ==================================================
class TrajectoryGenerator:
    """
    Simulates the control loop offline to generate a dense list of commands.
    """
    def __init__(self, start_pose):
        self.cmd_state = start_pose.copy()      # Internal integrator state
        self.filtered_state = start_pose.copy() # LPF output state
        self.buffer = []                        # List of (q_command)

    def _step_toward(self, target_q, speed_limit):
        """
        Simulates one control tick: Velocity Clamp -> LPF -> Store
        Returns: distance_to_target
        """
        # 1. Calculate raw error vector
        error = target_q - self.cmd_state
        dist = np.linalg.norm(error)

        # 2. Velocity Clamp (Max step per tick)
        max_step = speed_limit * DT
        
        if dist > max_step:
            step_vec = (error / dist) * max_step
        else:
            step_vec = error # Reached target in this tick

        # 3. Integrate Command
        self.cmd_state += step_vec

        # 4. Apply Low-Pass Filter (Simulated)
        self.filtered_state = (LPF_ALPHA * self.cmd_state) + ((1.0 - LPF_ALPHA) * self.filtered_state)

        # 5. Store Result
        self.buffer.append(self.filtered_state.copy())
        
        return dist

    def move_to_pose(self, target_q, speed_limit):
        """Generates frames to move from current state to target_q"""
        while True:
            dist = self._step_toward(target_q, speed_limit)
            if dist < 0.001: break # Convergence tolerance

    def follow_path(self, path_points, speed_limit):
        """Generates frames to follow a list of waypoints (carrot following)"""
        for i, waypoint in enumerate(path_points):
            # Move towards waypoint until close enough to switch to next
            while True:
                dist = self._step_toward(waypoint, speed_limit)
                # Corner cutting tolerance: 0.05 rad (~2.8 deg)
                # If we are this close, start blending to the next point
                if dist < 0.05: break 
        
        # Ensure we fully converge on the very last point
        self.move_to_pose(path_points[-1], speed_limit)

# ==================================================
# ROS Node
# ==================================================
class RRTExecutorNode:
    def __init__(self, dataset_path, episodes):
        rospy.init_node("rrt_path_executor")
        
        # 1. Load Paths
        self.paths = load_all_paths(dataset_path, episodes)
        if not self.paths: sys.exit(1)

        # 2. Setup ROS
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1000) # Large queue for streaming
        self.state_sub = rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)

        self.home_pose = None
        self.state_received = False
        self.trajectory_buffer = [] # The pre-computed mission

        # 3. Wait for Robot
        rospy.loginfo("Waiting for robot state...")
        while not self.state_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        self.home_pose = np.array(self.live_joints_cache)
        rospy.loginfo("Robot connected. Calculating trajectory...")

        # 4. Pre-Compute EVERYTHING
        self.compute_full_mission()

    def state_cb(self, msg):
        q_temp = np.zeros(NUM_JOINTS)
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        valid = True
        for i, name in enumerate(JOINT_NAMES):
            if name in name_to_idx: q_temp[i] = msg.position[name_to_idx[name]]
            else: valid = False
        if valid:
            self.live_joints_cache = q_temp
            self.state_received = True

    def compute_full_mission(self):
        """Generates the entire sequence of motor commands offline."""
        gen = TrajectoryGenerator(self.home_pose)
        
        total = len(self.paths)
        print(f"Pre-computing {total} episodes...")
        t0 = time.time()

        for i, path in enumerate(self.paths):
            path_start = path[0]
            
            # 1. Home -> Start
            gen.move_to_pose(path_start, TRANSITION_SPEED)
            
            # 2. Forward Path
            gen.follow_path(path, PATH_MAX_VEL)
            
            # 3. Reverse Path
            gen.follow_path(path[::-1], PATH_MAX_VEL)

        # 4. Return Home
        gen.move_to_pose(self.home_pose, TRANSITION_SPEED)

        self.trajectory_buffer = gen.buffer
        duration = len(self.trajectory_buffer) * DT
        print(f"Computation Done ({time.time()-t0:.3f}s).")
        print(f"Total Frames: {len(self.trajectory_buffer)} | Est Duration: {duration:.1f}s")

    def preview_mujoco(self):
        if not os.path.exists(XML_PATH): return
        print("\n=== STARTING PREVIEW (Press ESC to close early) ===")
        
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat = [0.3, 0, 0.3]; viewer.cam.distance = 1.5
            
            # Replay buffer faster than real-time for preview
            preview_step = 2 # Skip frames for speed (2x speed)
            
            for q in self.trajectory_buffer[::preview_step]:
                if not viewer.is_running(): break
                data.qpos[:6] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.005) 
        print("=== PREVIEW COMPLETE ===\n")

    def execute_hardware(self):
        print("!"*60)
        print(" WARNING: EXECUTION ON REAL ROBOT")
        print(f" {len(self.trajectory_buffer)} frames ready.")
        if input("Type 'y' to START: ").lower() != 'y': return

        rospy.loginfo("Streaming commands...")
        rate = rospy.Rate(RATE_HZ)
        
        msg = JointState()
        msg.name = JOINT_NAMES

        for i, q_cmd in enumerate(self.trajectory_buffer):
            if rospy.is_shutdown(): break
            
            msg.header.stamp = rospy.Time.now()
            msg.position = q_cmd.tolist()
            self.cmd_pub.publish(msg)
            
            if i % 200 == 0: # Log every second
                progress = (i / len(self.trajectory_buffer)) * 100
                sys.stdout.write(f"\rProgress: {progress:.1f}%")
                sys.stdout.flush()
            
            rate.sleep()
        
        print("\nDone.")

    def run(self):
        self.preview_mujoco()
        self.execute_hardware()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_dataset = os.path.expanduser("~/Desktop/MEng_project/classical_planner/random_rrt_dataset4")
    parser.add_argument("--path", type=str, default=default_dataset, help="Dataset path")
    parser.add_argument("--episodes", type=int, default=None, help="Max episodes")
    args = parser.parse_args()

    try:
        node = RRTExecutorNode(args.path, args.episodes)
        node.run()
    except rospy.ROSInterruptException:
        pass