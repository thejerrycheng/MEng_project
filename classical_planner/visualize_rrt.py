import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import time
import argparse

# ============================================================
# Configuration (Match these to your ROS Node)
# ============================================================
TRANSITION_SPEED = 0.5   # rad/s (Speed for Home->Start jumps)
PLAYBACK_SPEED   = 1.0   # 1.0 = Real time
CONTROL_HZ       = 60    # Simulation update rate
DT               = 1.0 / CONTROL_HZ

# Path to the dataset and XML
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
# XML must match the one used for generation
XML_PATH = os.path.join(PROJECT_ROOT, "mujoco_sim", "assets", "scene2.xml")
DEFAULT_DATASET = os.path.join(CURR_DIR, "random_rrt_dataset4")

# ============================================================
# Helper: Data Loader
# ============================================================
def load_all_paths(dataset_root, max_episodes=None):
    paths = []
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Dataset not found: {dataset_root}")
        return []

    ep_dirs = sorted([d for d in os.listdir(dataset_root) if d.startswith("episode_")])
    if max_episodes: ep_dirs = ep_dirs[:max_episodes]

    print(f"[INFO] Loading {len(ep_dirs)} episodes from {dataset_root}...")

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

# ============================================================
# Helper: Obstacle Sync
# ============================================================
def load_and_sync_obstacles(model, dataset_root, episode_idx=0):
    """
    Reads obstacles.csv from the first episode (since they are fixed) 
    and updates the MuJoCo model to match.
    """
    ep_dirs = sorted([d for d in os.listdir(dataset_root) if d.startswith("episode_")])
    if not ep_dirs: return

    obs_path = os.path.join(dataset_root, ep_dirs[episode_idx], "obstacles.csv")
    if not os.path.exists(obs_path): return

    with open(obs_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                pos = [float(row["x"]), float(row["y"]), float(row["z"])]
                # Simple yaw to quat
                yaw = float(row["yaw"])
                quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(quat, np.array([0,0,1]), yaw)
                
                model.geom_pos[gid] = pos
                model.geom_quat[gid] = quat

# ============================================================
# Virtual Executor
# ============================================================
class VirtualExecutor:
    def __init__(self, model, data, viewer):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.home_q = data.qpos[:6].copy() # Record initial spawn as "Home"

    def update_viz(self):
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
        time.sleep(DT)

    def move_segment_interpolated(self, target_q):
        """
        Cubic interpolation between current position and target.
        Simulates the safe transition logic of the ROS node.
        """
        start_q = self.data.qpos[:6].copy()
        
        # Calculate duration
        max_diff = np.max(np.abs(target_q - start_q))
        duration = max_diff / TRANSITION_SPEED
        duration = max(duration, 1.0) # Min 1 sec duration

        start_time = time.time()
        while True:
            now = time.time()
            t = (now - start_time) / duration
            
            if t >= 1.0:
                self.data.qpos[:6] = target_q
                self.update_viz()
                break

            # Cubic easing: s(t) = 3t^2 - 2t^3
            smooth_t = 3*t**2 - 2*t**3
            cmd_q = start_q + (target_q - start_q) * smooth_t
            
            self.data.qpos[:6] = cmd_q
            self.update_viz()

    def execute_rrt_path(self, path_array):
        """
        Plays the dense path array directly.
        """
        # Adjust playback speed logic if needed, but simple iteration works for dense paths
        for q in path_array:
            self.data.qpos[:6] = q
            self.update_viz()

    def run_sequence(self, paths):
        # 0. Start at Home
        print("[Status] Starting at HOME position.")
        self.data.qpos[:6] = self.home_q
        self.update_viz()
        time.sleep(1.0)

        for i, path in enumerate(paths):
            print(f"--- Running Path {i+1}/{len(paths)} ---")
            
            path_start = path[0]
            
            # 1. Home -> Start (Interpolated)
            print("   -> Moving to Path Start (Safe Transition)...")
            self.move_segment_interpolated(path_start)
            time.sleep(0.2)

            # 2. Start -> End (Forward Path)
            print("   -> Executing RRT Path (Forward)...")
            self.execute_rrt_path(path)
            time.sleep(0.2)

            # 3. End -> Start (Reverse Path)
            print("   -> Backtracking RRT Path (Reverse)...")
            self.execute_rrt_path(path[::-1])
            time.sleep(0.2)
        
        # 4. Return Home
        print("[Status] All paths done. Returning HOME...")
        self.move_segment_interpolated(self.home_q)
        print("[Status] Complete.")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_DATASET, help="Path to dataset")
    parser.add_argument("--episodes", type=int, default=None, help="Max episodes to run")
    args = parser.parse_args()

    # 1. Load Data
    paths = load_all_paths(args.path, args.episodes)
    if not paths: return

    # 2. Load MuJoCo
    if not os.path.exists(XML_PATH):
        print(f"XML not found: {XML_PATH}"); return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 3. Sync Obstacles (Visuals only)
    load_and_sync_obstacles(model, args.path)

    # 4. Launch Viewer & Run
    with mujoco.viewer.launch_passive(model, data) as viewer:
        executor = VirtualExecutor(model, data, viewer)
        
        # Initial Camera Setup (Optional)
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat = [0.3, 0, 0.3]
        
        executor.run_sequence(paths)

if __name__ == "__main__":
    main()