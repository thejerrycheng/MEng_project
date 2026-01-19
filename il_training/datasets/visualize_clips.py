import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco

# ==================================================
# Configuration (Hyperparameters)
# ==================================================
DATA_DIR = "/media/jerry/SSD/final_data_no_obstacle"
SPLIT = "train"        

# [Hyperparameter 1] How many distinct episodes to visualize?
NUM_EPISODES = 5

# [Hyperparameter 2] How many clips to pick from EACH episode?
# These will be selected evenly across the episode's timeline.
CLIPS_PER_EPISODE = 5

# [Hyperparameter 3] How many points to plot per segment (History/Future)?
# Reduce this to make the plot less crowded. Set None for all points.
POINTS_PER_SEGMENT = 10

# ==================================================
# 1. Kinematics Helper
# ==================================================
class IRISKinematics:
    def __init__(self):
        # Configuration extracted from your XML
        self.link_configs = [
            {'pos': [0, 0, 0.2487],         'euler': [0, 0, 0],      'axis': [0, 0, 1]},
            {'pos': [0.0218, 0, 0.059],     'euler': [0, 90, 180],   'axis': [0, 0, 1]},
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0],      'axis': [0, 0, 1]},
            {'pos': [0.02, 0, 0],           'euler': [0, 90, 0],     'axis': [0, 0, 1]},
            {'pos': [0, 0, 0.315],          'euler': [0, -90, 0],    'axis': [0, 0, 1]},
            {'pos': [0.042824, 0, 0],       'euler': [0, 90, 180],   'axis': [0, 0, 1]},
            {'pos': [0, 0, 0],              'euler': [0, 0, 0],      'axis': [0, 0, 0]} 
        ]

    def get_local_transform(self, cfg, q_rad):
        T_pos = np.eye(4)
        T_pos[:3, 3] = cfg['pos']

        R_fixed = np.eye(3)
        if any(cfg['euler']):
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            R_fixed = mat.reshape(3, 3)
        
        T_rot_fixed = np.eye(4)
        T_rot_fixed[:3, :3] = R_fixed

        T_joint = np.eye(4)
        if np.any(cfg['axis']):
            quat_j = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat_j, np.array(cfg['axis']), q_rad)
            mat_j = np.zeros(9)
            mujoco.mju_quat2Mat(mat_j, quat_j)
            R_joint = mat_j.reshape(3, 3)
            T_joint[:3, :3] = R_joint

        return T_pos @ T_rot_fixed @ T_joint

    def forward_traj(self, joint_seq):
        N = len(joint_seq)
        xyz_seq = np.zeros((N, 3))
        
        for t in range(N):
            q_rad = joint_seq[t] 
            T_accumulated = np.eye(4)
            for i in range(6): 
                T_link = self.get_local_transform(self.link_configs[i], q_rad[i])
                T_accumulated = T_accumulated @ T_link
            T_ee = self.get_local_transform(self.link_configs[6], 0)
            T_accumulated = T_accumulated @ T_ee
            xyz_seq[t] = T_accumulated[:3, 3]
        return xyz_seq

def subsample(points, num_target):
    if num_target is None or num_target >= len(points):
        return points
    indices = np.linspace(0, len(points) - 1, num=num_target, dtype=int)
    return points[indices]

# ==================================================
# 2. Data Organization Logic
# ==================================================
def group_clips_by_episode(target_dir):
    """
    Scans directory and groups clip folders by their Episode ID.
    Returns dict: { 'episode_name': [list of clip paths sorted] }
    """
    all_items = os.listdir(target_dir)
    episode_map = {}

    print(f"Indexing clips in {target_dir}...")
    for item in all_items:
        full_path = os.path.join(target_dir, item)
        if not os.path.isdir(full_path):
            continue
            
        # Parse Name: x_positive_..._episode_0008_clip_00012
        if "_clip_" in item:
            parts = item.split("_clip_")
            ep_name = parts[0] # Everything before _clip_
            
            if ep_name not in episode_map:
                episode_map[ep_name] = []
            episode_map[ep_name].append(item)

    # Sort clips within each episode to ensure timeline order
    for ep in episode_map:
        episode_map[ep].sort()
        
    return episode_map

def select_evenly_spaced_clips(clip_list, n_clips):
    """
    Selects n_clips from the list, evenly distributed.
    """
    total = len(clip_list)
    if total <= n_clips:
        return clip_list # Take all if not enough
    
    # Generate linear indices
    indices = np.linspace(0, total - 1, num=n_clips, dtype=int)
    return [clip_list[i] for i in indices]

# ==================================================
# 3. Main Script
# ==================================================
def main():
    target_dir = os.path.join(DATA_DIR, SPLIT)
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} does not exist.")
        return

    # 1. Group Data
    episode_map = group_clips_by_episode(target_dir)
    if not episode_map:
        print("No episodes found.")
        return
    
    available_episodes = list(episode_map.keys())
    print(f"Found {len(available_episodes)} unique episodes.")

    # 2. Select Episodes
    selected_ep_names = random.sample(available_episodes, min(NUM_EPISODES, len(available_episodes)))
    
    # 3. Collect Clips to Plot
    clips_to_process = []
    for ep_name in selected_ep_names:
        all_clips = episode_map[ep_name]
        # Pick evenly spaced clips
        chosen_clips = select_evenly_spaced_clips(all_clips, CLIPS_PER_EPISODE)
        
        print(f"Episode {ep_name}: Selected indices {[all_clips.index(c) for c in chosen_clips]}")
        
        for c in chosen_clips:
            clips_to_process.append(c)

    # 4. Visualization
    fk = IRISKinematics()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print(f"\nProcessing {len(clips_to_process)} total clips...")

    for clip_name in clips_to_process:
        json_path = os.path.join(target_dir, clip_name, "robot", "data.json")
        if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        joint_seq = np.array(data['joint_seq'])
        fut_delta = np.array(data['fut_delta'])
        current_q = joint_seq[-1] 
        future_q = current_q + fut_delta 

        # FK
        hist_xyz = fk.forward_traj(joint_seq)
        fut_xyz = fk.forward_traj(future_q)

        # Subsample
        hist_plot = subsample(hist_xyz, POINTS_PER_SEGMENT)
        fut_plot = subsample(fut_xyz, POINTS_PER_SEGMENT)

        # Plot
        ax.plot(hist_plot[:,0], hist_plot[:,1], hist_plot[:,2], 
                c='blue', alpha=0.5, linewidth=1, marker='.', markersize=2)
        ax.plot(fut_plot[:,0], fut_plot[:,1], fut_plot[:,2], 
                c='green', alpha=0.5, linewidth=1, marker='.', markersize=2)
        
        # Current State
        curr = hist_xyz[-1]
        ax.scatter(curr[0], curr[1], curr[2], c='red', s=20, edgecolors='k', zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Structured Visualization\n{len(selected_ep_names)} Episodes | {CLIPS_PER_EPISODE} Clips/Ep | {POINTS_PER_SEGMENT} Pts/Seg')
    ax.auto_scale_xyz([0, 0.5], [-0.5, 0.5], [0, 0.5]) 
    
    plt.show()

if __name__ == "__main__":
    main()