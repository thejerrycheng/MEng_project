import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco

# ==================================================
# Configuration
# ==================================================
# UPDATED PATH based on your ls output:
DATA_DIR = "/media/jerry/SSD/final_data_no_obstacle" 

SPLIT = "train"        # 'train', 'val', or 'test'
NUM_CLIPS = 50         # How many random clips to plot

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
        """
        Args: joint_seq (N, 6)
        Returns: xyz_seq (N, 3)
        """
        N = len(joint_seq)
        xyz_seq = np.zeros((N, 3))
        
        for t in range(N):
            q_rad = joint_seq[t] 
            
            T_accumulated = np.eye(4)
            for i in range(6): 
                T_link = self.get_local_transform(self.link_configs[i], q_rad[i])
                T_accumulated = T_accumulated @ T_link
            
            # EE Mount
            T_ee = self.get_local_transform(self.link_configs[6], 0)
            T_accumulated = T_accumulated @ T_ee
            
            xyz_seq[t] = T_accumulated[:3, 3]
            
        return xyz_seq

# ==================================================
# 2. Main Visualization Script
# ==================================================
def main():
    target_dir = os.path.join(DATA_DIR, SPLIT)
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} does not exist.")
        return

    # Get all clip folders
    clips = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    if not clips:
        print(f"No clips found in {target_dir}.")
        return

    # Select random sample
    selected_clips = random.sample(clips, min(NUM_CLIPS, len(clips)))
    print(f"Visualizing {len(selected_clips)} clips from {target_dir}...")

    fk = IRISKinematics()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for clip_name in selected_clips:
        json_path = os.path.join(target_dir, clip_name, "robot", "data.json")
        
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 1. Parse Data
        joint_seq = np.array(data['joint_seq'])
        fut_delta = np.array(data['fut_delta'])

        # 2. Reconstruct Absolute Future
        current_q = joint_seq[-1] 
        future_q = current_q + fut_delta 

        # 3. Compute FK
        hist_xyz = fk.forward_traj(joint_seq)
        fut_xyz = fk.forward_traj(future_q)

        # 4. Plot History (Blue)
        ax.plot(hist_xyz[:,0], hist_xyz[:,1], hist_xyz[:,2], 
                c='blue', alpha=0.5, linewidth=1)

        # 5. Plot Future (Green)
        ax.plot(fut_xyz[:,0], fut_xyz[:,1], fut_xyz[:,2], 
                c='green', alpha=0.5, linewidth=1)

        # 6. Plot Current State (Red Dot)
        curr = hist_xyz[-1]
        ax.scatter(curr[0], curr[1], curr[2], c='red', s=10)

    # Make plot pretty
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Clip Visualization ({len(selected_clips)} Samples)\nBlue=History, Green=Future, Red=Current')
    
    # Auto-scale
    ax.auto_scale_xyz([0, 0.5], [-0.5, 0.5], [0, 0.5]) 
    
    plt.show()

if __name__ == "__main__":
    main()