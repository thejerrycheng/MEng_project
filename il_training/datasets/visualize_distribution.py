import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
from tqdm import tqdm

# ==================================================
# Configuration
# ==================================================
DATA_DIR = "/media/jerry/SSD/final_data_no_obstacle"
SPLIT = "train"         
MAX_CLIPS = 5000       

# ==================================================
# 1. Kinematics Helper
# ==================================================
class IRISKinematics:
    def __init__(self):
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

    def forward_point(self, q_rad):
        """Computes singular EE position for one joint config"""
        T_accumulated = np.eye(4)
        for i in range(6): 
            T_link = self.get_local_transform(self.link_configs[i], q_rad[i])
            T_accumulated = T_accumulated @ T_link
        
        T_ee = self.get_local_transform(self.link_configs[6], 0)
        T_accumulated = T_accumulated @ T_ee
        return T_accumulated[:3, 3]

# ==================================================
# 2. Data Loading
# ==================================================
def load_dataset_stats(target_dir):
    clips = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    if not clips:
        return None

    random.shuffle(clips)
    if MAX_CLIPS and len(clips) > MAX_CLIPS:
        clips = clips[:MAX_CLIPS]

    print(f"Loading data from {len(clips)} clips...")

    all_joints = []
    all_deltas = []
    all_ee_positions = []
    
    fk = IRISKinematics()

    for clip_name in tqdm(clips):
        json_path = os.path.join(target_dir, clip_name, "robot", "data.json")
        if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Joints (History)
        joints = np.array(data['joint_seq'])
        all_joints.append(joints)
        
        # Deltas (Future Actions)
        deltas = np.array(data['fut_delta']) 
        all_deltas.append(deltas) 
        
        # EE Position
        current_q = joints[-1]
        ee_pos = fk.forward_point(current_q)
        all_ee_positions.append(ee_pos)

    return {
        "joints": np.vstack(all_joints),         # (Total_History_Steps, 6)
        "deltas": np.vstack(all_deltas),         # (Total_Future_Steps, 6)
        "ee_pos": np.array(all_ee_positions)
    }

# ==================================================
# 3. Plotting
# ==================================================
def main():
    target_dir = os.path.join(DATA_DIR, SPLIT)
    if not os.path.exists(target_dir):
        print(f"Error: {target_dir} not found.")
        return

    stats = load_dataset_stats(target_dir)
    if not stats:
        print("No data loaded.")
        return

    # -----------------------------------------------------------
    # Plot 1: 3D Workspace Coverage (Equal Aspect Ratio)
    # -----------------------------------------------------------
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    pts = stats["ee_pos"]
    p1 = ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap='viridis', s=5, alpha=0.6)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Figure 1: Workspace (Equal Aspect Ratio)\nTrue Physical Proportions')
    fig1.colorbar(p1, label='Height (Z)')

    # Force Equal Aspect Ratio
    mid_x = np.mean(pts[:,0])
    mid_y = np.mean(pts[:,1])
    mid_z = np.mean(pts[:,2])
    max_range = np.array([
        pts[:,0].max() - pts[:,0].min(),
        pts[:,1].max() - pts[:,1].min(),
        pts[:,2].max() - pts[:,2].min()
    ]).max() / 2.0

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

    # -----------------------------------------------------------
    # Plot 2: 3D Workspace Coverage (Unequal/Auto Aspect Ratio)
    # -----------------------------------------------------------
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Just scatter, let matplotlib handle scale
    p2 = ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap='plasma', s=5, alpha=0.6)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Figure 2: Workspace (Auto/Unequal Aspect Ratio)\nStretched to fit data')
    fig2.colorbar(p2, label='Height (Z)')
    
    plt.show()

    # -----------------------------------------------------------
    # Plot 3: Joint Angle Histograms
    # -----------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
    fig3.suptitle('Figure 3: Joint Position Distributions (Radians)', fontsize=16)
    
    joint_data = stats["joints"] 
    joint_names = [f"Joint {i+1}" for i in range(6)]
    
    axes3 = axes3.flatten()
    for i in range(6):
        ax = axes3[i]
        ax.hist(joint_data[:, i], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(joint_names[i])
        ax.set_xlabel('Angle (rad)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Plot 4: Action Distribution Per Joint
    # -----------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
    fig4.suptitle('Figure 4: Action Velocity Distributions (Deltas Per Joint)', fontsize=16)
    
    delta_data = stats["deltas"] 
    
    axes4 = axes4.flatten()
    for i in range(6):
        ax = axes4[i]
        ax.hist(delta_data[:, i], bins=50, color='salmon', edgecolor='black', alpha=0.7)
        
        ax.set_title(f"Joint {i+1} Delta")
        ax.set_xlabel('Delta (rad)')
        ax.set_ylabel('Count (Log Scale)')
        ax.set_yscale('log') 
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()