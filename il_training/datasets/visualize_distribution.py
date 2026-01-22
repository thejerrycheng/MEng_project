import os
import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
from tqdm import tqdm
import seaborn as sns

# ==================================================
# CONFIGURATION
# ==================================================
FIG_WIDTH = 3.5   # IEEE Single Column
FIG_HEIGHT = 2.0  # Reduced Height
DPI = 300

# Fonts: Helvetica/Arial Bold, Size 7
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "legend.fontsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "lines.linewidth": 0.8
})

# ==================================================
# 1. Kinematics Helper
# ==================================================
class IRISKinematics:
    def __init__(self):
        self.link_configs = [
            {'pos': [0, 0, 0.2487], 'euler': [0, 0, 0], 'axis': [0, 0, 1]},
            {'pos': [0.0218, 0, 0.059], 'euler': [0, 90, 180], 'axis': [0, 0, 1]},
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0], 'axis': [0, 0, 1]},
            {'pos': [0.02, 0, 0], 'euler': [0, 90, 0], 'axis': [0, 0, 1]},
            {'pos': [0, 0, 0.315], 'euler': [0, -90, 0], 'axis': [0, 0, 1]},
            {'pos': [0.042824, 0, 0], 'euler': [0, 90, 180], 'axis': [0, 0, 1]},
            {'pos': [0, 0, 0], 'euler': [0, 0, 0], 'axis': [0, 0, 0]} 
        ]

    def get_local_transform(self, cfg, q_rad):
        T_pos = np.eye(4); T_pos[:3, 3] = cfg['pos']
        R_fixed = np.eye(3)
        if any(cfg['euler']):
            quat = np.zeros(4); mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
            mat = np.zeros(9); mujoco.mju_quat2Mat(mat, quat)
            R_fixed = mat.reshape(3, 3)
        T_rot_fixed = np.eye(4); T_rot_fixed[:3, :3] = R_fixed
        T_joint = np.eye(4)
        if np.any(cfg['axis']):
            quat_j = np.zeros(4); mujoco.mju_axisAngle2Quat(quat_j, np.array(cfg['axis']), q_rad)
            mat_j = np.zeros(9); mujoco.mju_quat2Mat(mat_j, quat_j)
            R_joint = mat_j.reshape(3, 3)
            T_joint[:3, :3] = R_joint
        return T_pos @ T_rot_fixed @ T_joint

    def forward_point(self, q_rad):
        T = np.eye(4)
        for i in range(6): T = T @ self.get_local_transform(self.link_configs[i], q_rad[i])
        T = T @ self.get_local_transform(self.link_configs[6], 0)
        return T[:3, 3]

# ==================================================
# 2. Data Loading
# ==================================================
def load_data(target_dir, max_clips=5000):
    clips = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    random.shuffle(clips)
    if max_clips: clips = clips[:max_clips]
    print(f"Loading {len(clips)} clips...")
    fk = IRISKinematics()
    ee_pos, joints, deltas, labels = [], [], [], []

    for c in tqdm(clips):
        path = os.path.join(target_dir, c, "robot", "data.json")
        if not os.path.exists(path): continue
        is_obs = 1 if "obstacle" in c.lower() else 0
        with open(path) as f: d = json.load(f)
        j = np.array(d['joint_seq'])[::2] 
        act = np.array(d['fut_delta'])[::2]
        joints.append(j); deltas.append(act)
        ee_pos.append(fk.forward_point(j[-1]))
        labels.append(is_obs)

    return {"ee": np.array(ee_pos), "joints": np.vstack(joints),
            "deltas": np.vstack(deltas), "labels": np.array(labels)}

# ==================================================
# 3. Plotting
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_clips", type=int, default=3000)
    args = parser.parse_args()

    data = load_data(os.path.join(args.data_dir, args.split), args.max_clips)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # 1 Row, 3 Cols. Give Panel A more width (1.4x)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.1, 
                          left=0.05, right=0.99, top=0.92, bottom=0.15)

    # --- PANEL A: 3D Workspace ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    
    pts = data['ee']; lbl = data['labels']
    free = pts[lbl==0]; obs = pts[lbl==1]
    if len(free)>2000: free = free[np.random.choice(len(free), 2000, replace=False)]
    if len(obs)>2000: obs = obs[np.random.choice(len(obs), 2000, replace=False)]

    ax1.scatter(free[:,0], free[:,1], free[:,2], c='#1f77b4', s=0.5, alpha=0.1, label='Free')
    ax1.scatter(obs[:,0], obs[:,1], obs[:,2], c='#d62728', s=1, alpha=0.4, label='Obs')
    
    ax1.view_init(elev=30, azim=45)
    ax1.set_xlabel('X', labelpad=-12); ax1.set_ylabel('Y', labelpad=-12); ax1.set_zlabel('Z', labelpad=-12)
    ax1.set_title("(a) Workspace", pad=-5, loc='left')
    ax1.tick_params(axis='both', which='major', pad=-6, labelsize=5)
    ax1.dist = 7
    ax1.legend(loc='upper left', frameon=False, bbox_to_anchor=(-0.2, 1.05), handletextpad=0.1)

    # --- PANEL B: Joint Angles ---
    ax2 = fig.add_subplot(gs[0, 1])
    J = data['joints']
    j_long = []
    for i in range(6):
        vals = J[:, i]
        if len(vals)>5000: vals=np.random.choice(vals,5000)
        for v in vals: j_long.append({'rad': v, 'Joint': f"J{i+1}"})
    
    import pandas as pd
    sns.violinplot(data=pd.DataFrame(j_long), x='rad', y='Joint', ax=ax2, 
                   inner="quart", linewidth=0.5, color="#abacae", saturation=0.7)
    
    ax2.set_title("(b) Joint Dist.", loc='left', pad=3)
    ax2.set_xlabel("Rad", labelpad=2)
    ax2.set_ylabel("") # Save space
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # --- PANEL C: Action Velocity ---
    ax3 = fig.add_subplot(gs[0, 2])
    D = data['deltas']
    d_long = []
    for i in range(6):
        vals = D[:, i]
        if len(vals)>5000: vals=np.random.choice(vals,5000)
        for v in vals: d_long.append({'rad': v, 'Joint': f"J{i+1}"})
            
    sns.violinplot(data=pd.DataFrame(d_long), x='rad', y='Joint', ax=ax3, 
                   inner="quart", linewidth=0.5, color="#ff9f9b", saturation=0.7)
    
    ax3.set_title("(c) Action Vel.", loc='left', pad=3)
    ax3.set_xlabel("Rad/Step", labelpad=2)
    ax3.set_ylabel("")
    ax3.set_yticklabels([]) # Hide Y labels (shared with B)
    ax3.grid(axis='x', linestyle='--', alpha=0.3)

    out_path = "dataset_single_row_tight.pdf"
    plt.savefig(out_path, dpi=DPI)
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=DPI)
    print(f"Saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()