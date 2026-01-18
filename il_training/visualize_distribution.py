#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
from tqdm import tqdm

# ==================================================
# Configuration
# ==================================================
# Update these if your paths change
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

NUM_JOINTS = 6

# ==================================================
# 1. Kinematics Helper (Direct Mapping)
# ==================================================
class MujocoDirectKinematics:
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")

        # Load MuJoCo Model (headless)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Find End Effector
        # We try to find 'ee_site' first (common convention), otherwise 'ee_mount'
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.is_site = True
        
        if self.ee_id == -1:
            # Fallback to body 'ee_mount'
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
            self.is_site = False
        
        if self.ee_id == -1:
             # Last resort: Last body in the chain
             self.ee_id = self.model.nbody - 1
             self.is_site = False

    def forward(self, joint_batch):
        """
        Args:
            joint_batch: (N, 6) numpy array of calibrated joint angles
        Returns:
            (N, 3) numpy array of EE XYZ positions
        """
        n_steps = len(joint_batch)
        traj_xyz = np.zeros((n_steps, 3))

        for i in range(n_steps):
            # DIRECT MAPPING:
            # The ROS node maps msg.position[i] -> qpos[i] directly.
            # We assume the CSV columns pos_0...pos_5 match this order.
            self.data.qpos[:NUM_JOINTS] = joint_batch[i]
            
            # Forward Kinematics
            mujoco.mj_kinematics(self.model, self.data)
            
            # Read EE position
            if self.is_site:
                traj_xyz[i] = self.data.site_xpos[self.ee_id]
            else:
                traj_xyz[i] = self.data.xpos[self.ee_id]
                
        return traj_xyz

# ==================================================
# 2. Trajectory Processor
# ==================================================
def process_all_trajectories(root_dir, xml_path):
    print(f"Initializing Kinematics from: {xml_path}")
    try:
        fk = MujocoDirectKinematics(xml_path)
    except Exception as e:
        print(f"Error initializing kinematics: {e}")
        return []

    all_trajectories = [] 
    
    print(f"Scanning {root_dir} for joint_states.csv...")
    csv_paths = []
    for root, dirs, files in os.walk(root_dir):
        if "joint_states.csv" in files:
            csv_paths.append(os.path.join(root, "joint_states.csv"))
            
    if not csv_paths:
        print("No CSV files found.")
        return []

    print(f"Found {len(csv_paths)} episodes. Computing Forward Kinematics...")

    for csv_file in tqdm(csv_paths):
        try:
            df = pd.read_csv(csv_file)
            
            # Extract joint columns (pos_0 to pos_5)
            # Logic: We assume the CSV saves them in the order: joint_1, joint_2... 
            # which usually corresponds to pos_0, pos_1...
            joint_cols = [c for c in df.columns if c.startswith("pos_")]
            
            if len(joint_cols) < 6:
                continue
            
            # Get joints (N, 6)
            # No offsets, no differential conversion (as per your ROS node logic)
            joints = df[joint_cols[:6]].to_numpy()
            
            # Compute FK
            traj_xyz = fk.forward(joints)
            all_trajectories.append(traj_xyz)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    return all_trajectories

# ==================================================
# 3. Visualization
# ==================================================
def plot_3d_trajectories(trajectories):
    if not trajectories:
        print("No valid trajectories to plot.")
        return

    print(f"Plotting {len(trajectories)} trajectories...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Concatenate for auto-scaling
    all_points = np.concatenate(trajectories, axis=0)
    
    for traj in trajectories:
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
        
        # Plot path
        ax.plot(xs, ys, zs, alpha=0.3, linewidth=1)
        # Start (Blue)
        ax.scatter(xs[0], ys[0], zs[0], c='blue', s=20, alpha=0.6)
        # End (Red)
        ax.scatter(xs[-1], ys[-1], zs[-1], c='red', marker='^', s=30, alpha=0.8)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Start'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', label='End'),
        Line2D([0], [0], color='gray', lw=1, label='Trajectory'),
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Robot Trajectories ({len(trajectories)} Episodes)\nDirect Mapping Mode')

    # Auto-scale axes
    center = all_points.mean(axis=0)
    radius = np.linalg.norm(all_points - center, axis=1).max()
    
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

    plt.show()

# ==================================================
# Main Execution
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize robot trajectories using Direct Mapping MuJoCo kinematics.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing episode subfolders.")
    parser.add_argument("--xml_path", type=str, default=XML_PATH, help="Path to MuJoCo XML")

    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        trajs = process_all_trajectories(args.data_dir, args.xml_path)
        plot_3d_trajectories(trajs)
    else:
        print(f"Error: Data directory '{args.data_dir}' does not exist.")