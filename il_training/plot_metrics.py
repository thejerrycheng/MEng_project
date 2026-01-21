import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import mujoco

# --- STYLE SETTINGS ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# --------------------------
# 1. KINEMATICS (For Orientation Error)
# --------------------------
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

    def _get_transform(self, cfg, q):
        T_pos = np.eye(4); T_pos[:3, 3] = cfg['pos']
        quat = np.zeros(4); mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
        mat = np.zeros(9); mujoco.mju_quat2Mat(mat, quat)
        T_rot = np.eye(4); T_rot[:3, :3] = mat.reshape(3,3)
        T_joint = np.eye(4)
        if np.any(cfg['axis']):
            q_j = np.zeros(4); mujoco.mju_axisAngle2Quat(q_j, np.array(cfg['axis']), q)
            m_j = np.zeros(9); mujoco.mju_quat2Mat(m_j, q_j)
            T_joint[:3, :3] = m_j.reshape(3,3)
        return T_pos @ T_rot @ T_joint

    def forward_rot(self, q):
        """Returns 3x3 Rotation Matrix of End-Effector"""
        T = np.eye(4)
        for i in range(6): T = T @ self._get_transform(self.link_configs[i], q[i])
        T = T @ self._get_transform(self.link_configs[6], 0)
        return T[:3, :3]

# --------------------------
# 2. METRICS
# --------------------------
def calculate_metrics(df, name, fk_solver):
    time = df['time'].values
    xyz = df[['x', 'y', 'z']].values
    joints = df[['j0','j1','j2','j3','j4','j5']].values
    
    dt = np.diff(time)
    dt = np.where(dt == 0, 1e-4, dt) 

    # 1. Smoothness (Mean Squared Jerk)
    vel = np.diff(xyz, axis=0) / dt[:, None]
    acc = np.diff(vel, axis=0) / dt[:-1, None]
    jerk = np.diff(acc, axis=0) / dt[:-2, None]
    # Scaling factor 1e4 makes numbers readable (e.g., 5.2 instead of 0.00052)
    e_smooth = np.mean(np.sum(jerk**2, axis=1))

    # 2. Path Length
    path_len = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    
    # 3. Framing Accuracy (Orientation Stability/Error)
    total_rot_deg = 0.0
    prev_R = fk_solver.forward_rot(joints[0])
    
    for i in range(1, len(joints), 5): # Sample every 5th to reduce noise
        curr_R = fk_solver.forward_rot(joints[i])
        # Geodesic distance: tr(R_diff)
        R_diff = curr_R @ prev_R.T
        tr = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
        angle = np.degrees(np.arccos(tr))
        total_rot_deg += abs(angle)
        prev_R = curr_R

    # 4. Average Speed
    avg_speed = np.mean(np.linalg.norm(vel, axis=1))

    return {
        "Name": name,
        "Smoothness": e_smooth,
        "PathLength": path_len,
        "AvgSpeed": avg_speed,
        "TotalRot": total_rot_deg,
        "data": {"time": time, "xyz": xyz, "vel": np.linalg.norm(vel, axis=1), "vel_time": time[:-1]}
    }

# --------------------------
# 3. PLOTTING
# --------------------------
def plot_summary(metrics_list, save_path):
    # Create a figure with a GridSpec layout
    # Row 0: 3D Trajectory (Left) + Table (Right)
    # Row 1: Velocity Profile (Spans full width)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    # --- Plot A: 3D Trajectory ---
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    for m in metrics_list:
        xyz = m['data']['xyz']
        ax3d.plot(xyz[:,0], xyz[:,1], xyz[:,2], label=m['Name'], linewidth=2)
        ax3d.scatter(xyz[0,0], xyz[0,1], xyz[0,2], c='g', s=30) # Start
        ax3d.scatter(xyz[-1,0], xyz[-1,1], xyz[-1,2], c='r', marker='x', s=50) # End
        
    ax3d.set_title("Cartesian Trajectory Comparison")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend()

    # --- Plot B: Metrics Table ---
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis('off')
    
    # Prepare Table Data
    col_labels = ["Method", "Jerk (Smooth)\n(1e3)", "Path Len\n(m)", "Avg Speed\n(m/s)", "Ang. Travel\n(deg)"]
    cell_text = []
    
    for m in metrics_list:
        row = [
            m['Name'],
            f"{m['Smoothness']/1000:.2f}", # Scaled for readability
            f"{m['PathLength']:.3f}",
            f"{m['AvgSpeed']:.3f}",
            f"{m['TotalRot']:.1f}"
        ]
        cell_text.append(row)
        
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2) # Make rows taller
    ax_table.set_title("Quantitative Performance Metrics", pad=20)

    # --- Plot C: Velocity Profile ---
    ax_vel = fig.add_subplot(gs[1, :])
    
    for m in metrics_list:
        ax_vel.plot(m['data']['vel_time'], m['data']['vel'], label=m['Name'], linewidth=1.5, alpha=0.8)
        
    ax_vel.set_title("Velocity Profile (Smoothness Visualizer)")
    ax_vel.set_ylabel("Speed (m/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.legend()
    ax_vel.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved consolidated plot to: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="Path to CSV log files")
    parser.add_argument("--labels", nargs='+', help="Labels: 'Ours' 'RRT' 'Human'")
    parser.add_argument("--out", default="final_paper_plot.png")
    args = parser.parse_args()

    fk_solver = IRISKinematics()
    metrics = []

    for i, f in enumerate(args.files):
        if not os.path.exists(f): continue
        
        name = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(f)
        df = pd.read_csv(f)
        metrics.append(calculate_metrics(df, name, fk_solver))

    if metrics:
        plot_summary(metrics, args.out)
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()