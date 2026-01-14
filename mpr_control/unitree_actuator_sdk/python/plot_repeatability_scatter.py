#!/usr/bin/env python
# ---------------------------------------------------------
# CRITICAL FIX: Set backend to TkAgg to prevent Segfaults
# ---------------------------------------------------------
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    print("Warning: TkAgg not found. Switching to 'Agg'.")
    matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R
import sys
import glob
import os

# ---------------------------------------------------------
# Robot Configuration
# ---------------------------------------------------------
LINK_CONFIGS = [
    {'pos': [0, 0, 0.2487],        'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
    {'pos': [0.0218, 0, 0.059],    'euler': [0, 90, 180], 'axis': [0, 0, 1]}, 
    {'pos': [0.299774, 0, -0.0218],'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
    {'pos': [0.02, 0, 0],          'euler': [0, 90, 0],   'axis': [0, 0, 1]}, 
    {'pos': [0, 0, 0.315],         'euler': [0, -90, 0],  'axis': [0, 0, 1]}, 
    {'pos': [0.042824, 0, 0],      'euler': [0, 90, 0],   'axis': [0, 0, 1]}  
]

# ---------------------------------------------------------
# Kinematics & Math Helpers
# ---------------------------------------------------------
def get_transform_matrix(pos, euler, axis, q_val):
    T_static = np.eye(4)
    T_static[:3, 3] = pos
    T_static[:3, :3] = R.from_euler('xyz', euler, degrees=True).as_matrix()
    
    r_vec = np.array(axis) * q_val
    r_joint = R.from_rotvec(r_vec).as_matrix()
    T_joint = np.eye(4)
    T_joint[:3, :3] = r_joint
    
    return T_static @ T_joint

def compute_fk(joint_angles):
    T_curr = np.eye(4)
    for i, cfg in enumerate(LINK_CONFIGS):
        q = joint_angles[i]
        T_curr = T_curr @ get_transform_matrix(cfg['pos'], cfg['euler'], cfg['axis'], q)
    return T_curr[:3, 3]

def get_2d_covariance_ellipse(points_2d, n_std=3.0):
    """
    Returns (center, width, height, angle_degrees) for a 2D confidence ellipse.
    """
    if len(points_2d) < 3:
        return None
        
    mean = np.mean(points_2d, axis=0)
    cov = np.cov(points_2d, rowvar=False)
    
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(np.maximum(lambda_, 0))
    
    # Ellipse params
    width = lambda_[0] * n_std * 2
    height = lambda_[1] * n_std * 2
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    return (mean, width, height, angle)

def plot_3d_covariance(ax, points, color, n_std=3.0):
    if len(points) < 4: return
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = n_std * np.sqrt(np.maximum(eigvals, 0))
    
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 20)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones_like(u), np.cos(v))
    
    sphere = np.stack([x_unit.flatten(), y_unit.flatten(), z_unit.flatten()])
    final = (eigvecs @ (np.diag(radii) @ sphere)).T + mean
    
    X = final[:, 0].reshape(x_unit.shape)
    Y = final[:, 1].reshape(y_unit.shape)
    Z = final[:, 2].reshape(z_unit.shape)
    ax.plot_wireframe(X, Y, Z, color=color, alpha=0.15, rstride=2, cstride=2)

# ---------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------
def main():
    # 1. Load Data
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        files = glob.glob("repeatability_test_*.csv")
        if not files:
            print("No CSV files found.")
            sys.exit(1)
        filename = max(files, key=os.path.getctime)
    
    print(f"Loading: {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Extract Final Positions
    # Filter out "ReturnHome" (the parking move), keep everything else
    df_filtered = df[~df['segment'].str.contains("ReturnHome")]
    
    # Get last row of every segment in every cycle
    steady_points = df_filtered.groupby(['cycle', 'segment']).tail(1).copy()
    
    # 3. Categorize/Rename Segments for Plotting
    # We want to merge "InitToStart", "CycleReset", "WrapToStart" -> "Start (WP0)"
    def map_label(seg_name):
        if any(x in seg_name for x in ['InitToStart', 'CycleReset', 'WrapToStart', 'HomeToStart']):
            return "Start (WP0)"
        if "Seg_" in seg_name and "_to_" in seg_name:
            # format Seg_1_to_2 -> Waypoint 2
            try:
                parts = seg_name.split('_')
                return f"WP {parts[-1]}"
            except:
                return seg_name
        if "ToPoint_" in seg_name:
             parts = seg_name.split('_')
             return f"WP {parts[-1]}"
        return seg_name

    steady_points['label'] = steady_points['segment'].apply(map_label)
    
    # 4. Compute Cartesian Positions
    ee_data = []
    for _, row in steady_points.iterrows():
        joints = [row[f'act_m{i}'] for i in range(6)]
        xyz = compute_fk(joints)
        ee_data.append({
            'label': row['label'],
            'x': xyz[0], 'y': xyz[1], 'z': xyz[2]
        })
    
    plot_df = pd.DataFrame(ee_data)
    unique_labels = sorted(plot_df['label'].unique())
    
    # 5. Plotting (4 Subplots)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Repeatability Analysis (3-Sigma): {filename}", fontsize=16)

    # Define Axes
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    print("\n--- Cluster Statistics (mm) ---")
    
    for idx, label in enumerate(unique_labels):
        cluster = plot_df[plot_df['label'] == label]
        pts = cluster[['x', 'y', 'z']].values
        
        # Stats
        mean = np.mean(pts, axis=0)
        dist_from_mean = np.linalg.norm(pts - mean, axis=1)
        max_err_mm = np.max(dist_from_mean) * 1000
        std_mm = np.std(pts, axis=0) * 1000
        
        print(f"[{label}] N={len(pts)} | Max Err: {max_err_mm:.3f} | Std(xyz): {np.round(std_mm, 2)}")
        
        c = colors[idx]
        
        # --- 3D Plot ---
        ax_3d.scatter(pts[:,0], pts[:,1], pts[:,2], color=c, label=f"{label}", s=25)
        plot_3d_covariance(ax_3d, pts, c)
        
        # --- 2D Plots Helper ---
        def plot_2d_view(ax, dim1, dim2, idx1, idx2, xlabel, ylabel):
            pts_2d = pts[:, [idx1, idx2]]
            ax.scatter(pts_2d[:,0], pts_2d[:,1], color=c, s=15, alpha=0.6)
            
            # Covariance Ellipse
            ell_params = get_2d_covariance_ellipse(pts_2d)
            if ell_params:
                mean_2d, w, h, ang = ell_params
                ell = Ellipse(xy=mean_2d, width=w, height=h, angle=ang, 
                              edgecolor=c, fc='None', lw=1.5, ls='--')
                ax.add_patch(ell)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=':', alpha=0.6)

        # XY View (Top)
        plot_2d_view(ax_xy, 'x', 'y', 0, 1, 'X (m)', 'Y (m)')
        ax_xy.set_title("XY Plane (Top View)")
        
        # XZ View (Side)
        plot_2d_view(ax_xz, 'x', 'z', 0, 2, 'X (m)', 'Z (m)')
        ax_xz.set_title("XZ Plane (Side View)")
        
        # YZ View (Front)
        plot_2d_view(ax_yz, 'y', 'z', 1, 2, 'Y (m)', 'Z (m)')
        ax_yz.set_title("YZ Plane (Front View)")

    # Final Styling
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title("3D Isometric View")
    ax_3d.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for suptitle
    
    plt.show()

if __name__ == "__main__":
    main()