#!/usr/bin/env python
# ---------------------------------------------------------
# CRITICAL FIX: Set backend to TkAgg
# ---------------------------------------------------------
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    print("Warning: TkAgg not found. Switching to 'Agg'.")
    matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Kinematics & Math
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

def get_error_ellipse(points_2d, n_std=3.0):
    if len(points_2d) < 3: return None
    cov = np.cov(points_2d, rowvar=False)
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(np.maximum(lambda_, 0))
    width = lambda_[0] * n_std * 2
    height = lambda_[1] * n_std * 2
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    return ((0,0), width, height, angle)

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Plot Repeatability Metrics")
    parser.add_argument("--name", type=str, help="Path to the input CSV file", default=None)
    args = parser.parse_args()

    # Determine Filename
    if args.name:
        filename = args.name
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
    else:
        # Auto-find newest file if --name is not provided
        files = glob.glob("repeatability_test_*.csv")
        if not files:
            print("No CSV files found (and no --name provided).")
            sys.exit(1)
        filename = max(files, key=os.path.getctime)
    
    print(f"Loading: {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # 2. Filter Data (Exclude Cycle 1 & ReturnHome)
    if 'segment' not in df.columns or 'cycle' not in df.columns:
        print("Error: CSV missing required columns ('cycle', 'segment').")
        sys.exit(1)

    df_filtered = df[~df['segment'].str.contains("ReturnHome")]
    df_filtered = df_filtered[df_filtered['cycle'] > 1]
    
    if df_filtered.empty:
        print("Error: No data remaining. Ensure >1 cycle in data.")
        sys.exit(1)

    steady_points = df_filtered.groupby(['cycle', 'segment']).tail(1).copy()
    
    def map_label(seg_name):
        if any(x in seg_name for x in ['InitToStart', 'CycleReset', 'WrapToStart', 'HomeToStart']):
            return "Start"
        if "Seg_" in seg_name:
            try: return f"WP{seg_name.split('_')[-1]}"
            except: pass
        if "ToPoint_" in seg_name:
             try: return f"WP{seg_name.split('_')[-1]}"
             except: pass
        return seg_name

    steady_points['label'] = steady_points['segment'].apply(map_label)
    unique_labels = sorted(steady_points['label'].unique())

    # 3. Calculate Stats
    clusters = {}
    stats = {}
    all_max_devs = []

    print(f"\n--- Statistics (Cycles 2-10) ---")
    for label in unique_labels:
        subset = steady_points[steady_points['label'] == label]
        
        # Compute FK (XYZ)
        positions = []
        for _, row in subset.iterrows():
            q = [row[f'act_m{i}'] for i in range(6)]
            positions.append(compute_fk(q))
        positions = np.array(positions)
        
        # Mean Center & Deviations (mm)
        mean_pos = np.mean(positions, axis=0)
        devs_mm = (positions - mean_pos) * 1000
        clusters[label] = devs_mm

        # A. Mean Error (Euclidean distance from centroid)
        dists = np.linalg.norm(devs_mm, axis=1) # 3D distance
        mean_err = np.mean(dists)
        
        # B. Max Error (Max 3D deviation)
        max_err = np.max(dists)
        
        # C. Total Variance (Trace of Covariance Matrix in XYZ)
        if len(positions) > 1:
            cov_matrix = np.cov(devs_mm, rowvar=False)
            total_variance = np.trace(cov_matrix)
        else:
            total_variance = 0.0
        
        stats[label] = {
            'mean_err': mean_err,
            'max_err': max_err,
            'tot_var': total_variance
        }
        
        # For plot scaling (use max XY deviation)
        max_xy = np.max(np.linalg.norm(devs_mm[:, 0:2], axis=1))
        all_max_devs.append(max_xy)
        
        print(f"[{label}] MeanErr: {mean_err:.3f} | MaxErr: {max_err:.3f} | TotVar: {total_variance:.2f}")

    # Axis Limits
    if all_max_devs:
        global_limit = max(all_max_devs) * 1.35
        global_limit = max(global_limit, 0.2)
    else:
        global_limit = 1.0

    # 4. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.5), constrained_layout=True)
    flat_axes = axes.flatten()
    fig.suptitle('Repeatability Metrics (XY View - 10 Cycles)', fontsize=9)

    for i, ax in enumerate(flat_axes):
        if i < len(unique_labels):
            label = unique_labels[i]
            devs = clusters[label]
            st = stats[label]
            
            x_vals = devs[:, 0]
            y_vals = devs[:, 1]
            points_2d = np.column_stack((x_vals, y_vals))
            
            # Scatter
            ax.scatter(x_vals, y_vals, c='blue', s=8, alpha=0.6, edgecolors='none')
            ax.scatter([0], [0], c='red', marker='+', s=40, linewidth=1.0)
            
            # Ellipse (XY Plane)
            ell_params = get_error_ellipse(points_2d, n_std=3.0)
            if ell_params:
                center, w, h, ang = ell_params
                ell = Ellipse(xy=center, width=w, height=h, angle=ang, 
                              edgecolor='red', fc='None', lw=0.8, ls='--')
                ax.add_patch(ell)
            
            # --- INFO BOX ---
            text_str = (f"Mean: {st['mean_err']:.2f}mm\n"
                        f"Max:  {st['max_err']:.2f}mm\n"
                        f"Var:  {st['tot_var']:.2f}")
            
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=5,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.2))

            ax.set_title(label, fontsize=8, pad=2)
        else:
            ax.axis('off')

        if i < len(unique_labels):
            ax.set_xlim(-global_limit, global_limit)
            ax.set_ylim(-global_limit, global_limit)
            ax.set_aspect('equal', 'box')
            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            if i in [2, 3]: ax.set_xlabel('dX (mm)', fontsize=7)
            if i in [0, 2]: ax.set_ylabel('dY (mm)', fontsize=7)

    # Save
    base_name = os.path.basename(filename)
    output_filename = base_name.replace('.csv', '_xyz_stats.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n>> Plot saved to: {output_filename}")
    
    if matplotlib.get_backend() != 'Agg':
        plt.show()

if __name__ == "__main__":
    main()