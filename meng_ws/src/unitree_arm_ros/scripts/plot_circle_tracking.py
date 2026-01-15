#!/usr/bin/env python3
import matplotlib
# Force backend to avoid ROS/Conda conflict
try:
    matplotlib.use('TkAgg') 
except:
    matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# Config
# ==========================================
CSV_FILE = "circle_path_tracking_error.csv"
SAVE_IMG = "circle_tracking_aligned_left_legend.png"

# Side-by-side layout (approx 7.5 inches wide)
FIG_SIZE_INCHES = (7.5, 3.5) 

def set_axes_equal(ax):
    """Make 3D axes of equal aspect ratio."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_FILE}.")
        return

    # Filter for CIRCLE phase
    df_circle = df[df['phase'] == 'CIRCLE'].copy()
    if df_circle.empty:
        print("Warning: No 'CIRCLE' phase found. Plotting all data.")
        df_plot = df
    else:
        df_plot = df_circle
        # Zero the time
        df_plot['timestamp'] = df_plot['timestamp'] - df_plot['timestamp'].iloc[0]

    # Arrays
    t = df_plot['timestamp'].values
    des = df_plot[['ee_x_des', 'ee_y_des', 'ee_z_des']].values
    act = df_plot[['ee_x_act', 'ee_y_act', 'ee_z_act']].values
    
    # --- ALIGNMENT STEP ---
    # Shift desired trajectory so its start point matches the actual start point
    start_offset = act[0] - des[0]
    des_aligned = des + start_offset
    print(f"Aligning Reference Trajectory by offset: {np.round(start_offset, 4)} m")

    # --- ERROR CALCULATION (Aligned) ---
    error_vec = act - des_aligned
    error_norm = np.linalg.norm(error_vec, axis=1)
    rmse = np.sqrt(np.mean(error_norm**2))
    
    min_err = np.min(error_norm)
    max_err = np.max(error_norm)
    
    # Calculate Split Time (Assuming 2 Loops, split is at 50% duration)
    total_duration = t[-1]
    split_time = total_duration / 2.0

    print(f"RMSE (Aligned): {rmse:.4f} m | Max Err: {max_err:.4f} m")

    # 2. Setup Figure (Side-by-Side)
    fig = plt.figure(figsize=FIG_SIZE_INCHES, dpi=200)

    # ==========================================
    # LEFT PANEL: 3D Trajectory
    # ==========================================
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Custom Colormap: Green -> Red
    colors = ["#00cc00", "#ff0000"]
    cmap = LinearSegmentedColormap.from_list("GreenRed", colors)
    
    # Create Segments
    points = act.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(vmin=min_err, vmax=max_err)
    
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=1.5)
    lc.set_array(error_norm)
    ax3d.add_collection(lc)
    
    # Plot ALIGNED Reference Line
    ax3d.plot(des_aligned[:,0], des_aligned[:,1], des_aligned[:,2], 
              color='black', linewidth=0.8, linestyle='--', label='Ref (Aligned)', alpha=0.6)

    # Markers
    ax3d.scatter(act[0,0], act[0,1], act[0,2], color='blue', s=20, label='Start')
    
    # Formatting
    ax3d.set_title(f"3D Path", fontsize=10, fontweight='bold')
    ax3d.tick_params(labelsize=6)
    ax3d.set_xlabel('X [m]', fontsize=7)
    ax3d.set_ylabel('Y [m]', fontsize=7)
    ax3d.set_zlabel('Z [m]', fontsize=7)
    
    # Equal Aspect Ratio
    ax3d.set_xlim(act[:,0].min(), act[:,0].max())
    ax3d.set_ylim(act[:,1].min(), act[:,1].max())
    ax3d.set_zlim(act[:,2].min(), act[:,2].max())
    set_axes_equal(ax3d)

    # Colorbar on the LEFT
    cbar = fig.colorbar(lc, ax=ax3d, location='left', fraction=0.03, pad=0.15)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Error [m]', fontsize=7)

    # ==========================================
    # RIGHT PANEL: Time Series (Horizontal)
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Plot line
    ax2.plot(t, error_norm, color='black', linewidth=1.0)
    ax2.fill_between(t, error_norm, color='gray', alpha=0.1)
    
    # --- INDICATOR FOR 2nd LOOP ---
    ax2.axvline(x=split_time, color='blue', linestyle='--', linewidth=1.0, alpha=0.8)
    # Text near bottom (10% height)
    ax2.text(split_time + 0.5, max_err * 0.1, "Start Loop 2 \u2192", color='blue', fontsize=8, verticalalignment='center')

    # Formatting
    ax2.set_title("Tracking Error", fontsize=10, fontweight='bold')
    ax2.set_ylabel('Error ||x - x*|| [m]', fontsize=8)
    ax2.set_xlabel('Time [s]', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Stats Box
    stats_text = f'RMSE: {rmse:.4f} m\nMax: {max_err:.4f} m'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(SAVE_IMG, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {SAVE_IMG}")
    
    if matplotlib.get_backend() != 'Agg':
        plt.show()

if __name__ == "__main__":
    main()