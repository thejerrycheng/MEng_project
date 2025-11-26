import json
import argparse
import matplotlib
matplotlib.use('TkAgg')  # or try 'Agg' for a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def dh_transform(theta, d, alpha, a):
    """Compute the transformation matrix using DH parameters."""
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics(joint_angles):
    """Compute the forward kinematics of the xArm Lite 6 given joint angles."""
    # DH Table (theta offset, d, alpha, a) for 6 joints
    dh_params = [
        [0,    243.3,   0,   0],
        [-90,   0,    -90,   0],
        [-90,   0,    180, 200],
        [0,   227.6,   90,  87],
        [0,     0,     90,   0],
        [0,    61.5,  -90,   0]
    ]
    
    if len(joint_angles) != len(dh_params):
        raise ValueError(f"Expected {len(dh_params)} joint angles, but got {len(joint_angles)}.")
    
    T = np.eye(4)
    positions = []
    orientations = []
    
    for i in range(len(dh_params)):
        theta_offset, d, alpha, a = dh_params[i]
        theta = joint_angles[i] + theta_offset
        T_next = dh_transform(theta, d, alpha, a)
        T = np.dot(T, T_next)
        positions.append(T[:3, 3])
        orientations.append(T[:3, :3])
    
    return np.array(positions), orientations

def load_joint_data(json_file):
    with open(json_file, 'r') as f:
        joint_data = json.load(f)
    
    timestamps = np.arange(len(joint_data))  # Assuming one timestamp per joint state
    joint_positions = []
    
    for state in joint_data:
        if "position" in state:
            joint_positions.append(state["position"])
    
    if not joint_positions:
        print("No joint position data found in JSON file.")
        return None, None
    
    joint_positions = np.array(joint_positions)
    print("Successfully finished reading the JSON file ...")
    print("The size of the joint_positions is:", joint_positions.shape)
    
    return timestamps, joint_positions

def plot_joint_trajectories(json_file):
    timestamps, joint_positions = load_joint_data(json_file)
    
    if timestamps is None or joint_positions is None:
        return
    
    print("Plotting the joint trajectories ...")
    for i in range(6):
        plt.plot(timestamps, joint_positions[:, i], label=f'Joint {i+1}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Joint Position (radians)')
    plt.title('Joint Trajectories Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def fit_line_least_squares(points):
    """
    Fit a straight line to the given 3D points using least squares.
    Returns the centroid and the unit direction vector.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, s, Vt = np.linalg.svd(centered)
    direction = Vt[0]
    direction /= np.linalg.norm(direction)
    return centroid, direction

def get_line_endpoints(points, centroid, direction):
    """
    Compute endpoints for the best-fit line by projecting the points onto the direction vector.
    """
    projections = np.dot(points - centroid, direction)
    t_min, t_max = projections.min(), projections.max()
    endpoint1 = centroid + t_min * direction
    endpoint2 = centroid + t_max * direction
    return endpoint1, endpoint2

def compute_line_loss(points, centroid, direction):
    """
    Compute the sum of squared distances (loss) from each point to the best-fit line.
    """
    centered = points - centroid
    proj_scalars = np.dot(centered, direction)
    projections = np.outer(proj_scalars, direction)
    residuals = centered - projections
    squared_distances = np.sum(residuals**2, axis=1)
    total_loss = np.sum(squared_distances)
    return total_loss

def plot_end_effector(json_file):
    timestamps, joint_positions = load_joint_data(json_file)
    
    if timestamps is None or joint_positions is None:
        return
    
    ee_positions = []
    orientations = []
    
    for idx, joint_set in enumerate(joint_positions):
        if len(joint_set) != 6:
            print(f"Skipping frame {idx}: Expected 6 joint values, got {len(joint_set)}")
            continue
        try:
            ee_pos, ee_orientation = forward_kinematics(joint_set)
            ee_positions.append(ee_pos[-1])  # Last position: end-effector
            orientations.append(ee_orientation[-1])
        except ValueError as e:
            print(f"Error at frame {idx}: {e}")
            continue
    
    if not ee_positions:
        print("No valid end-effector positions computed.")
        return

    ee_positions = np.array(ee_positions)
    
    # Fit best-fit line using least squares
    centroid, direction = fit_line_least_squares(ee_positions)
    endpoint1, endpoint2 = get_line_endpoints(ee_positions, centroid, direction)
    loss = compute_line_loss(ee_positions, centroid, direction)
    print(f"Total Sum of Squared Distances (Loss): {loss:.4f}")
    
    # 3D Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot end-effector trajectory
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
            '-', color='blue', label='End-Effector Path')
    
    # Mark start and end positions
    ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
               c='g', marker='o', s=100, label='Start')
    ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
               c='r', marker='o', s=100, label='End')
    
    # Plot the best-fit line (in red)
    ax.plot([endpoint1[0], endpoint2[0]],
            [endpoint1[1], endpoint2[1]],
            [endpoint1[2], endpoint2[2]],
            '-', color='red', linewidth=2, label='Best-Fit Line')
    
    # Optional: Plot orientation quivers at sparse intervals
    for i in range(0, len(ee_positions), max(1, len(ee_positions) // 10)):
        R_matrix = orientations[i]
        origin = ee_positions[i]
        ax.quiver(origin[0], origin[1], origin[2],
                  R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0],
                  color='r', length=0.05)
        ax.quiver(origin[0], origin[1], origin[2],
                  R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1],
                  color='g', length=0.05)
        ax.quiver(origin[0], origin[1], origin[2],
                  R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2],
                  color='b', length=0.05)
    
    # Set equal aspect ratio for all axes
    max_range = np.array([ee_positions[:, 0].max() - ee_positions[:, 0].min(),
                          ee_positions[:, 1].max() - ee_positions[:, 1].min(),
                          ee_positions[:, 2].max() - ee_positions[:, 2].min()]).max() / 2.0
    mid_x = (ee_positions[:, 0].max() + ee_positions[:, 0].min()) * 0.5
    mid_y = (ee_positions[:, 1].max() + ee_positions[:, 1].min()) * 0.5
    mid_z = (ee_positions[:, 2].max() + ee_positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('End-Effector Trajectory with Best-Fit Line')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot joint trajectories and end-effector path from JSON joint data.")
    parser.add_argument('--data', required=True, help='Path to the JSON file containing joint states.')
    args = parser.parse_args()
    
    json_file = args.data
    # Plot joint trajectories over time
    plot_joint_trajectories(json_file)
    # Plot 3D end-effector trajectory with best-fit line and compute the loss
    plot_end_effector(json_file)
