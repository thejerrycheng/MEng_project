import open3d as o3d
import numpy as np
import os
import glob
import cv2
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
# Path to a SINGLE clip you want to visualize
CLIP_PATH = "/media/jerry/SSD/final_data_mixed/test/diverse_2_20260121_214026_0_episode_0002_clip_00034"

# Camera Intrinsics (Approximate RealSense D435 values if unknown)
FX, FY = 605.0, 605.0
CX, CY = 320.0, 240.0
DEPTH_SCALE = 1000.0  # 1000 for RealSense (mm to meters)

# Hand-Eye Calibration (Offset from Robot End-Effector to Camera)
# Identity if your "robot" pose is already the camera optical frame.
# If camera is mounted on wrist, adjust z/y offset here.
H_EYE_TO_HAND = np.eye(4)
# Example: Camera is 5cm forward of the gripper
# H_EYE_TO_HAND[2, 3] = 0.05 
# =================================================

def load_pose(npy_path):
    """
    Reads the robot pose. 
    ASSUMPTION: .npy contains [x, y, z, qx, qy, qz, qw] or [x, y, z, r, p, y].
    Adjust the indices below to match your data format!
    """
    data = np.load(npy_path)
    # If data is a sequence, take the last one or matching index
    if len(data.shape) > 1: 
        pose_vec = data[-1] # Take last pose of the file
    else:
        pose_vec = data

    # Create 4x4 Matrix
    mat = np.eye(4)
    
    # 1. Translation (x, y, z)
    mat[:3, 3] = pose_vec[:3] 

    # 2. Rotation
    # Try to detect if Quaternion (4 values) or Euler (3 values)
    rot_data = pose_vec[3:7] # Assuming indices 3,4,5,6 are quat
    
    if len(rot_data) == 4:
        # Quaternion [x, y, z, w]
        r = R.from_quat(rot_data)
        mat[:3, :3] = r.as_matrix()
    elif len(rot_data) == 3:
        # Euler Angles
        r = R.from_euler('xyz', rot_data)
        mat[:3, :3] = r.as_matrix()
        
    return mat

def create_frustum(color=[1, 0, 0], scale=0.05):
    """Creates a wireframe pyramid representing the camera."""
    points = [
        [0, 0, 0], [1, 1, 2], [-1, 1, 2], [-1, -1, 2], [1, -1, 2]
    ]
    points = np.array(points) * scale
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

def main():
    print(f"Visualizing Clip: {os.path.basename(CLIP_PATH)}")

    # 1. Setup paths
    rgb_folder = os.path.join(CLIP_PATH, "rgb")
    depth_folder = os.path.join(CLIP_PATH, "depth")
    robot_folder = os.path.join(CLIP_PATH, "robot")

    if not os.path.exists(depth_folder):
        print("Error: 'depth' folder is required for 3D meshing!")
        return

    # Get sorted files
    rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")) + glob.glob(os.path.join(rgb_folder, "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.png")) + glob.glob(os.path.join(depth_folder, "*.npy")))
    robot_files = sorted(glob.glob(os.path.join(robot_folder, "*.npy")))

    # Open3D Intrinsic Object
    img = cv2.imread(rgb_files[0])
    H, W = img.shape[:2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, FX, FY, CX, CY)

    # Visualization Container
    geometries = []
    
    # Trajectory Line Points
    camera_centers = []

    print(f"Processing {len(rgb_files)} frames...")

    # 2. Iterate Frames
    # (Step by 2 or 3 to save memory if needed)
    for i in range(0, len(rgb_files), 2): 
        # A. Load Pose
        if i < len(robot_files):
            robot_pose = load_pose(robot_files[i])
            # Apply Hand-Eye Calibration
            cam_pose = robot_pose @ H_EYE_TO_HAND
        else:
            continue

        # B. Load Images
        color_raw = o3d.io.read_image(rgb_files[i])
        
        # Load Depth (Handle .npy or .png)
        d_path = depth_files[i]
        if d_path.endswith('.npy'):
            d_arr = np.load(d_path)
            # Normalize or Convert if necessary
            depth_raw = o3d.geometry.Image(d_arr.astype(np.float32))
        else:
            depth_raw = o3d.io.read_image(d_path)

        # C. Create RGBD Image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, 
            depth_scale=DEPTH_SCALE, 
            depth_trunc=1.5, # Clip depth beyond 1.5 meters
            convert_rgb_to_intensity=False
        )

        # D. Project to Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsics
        )
        
        # E. Transform Point Cloud to World Space
        pcd.transform(cam_pose)
        
        # Optimization: Voxel Downsample to keep view smooth
        pcd = pcd.voxel_down_sample(voxel_size=0.01) 
        geometries.append(pcd)

        # F. Add Camera Frustum Visualization
        frustum = create_frustum(color=[0, 1, 0], scale=0.05)
        frustum.transform(cam_pose)
        geometries.append(frustum)
        
        # Store center for trajectory line
        camera_centers.append(cam_pose[:3, 3])

    # 3. Add Trajectory Line
    if len(camera_centers) > 1:
        lines = [[j, j+1] for j in range(len(camera_centers)-1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(camera_centers)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0]) # Red Line
        geometries.append(line_set)

    # 4. Add Coordinate Frame (Origin)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(coord_frame)

    print("Done processing. Opening Interactive Viewer...")
    print("Controls:")
    print("  [Mouse Left]   Rotate")
    print("  [Mouse Wheel]  Zoom")
    print("  [Mouse Right]  Pan")
    
    # 5. Visualize
    o3d.visualization.draw_geometries(geometries, 
                                      window_name="Trajectory Reconstruction",
                                      width=1280, height=720)

if __name__ == "__main__":
    main()