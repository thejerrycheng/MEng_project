import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import argparse
import random

# ============================================================
# Path resolution
# ============================================================
CURR_DIR = os.path.dirname(os.path.abspath(__file__))          
PROJECT_ROOT = os.path.dirname(CURR_DIR)                       

# Ensure this points to the EXACT same XML used in generation
XML_PATH = os.path.join(
    PROJECT_ROOT,
    "mujoco_sim",
    "assets",
    "scene2.xml"
)

MAX_GEOMS = 15000  # High limit for complex RRT trees

# ============================================================
# CSV loaders
# ============================================================
def load_path_csv(path):
    waypoints = []
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coordinates are pulled directly from generated 'x', 'y', 'z' columns
            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            waypoints.append(pos)
    return waypoints

def load_obstacles_csv(path):
    obstacles = []
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obstacles.append({
                "name": row["name"],
                "pos": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "yaw": float(row["yaw"])
            })
    return obstacles

# ============================================================
# Environment Sync
# ============================================================
def apply_obstacles(model, obstacles):
    """Overrides XML default positions with the generated CSV positions."""
    if not obstacles: return
    for obs in obstacles:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obs["name"])
        if gid >= 0:
            model.geom_pos[gid] = obs["pos"]
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), obs["yaw"])
            model.geom_quat[gid] = quat
        else:
            print(f"[WARN] Obstacle '{obs['name']}' in CSV not found in XML worldbody.")

def get_random_color():
    return np.array([random.random(), random.random(), random.random(), 0.7])

# ============================================================
# Main visualization
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Folder: random_rrt_dataset2")
    parser.add_argument("--num", type=int, default=-1, help="Trajectories to show")
    parser.add_argument("--random", action="store_true", help="Shuffle episode selection")
    parser.add_argument("--lines", action="store_true", help="Draw connecting lines")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"[ERROR] Path {args.path} is not a directory.")
        return

    # Filter and sort episode directories
    episode_dirs = sorted([d for d in os.listdir(args.path) if d.startswith("episode_")])
    
    if args.random:
        random.shuffle(episode_dirs)
    if args.num > 0:
        episode_dirs = episode_dirs[:args.num]

    # Load Model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    trajectories = []
    for ep in episode_dirs:
        ep_root = os.path.join(args.path, ep)
        traj = load_path_csv(os.path.join(ep_root, "path.csv"))
        obs = load_obstacles_csv(os.path.join(ep_root, "obstacles.csv"))
        
        if traj:
            trajectories.append({"path": traj, "obstacles": obs, "color": get_random_color()})

    if not trajectories:
        print("[ERROR] No valid data found.")
        return

    # SYNC ENVIRONMENT: Use obstacle data from the first trajectory to set the scene
    # (Assuming all trajectories in one folder share the same obstacle layout)
    apply_obstacles(model, trajectories[0]["obstacles"])
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"[INFO] Showing {len(trajectories)} paths. Obstacles synced to {episode_dirs[0]}")
        
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            for traj in trajectories:
                path = traj["path"]
                color = traj["color"]

                # Draw spheres at waypoints
                for p in path:
                    if viewer.user_scn.ngeom >= MAX_GEOMS: break
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                       mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.005, 0, 0]),
                                       p, np.eye(3).flatten(), color)
                    viewer.user_scn.ngeom += 1

                # Draw lines
                if args.lines:
                    for i in range(len(path)-1):
                        if viewer.user_scn.ngeom >= MAX_GEOMS: break
                        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                            mujoco.mjtGeom.mjGEOM_LINE, 0.002, 
                                            path[i], path[i+1])
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = color
                        viewer.user_scn.ngeom += 1

            viewer.sync()

if __name__ == "__main__":
    main()