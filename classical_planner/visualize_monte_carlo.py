import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import argparse
import random
import time

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

MAX_GEOMS = 20000  # Increased limit for clearer visuals

# ============================================================
# CSV loaders
# ============================================================
def load_path_csv(path):
    waypoints = []
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
# Helper Functions
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

def get_aesthetic_color():
    # Generate bright, pastel-like colors (high value/saturation) for visibility
    return np.array([
        random.uniform(0.5, 1.0), 
        random.uniform(0.5, 1.0), 
        random.uniform(0.5, 1.0), 
        0.6 # Semi-transparent
    ])

# ============================================================
# Main visualization
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Folder containing episodes")
    parser.add_argument("--num", type=int, default=-1, help="Number of trajectories to show")
    parser.add_argument("--random", action="store_true", help="Shuffle episode selection")
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
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except ValueError:
        print(f"[ERROR] Could not load XML at {XML_PATH}")
        return
        
    data = mujoco.MjData(model)

    trajectories = []
    for ep in episode_dirs:
        ep_root = os.path.join(args.path, ep)
        traj_path = os.path.join(ep_root, "path.csv")
        obs_path = os.path.join(ep_root, "obstacles.csv")
        
        traj = load_path_csv(traj_path)
        obs = load_obstacles_csv(obs_path)
        
        if traj:
            trajectories.append({
                "path": traj, 
                "obstacles": obs, 
                "color": get_aesthetic_color()
            })

    if not trajectories:
        print("[ERROR] No valid data found.")
        return

    # Sync environment to first episode
    apply_obstacles(model, trajectories[0]["obstacles"])
    mujoco.mj_forward(model, data)

    print(f"[INFO] Visualizing {len(trajectories)} trajectories.")
    print("      green sphere = START | red box = END")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            
            for traj in trajectories:
                path = traj["path"]
                color = traj["color"]
                
                if len(path) < 2: continue

                # 1. Draw Start Point (Green Sphere)
                if viewer.user_scn.ngeom < MAX_GEOMS:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE, 
                        np.array([0.015, 0, 0]), # Size
                        path[0], 
                        np.eye(3).flatten(), 
                        np.array([0, 1, 0, 1]) # Solid Green
                    )
                    viewer.user_scn.ngeom += 1

                # 2. Draw End Point (Red Box)
                if viewer.user_scn.ngeom < MAX_GEOMS:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_BOX, 
                        np.array([0.012, 0.012, 0.012]), # Size
                        path[-1], 
                        np.eye(3).flatten(), 
                        np.array([1, 0, 0, 1]) # Solid Red
                    )
                    viewer.user_scn.ngeom += 1

                # 3. Draw Path Lines (Continuous Strip)
                for i in range(len(path) - 1):
                    if viewer.user_scn.ngeom >= MAX_GEOMS: break
                    
                    # Connector (Line)
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, # Capsule looks smoother than LINE
                        np.array([0.003, 0, 0]), # Radius
                        np.zeros(3), np.zeros(9), # Pos/Rot ignored for connector
                        color
                    )
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.003,
                        path[i],
                        path[i+1]
                    )
                    viewer.user_scn.ngeom += 1
                    
                    # 4. Small Waypoint dots (Optional, adds texture)
                    if i % 5 == 0: # Only draw every 5th dot to save geoms
                        if viewer.user_scn.ngeom >= MAX_GEOMS: break
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.004, 0, 0]),
                            path[i],
                            np.eye(3).flatten(),
                            color
                        )
                        viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()