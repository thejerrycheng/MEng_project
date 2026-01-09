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
CURR_DIR = os.path.dirname(os.path.abspath(__file__))          # classical_planner/
PROJECT_ROOT = os.path.dirname(CURR_DIR)                       # MEng_project/

XML_PATH = os.path.join(
    PROJECT_ROOT,
    "mujoco_sim",
    "assets",
    "scene2.xml"
)

RRT_DATASET_ROOT = os.path.join(CURR_DIR, "random_rrt_dataset2")
APF_DATASET_ROOT = os.path.join(CURR_DIR, "APF_path")

MAX_GEOMS = 4000

# ============================================================
# CSV loaders
# ============================================================
def load_path_csv(path):
    waypoints = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = np.array([
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            ])
            waypoints.append(pos)
    return waypoints


def load_obstacles_csv(path):
    obstacles = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obstacles.append({
                "name": row["name"],
                "pos": np.array([
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"])
                ]),
                "yaw": float(row["yaw"])
            })
    return obstacles

# ============================================================
# Apply obstacle placements
# ============================================================
def apply_obstacles(model, obstacles):
    for obs in obstacles:
        gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, obs["name"]
        )
        if gid < 0:
            continue

        model.geom_pos[gid] = obs["pos"]

        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), obs["yaw"])
        model.geom_quat[gid] = quat

# ============================================================
# Main visualization
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        type=int,
        default=20,
        help="Number of trajectories to visualize (default: 20)"
    )
    parser.add_argument(
        "--apf",
        action="store_true",
        help="Use APF dataset instead of RRT"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample episodes instead of first N"
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Draw thin line segments between waypoints"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Dataset selection
    # --------------------------------------------------------
    if args.apf:
        dataset_root = APF_DATASET_ROOT
        dataset_name = "APF"
    else:
        dataset_root = RRT_DATASET_ROOT
        dataset_name = "RRT"

    episodes = sorted(
        d for d in os.listdir(dataset_root)
        if d.startswith("episode_")
    )

    assert len(episodes) > 0, f"No episodes found in {dataset_root}"

    if args.random:
        episodes = random.sample(
            episodes, min(args.num, len(episodes))
        )
    else:
        episodes = episodes[:args.num]

    print(f"[INFO] Dataset      : {dataset_name}")
    print(f"[INFO] Trajectories : {len(episodes)}")

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    id_mat = np.eye(3).flatten()
    zero_v = np.zeros(3)

    # --------------------------------------------------------
    # Load all trajectories
    # --------------------------------------------------------
    trajectories = []

    for ep in episodes:
        ep_dir = os.path.join(dataset_root, ep)

        path_csv = os.path.join(ep_dir, "path.csv")
        obs_csv = os.path.join(ep_dir, "obstacles.csv")

        if not os.path.exists(path_csv) or not os.path.exists(obs_csv):
            continue

        traj = load_path_csv(path_csv)
        obstacles = load_obstacles_csv(obs_csv)

        trajectories.append({
            "episode": ep,
            "path": traj,
            "obstacles": obstacles
        })

    assert len(trajectories) > 0, "No valid trajectories loaded."

    # --------------------------------------------------------
    # Use obstacles from FIRST episode (shared scene)
    # --------------------------------------------------------
    apply_obstacles(model, trajectories[0]["obstacles"])
    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.user_scn.ngeom = 0

            # ------------------------------------------------
            # Draw trajectories
            # ------------------------------------------------
            for traj in trajectories:
                path = traj["path"]

                # Light gray dots
                for p in path:
                    if viewer.user_scn.ngeom >= MAX_GEOMS:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.006, 0, 0]),
                        p,
                        id_mat,
                        np.array([0.75, 0.75, 0.75, 0.25])
                    )
                    viewer.user_scn.ngeom += 1

                # Optional connecting lines
                if args.lines:
                    for p0, p1 in zip(path[:-1], path[1:]):
                        if viewer.user_scn.ngeom >= MAX_GEOMS:
                            break
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            zero_v,
                            zero_v,
                            id_mat,
                            np.array([0.6, 0.6, 0.6, 0.25])
                        )
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            0.002,
                            p0,
                            p1
                        )
                        viewer.user_scn.ngeom += 1

            viewer.sync()

if __name__ == "__main__":
    main()
