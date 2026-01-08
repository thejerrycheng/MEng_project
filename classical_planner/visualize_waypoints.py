import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import time
import argparse

# ============================================================
# Path resolution
# ============================================================
CURR_DIR = os.path.dirname(os.path.abspath(__file__))          # classical_planner/
PROJECT_ROOT = os.path.dirname(CURR_DIR)                       # MEng_project/

XML_PATH = os.path.join(
    PROJECT_ROOT,
    "mujoco_sim",
    "assets",
    "obstacle.xml"
)

RRT_DATASET_ROOT = os.path.join(CURR_DIR, "random_rrt_dataset")
APF_DATASET_ROOT = os.path.join(CURR_DIR, "APF_path")

# ============================================================
# CSV loaders
# ============================================================
def load_path_csv(path):
    waypoints = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = np.array([
                float(row["q1"]), float(row["q2"]), float(row["q3"]),
                float(row["q4"]), float(row["q5"]), float(row["q6"])
            ])
            pos = np.array([
                float(row["x"]), float(row["y"]), float(row["z"])
            ])
            waypoints.append((q, pos))
    return waypoints


def load_obstacles_csv(path):
    obstacles = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obstacles.append({
                "name": row["name"],
                "type": row["type"],
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
            print(f"[WARN] geom not found: {obs['name']}")
            continue

        model.geom_pos[gid] = obs["pos"]

        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), obs["yaw"])
        model.geom_quat[gid] = quat

# ============================================================
# Visualization
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index (0 -> episode_000, 1 -> episode_001, etc.)"
    )

    parser.add_argument(
        "--rrt",
        action="store_true",
        help="Use RRT dataset (default)"
    )

    parser.add_argument(
        "--apf",
        action="store_true",
        help="Use Artificial Potential Field dataset"
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help="Animate robot along the path"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier"
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

    ep_name = f"episode_{args.episode:03d}"
    ep_dir = os.path.join(dataset_root, ep_name)

    assert os.path.isdir(ep_dir), f"{dataset_name} episode not found: {ep_dir}"

    path_csv = os.path.join(ep_dir, "path.csv")
    obs_csv = os.path.join(ep_dir, "obstacles.csv")

    assert os.path.exists(path_csv), f"Missing path.csv in {ep_dir}"
    assert os.path.exists(obs_csv), f"Missing obstacles.csv in {ep_dir}"

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    waypoints = load_path_csv(path_csv)
    obstacles = load_obstacles_csv(obs_csv)

    print(f"[INFO] Dataset : {dataset_name}")
    print(f"[INFO] Episode : {ep_name}")
    print(f"[INFO] Waypoints: {len(waypoints)}")
    print(f"[INFO] Obstacles: {len(obstacles)}")

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount"
    )

    apply_obstacles(model, obstacles)
    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    id_mat = np.eye(3).flatten()
    zero_v = np.zeros(3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wp_idx = 0

        while viewer.is_running():
            viewer.user_scn.ngeom = 0

            # ------------------------------------------------
            # Draw all waypoints (red)
            # ------------------------------------------------
            # Draw all waypoints (light gray)
            for _, pos in waypoints:
                if viewer.user_scn.ngeom >= 1500:
                    break
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.005, 0, 0]),
                    pos,
                    id_mat,
                    np.array([0.75, 0.75, 0.75, 0.5])  # light gray, semi-transparent
                )
                viewer.user_scn.ngeom += 1


            # ------------------------------------------------
            # Current waypoint (green)
            # ------------------------------------------------
            q, pos = waypoints[wp_idx]

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]),
                pos,
                id_mat,
                np.array([0.0, 1.0, 0.0, 0.9])
            )
            viewer.user_scn.ngeom += 1

            # ------------------------------------------------
            # Robot playback
            # ------------------------------------------------
            if args.play:
                data.qpos[:6] = q
                data.qvel[:6] = 0
                mujoco.mj_forward(model, data)

                wp_idx += 1
                if wp_idx >= len(waypoints):
                    wp_idx = len(waypoints) - 1

            viewer.sync()
            time.sleep(model.opt.timestep / max(1e-3, args.speed))


if __name__ == "__main__":
    main()
