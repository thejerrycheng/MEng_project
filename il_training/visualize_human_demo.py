import os
import argparse
import numpy as np
import pandas as pd
import time
import yaml

import mujoco
import mujoco.viewer


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(
    description="Visualize human demonstration trajectories in MuJoCo (auto-play episodes)"
)
parser.add_argument("--data_root", type=str, required=True,
                    help="Path to processed_data directory")
parser.add_argument("--bag_prefix", type=str, required=True,
                    help="Episode prefix to visualize")
parser.add_argument("--xml_path", type=str, default="mujoco_sim/assets/iris.xml",
                    help="Path to IRIS MuJoCo XML model")
parser.add_argument("--calib_yaml", type=str,
                    default="meng_ws/src/unitree_arm_ros/config/calibration.yaml",
                    help="Path to calibration.yaml")
parser.add_argument("--max_eps", type=int, default=5)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--pause_between", type=float, default=1.0,
                    help="Seconds to pause between episodes")

args = parser.parse_args()


# ----------------------------
# Load calibration offsets
# ----------------------------
def load_calibration_offsets(yaml_path):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    offs = cfg["joint_offsets"]
    offsets = np.array([
        offs["joint_1"],
        offs["joint_2"],
        offs["joint_3"],
        offs["joint_4"],
        offs["joint_5"],
        offs["joint_6"],
    ], dtype=np.float32)

    print("Loaded joint calibration offsets (rad):")
    print(offsets)
    return offsets


# ----------------------------
# Episode discovery
# ----------------------------
def list_episode_dirs(data_root, bag_prefix):
    eps = []
    for name in sorted(os.listdir(data_root)):
        if name.startswith(bag_prefix + "_episode_"):
            ep = os.path.join(data_root, name)
            if os.path.isdir(ep):
                eps.append(ep)
    return eps


def load_episode_joint_csv(ep_dir):
    csv_path = os.path.join(ep_dir, "robot", "joint_states.csv")
    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    pos_cols = pos_cols[:6]

    q_real = df[pos_cols].to_numpy(dtype=np.float32)  # (T,6)
    return q_real


# ----------------------------
# Draw gray EE trace
# ----------------------------
def draw_gray_trace(viewer, points):
    scn = viewer.user_scn
    scn.ngeom = 0

    max_pts = min(len(points), 400)

    for i in range(max_pts):
        p = points[-max_pts + i]

        mujoco.mjv_initGeom(
            scn.geoms[i],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.003, 0, 0]),
            p,
            np.eye(3).flatten(),
            np.array([0.6, 0.6, 0.6, 0.8])  # gray
        )
        scn.ngeom += 1


# ----------------------------
# Main visualization loop
# ----------------------------
def visualize_all_episodes(model, data, episodes, q_offsets, fps, pause_between):
    dt = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nMuJoCo window opened — playing episodes sequentially.")
        print("Close the window only when finished.\n")

        for ep_idx, ep_dir in enumerate(episodes):
            print(f"▶ Episode {ep_idx+1}/{len(episodes)}: {os.path.basename(ep_dir)}")

            q_real = load_episode_joint_csv(ep_dir)
            if q_real is None:
                print("  Skipped (no joint_states.csv)")
                continue

            ee_trace = []

            for t in range(len(q_real)):
                if not viewer.is_running():
                    return

                # Apply calibration
                q_mj = q_real[t] - q_offsets
                data.qpos[:6] = q_mj
                mujoco.mj_forward(model, data)

                # Record EE position
                ee_pos = data.body("wrist2").xpos.copy()
                ee_trace.append(ee_pos)

                draw_gray_trace(viewer, ee_trace)

                viewer.sync()
                time.sleep(dt)

            # Small pause before next episode
            pause_t0 = time.time()
            while time.time() - pause_t0 < pause_between:
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(0.01)

        print("\nAll episodes finished. Close window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


# ----------------------------
# Main
# ----------------------------
def main():
    if not os.path.isfile(args.xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {args.xml_path}")

    q_offsets = load_calibration_offsets(args.calib_yaml)

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    episode_dirs = list_episode_dirs(args.data_root, args.bag_prefix)
    if len(episode_dirs) == 0:
        raise RuntimeError("No episodes found")

    episode_dirs = episode_dirs[:args.max_eps]

    print(f"\nFound {len(episode_dirs)} episodes to play")

    visualize_all_episodes(
        model, data,
        episode_dirs,
        q_offsets,
        fps=args.fps,
        pause_between=args.pause_between
    )


if __name__ == "__main__":
    main()
