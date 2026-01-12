import os
import argparse
import pickle
import pandas as pd
import numpy as np
import sys

# Adjust path to find kinematics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kinematics import IRISKinematics

NUM_JOINTS = 6
fk = IRISKinematics()

def list_episode_dirs(root, prefix):
    return sorted([
        os.path.join(root, d) for d in os.listdir(root)
        if d.startswith(prefix + "_episode_")
    ])

def load_episode(ep_dir):
    rgb_dir = os.path.join(ep_dir, "rgb")
    robot_csv = os.path.join(ep_dir, "robot", "joint_states.csv")
    
    if not os.path.isfile(robot_csv) or not os.path.isdir(rgb_dir):
        return None

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    df = pd.read_csv(robot_csv)
    
    # Filter columns
    pos_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
    joints = df[pos_cols].to_numpy(dtype=np.float32)

    # Sync lengths
    T = min(len(rgb_files), joints.shape[0])
    rgb_files = rgb_files[:T]
    joints = joints[:T]

    if T < 10: # Skip very short episodes
        return None

    goal_joint = joints[-1].copy()
    goal_xyz = fk.forward(np.rad2deg(goal_joint))

    return dict(
        rgb_dir=rgb_dir,
        rgb_files=rgb_files,
        joints=joints,
        goal_joint=goal_joint,
        goal_xyz=goal_xyz
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", required=True, help="Path to raw data")
    parser.add_argument("--prefix", required=True, help="e.g. 'iris'")
    parser.add_argument("--out_dir", default="processed_data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning {args.ssd_root}...")
    dirs = list_episode_dirs(args.ssd_root, args.prefix)
    episodes = [load_episode(d) for d in dirs]
    episodes = [e for e in episodes if e is not None]
    
    print(f"Loaded {len(episodes)} valid episodes.")

    # Split 80/10/10
    N = len(episodes)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train : n_train + n_val]
    test_eps = episodes[n_train + n_val:]

    # Save raw python lists (Manifests)
    with open(os.path.join(args.out_dir, "train_episodes.pkl"), "wb") as f:
        pickle.dump(train_eps, f)
    with open(os.path.join(args.out_dir, "val_episodes.pkl"), "wb") as f:
        pickle.dump(val_eps, f)
    with open(os.path.join(args.out_dir, "test_episodes.pkl"), "wb") as f:
        pickle.dump(test_eps, f)

    print(f"Saved manifests to {args.out_dir}")

if __name__ == "__main__":
    main()