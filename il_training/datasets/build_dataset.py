import os
import argparse
import pickle
import pandas as pd
import numpy as np
import sys

# NOTE: We removed IRISKinematics because the goal is now visual (last image),
# so we don't need to compute the Cartesian XYZ goal anymore.

NUM_JOINTS = 6

def list_episode_dirs(root, prefix):
    # Lists directories like 'iris_episode_001', 'iris_episode_002', etc.
    return sorted([
        os.path.join(root, d) for d in os.listdir(root)
        if d.startswith(prefix + "_episode_") and os.path.isdir(os.path.join(root, d))
    ])

def load_episode(ep_dir):
    """
    Parses a single episode folder. 
    Returns a dictionary containing the manifest of image files and joint data.
    """
    rgb_dir = os.path.join(ep_dir, "rgb")
    robot_csv = os.path.join(ep_dir, "robot", "joint_states.csv")
    
    # 1. Basic Validation
    if not os.path.isfile(robot_csv):
        print(f"Skipping {ep_dir}: Missing joint_states.csv")
        return None
    if not os.path.isdir(rgb_dir):
        print(f"Skipping {ep_dir}: Missing rgb folder")
        return None

    # 2. Load Data
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    try:
        df = pd.read_csv(robot_csv)
    except Exception as e:
        print(f"Skipping {ep_dir}: CSV Corrupt ({e})")
        return None
    
    # 3. Filter Joint Columns
    # Assumes columns like 'pos_0', 'pos_1', ... or matches your CSV format
    pos_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
    if len(pos_cols) != NUM_JOINTS:
        print(f"Skipping {ep_dir}: Found {len(pos_cols)} joints, expected {NUM_JOINTS}")
        return None

    joints = df[pos_cols].to_numpy(dtype=np.float32)

    # 4. Synchronization
    # We truncate to the length of the shorter modality (usually images are fewer if dropped)
    T = min(len(rgb_files), joints.shape[0])
    
    # Validation: Skip very short episodes (e.g., failed captures)
    if T < 10: 
        print(f"Skipping {ep_dir}: Too short (T={T})")
        return None

    rgb_files = rgb_files[:T]
    joints = joints[:T]

    # 5. Extract Metadata
    # goal_joint is kept for potential auxiliary loss, though goal_image is the main driver
    goal_joint = joints[-1].copy()

    # NOTE: goal_xyz calculation is removed.
    # The dataset loader will simply grab rgb_files[-1] as the goal.

    return dict(
        rgb_dir=rgb_dir,        # Path to images
        rgb_files=rgb_files,    # List of filenames ['0.png', '1.png'...]
        joints=joints,          # Numpy array (T, 6)
        goal_joint=goal_joint,  # (6,)
        length=T
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", required=True, help="Path to raw data folders")
    parser.add_argument("--prefix", required=True, help="Episode prefix, e.g. 'iris'")
    parser.add_argument("--out_dir", default="processed_data", help="Output folder for .pkl files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning {args.ssd_root} for prefix '{args.prefix}'...")
    dirs = list_episode_dirs(args.ssd_root, args.prefix)
    
    if not dirs:
        print("No episode directories found! Check your path and prefix.")
        return

    print(f"Found {len(dirs)} directories. Processing...")
    
    episodes = []
    for d in dirs:
        ep = load_episode(d)
        if ep is not None:
            episodes.append(ep)
    
    print(f"Successfully loaded {len(episodes)} valid episodes.")

    if len(episodes) == 0:
        print("No valid episodes found after filtering. Exiting.")
        return

    # Split Data (80% Train, 10% Val, 10% Test)
    # Shuffle for randomness before splitting
    np.random.seed(42)
    np.random.shuffle(episodes)

    N = len(episodes)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train : n_train + n_val]
    test_eps = episodes[n_train + n_val:]

    print(f"Splits: Train={len(train_eps)}, Val={len(val_eps)}, Test={len(test_eps)}")

    # Save Pickle Manifests
    # These contain lightweight paths/metadata, not heavy image data
    with open(os.path.join(args.out_dir, "train_episodes.pkl"), "wb") as f:
        pickle.dump(train_eps, f)
    with open(os.path.join(args.out_dir, "val_episodes.pkl"), "wb") as f:
        pickle.dump(val_eps, f)
    with open(os.path.join(args.out_dir, "test_episodes.pkl"), "wb") as f:
        pickle.dump(test_eps, f)

    print(f"Saved manifests to {args.out_dir}")

if __name__ == "__main__":
    main()