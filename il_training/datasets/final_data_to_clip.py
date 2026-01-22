import os
import shutil
import argparse
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# --------------------------
# Configuration
# --------------------------
SEQ_LEN = 8          # Input history length
FUTURE_STEPS = 15    # Prediction horizon
NUM_JOINTS = 6       # Number of robot joints
STRIDE = 1           # Sliding window stride

def get_episode_dirs(root_dirs):
    """
    Finds all valid episode directories in a LIST of root folders.
    """
    episode_list = []
    
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    for root in root_dirs:
        if not os.path.exists(root):
            print(f"Warning: Source directory {root} does not exist. Skipping.")
            continue
            
        print(f"Scanning {root}...")
        for d in sorted(os.listdir(root)):
            full_path = os.path.join(root, d)
            # Check if it looks like an episode folder (contains 'episode')
            if os.path.isdir(full_path) and "episode" in d:
                episode_list.append(full_path)
    
    return episode_list

def load_episode_data(ep_dir):
    """
    Loads raw data from a single episode directory.
    """
    rgb_dir = os.path.join(ep_dir, "rgb")
    depth_dir = os.path.join(ep_dir, "depth")
    csv_path = os.path.join(ep_dir, "robot", "joint_states.csv")

    # 1. Validation
    if not os.path.exists(rgb_dir) or not os.path.exists(csv_path):
        return None

    # 2. Get File Lists
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    depth_files = []
    if os.path.exists(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    # 3. Load Joints
    try:
        df = pd.read_csv(csv_path)
        # Dynamic column finding for 'pos_'
        joint_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
        
        if len(joint_cols) != NUM_JOINTS:
            # Fallback for datasets that might use different headers
            if df.shape[1] >= NUM_JOINTS:
                 joints = df.iloc[:, :NUM_JOINTS].to_numpy(dtype=np.float32)
            else:
                return None
        else:
            joints = df[joint_cols].to_numpy(dtype=np.float32)
            
    except Exception as e:
        print(f"Error reading CSV in {ep_dir}: {e}")
        return None

    # 4. Synchronization (Truncate to shortest length)
    min_len = min(len(rgb_files), len(joints))
    if len(depth_files) > 0:
        min_len = min(min_len, len(depth_files))

    if min_len < (SEQ_LEN + FUTURE_STEPS):
        return None

    rgb_files = rgb_files[:min_len]
    depth_files = depth_files[:min_len]
    joints = joints[:min_len]
    
    # Goal Image is the last frame of the VALID trajectory
    goal_img_path = os.path.join(rgb_dir, rgb_files[-1])

    return {
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "joints": joints,
        "goal_img_path": goal_img_path,
        "base_rgb_dir": rgb_dir,
        "base_depth_dir": depth_dir,
        "ep_name": os.path.basename(ep_dir)
    }

def save_clip(clip_data, output_dir):
    """
    Saves a single processed clip.
    """
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "robot"), exist_ok=True)
    if clip_data["seq_depth_files"]:
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    # Copy RGB Images
    for i, fname in enumerate(clip_data["seq_rgb_files"]):
        src = os.path.join(clip_data["base_rgb_dir"], fname)
        dst = os.path.join(output_dir, "rgb", f"input_{i:04d}.png")
        shutil.copy2(src, dst)

    # Copy Depth Images
    for i, fname in enumerate(clip_data["seq_depth_files"]):
        src = os.path.join(clip_data["base_depth_dir"], fname)
        dst = os.path.join(output_dir, "depth", f"input_{i:04d}.png")
        shutil.copy2(src, dst)

    # Copy Goal
    shutil.copy2(clip_data["goal_img_path"], os.path.join(output_dir, "rgb", "goal.png"))

    # Save Data
    data_payload = {
        "joint_seq": clip_data["joint_seq"].tolist(),
        "fut_delta": clip_data["fut_delta"].tolist(),
    }
    with open(os.path.join(output_dir, "robot", "data.json"), "w") as f:
        json.dump(data_payload, f)

def process_batch(episodes, dest_root, split_name):
    """
    Helper to process a list of episodes into a specific split folder (train/val/test).
    """
    save_dir = os.path.join(dest_root, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    for ep in tqdm(episodes, desc=f"Processing {split_name}"):
        ep_data = load_episode_data(ep)
        if ep_data is None: continue

        num_frames = len(ep_data["joints"])
        max_start = num_frames - (SEQ_LEN + FUTURE_STEPS) + 1

        for start_idx in range(0, max_start, STRIDE):
            clip_name = f"{ep_data['ep_name']}_clip_{start_idx:05d}"
            clip_dir = os.path.join(save_dir, clip_name)

            # Extract Window
            seq_rgb = ep_data["rgb_files"][start_idx : start_idx + SEQ_LEN]
            seq_depth = ep_data["depth_files"][start_idx : start_idx + SEQ_LEN] if ep_data["depth_files"] else []
            seq_joints = ep_data["joints"][start_idx : start_idx + SEQ_LEN]

            current_q = ep_data["joints"][start_idx + SEQ_LEN - 1]
            future_q = ep_data["joints"][start_idx + SEQ_LEN : start_idx + SEQ_LEN + FUTURE_STEPS]
            fut_delta = future_q - current_q

            clip_payload = {
                "seq_rgb_files": seq_rgb,
                "seq_depth_files": seq_depth,
                "base_rgb_dir": ep_data["base_rgb_dir"],
                "base_depth_dir": ep_data["base_depth_dir"],
                "goal_img_path": ep_data["goal_img_path"],
                "joint_seq": seq_joints,
                "fut_delta": fut_delta
            }

            save_clip(clip_payload, clip_dir)
            count += 1
    return count

def generate_dataset(episode_list, output_path):
    """
    Splits episodes into Train/Val/Test and processes them.
    """
    if os.path.exists(output_path):
        print(f"⚠️  Output directory {output_path} already exists.")
        val = input("    Type 'y' to delete and overwrite, or anything else to skip: ")
        if val.lower() == 'y':
            shutil.rmtree(output_path)
        else:
            print("    Skipping...")
            return

    print(f"\n🚀 Generating Dataset at: {output_path}")
    print(f"   Input Episodes: {len(episode_list)}")

    # Shuffle and Split
    np.random.seed(42)
    shuffled_eps = np.array(episode_list) # Copy to avoid messing up original list
    np.random.shuffle(shuffled_eps)
    
    N = len(shuffled_eps)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    
    train_eps = shuffled_eps[:n_train]
    val_eps = shuffled_eps[n_train : n_train + n_val]
    test_eps = shuffled_eps[n_train + n_val:]

    print(f"   Split: {len(train_eps)} Train | {len(val_eps)} Val | {len(test_eps)} Test")

    # Process
    c_train = process_batch(train_eps, output_path, "train")
    c_val = process_batch(val_eps, output_path, "val")
    c_test = process_batch(test_eps, output_path, "test")

    print(f"✅ Finished {output_path}")
    print(f"   Clips Generated: {c_train + c_val + c_test}\n")

def main():
    # --------------------------
    # Hardcoded Paths based on your LS
    # --------------------------
    # Inputs
    SRC_NO_OBSTACLE = "/media/jerry/SSD/trajectory_no_obstacle"
    SRC_WITH_OBSTACLE = "/media/jerry/SSD/trajectory_with_obstacle"
    
    # Outputs
    OUT_NO_OBSTACLE = "/media/jerry/SSD/final_data_no_obstacle"
    OUT_MIXED       = "/media/jerry/SSD/final_data_mixed"

    print("=== Dataset Generation Tool ===")
    
    # 1. Get Episode Lists
    eps_no_obs = get_episode_dirs(SRC_NO_OBSTACLE)
    eps_with_obs = get_episode_dirs(SRC_WITH_OBSTACLE)
    
    if not eps_no_obs:
        print("Error: No episodes found in no_obstacle path.")
        return

    # ---------------------------------------------------------
    # DATASET 1: Pure No Obstacle
    # ---------------------------------------------------------
    generate_dataset(eps_no_obs, OUT_NO_OBSTACLE)

    # ---------------------------------------------------------
    # DATASET 2: Mixed (No Obs + With Obs)
    # ---------------------------------------------------------
    eps_mixed = eps_no_obs + eps_with_obs
    generate_dataset(eps_mixed, OUT_MIXED)

if __name__ == "__main__":
    main()