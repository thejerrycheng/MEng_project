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

def get_episode_dirs(root_dir):
    dirs = []
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found.")
        return []
    for d in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, d)
        if os.path.isdir(full_path) and "_episode_" in d:
            dirs.append(full_path)
    return dirs

def load_episode_data(ep_dir):
    """
    Loads RGB images and Joint states with robust error handling.
    """
    rgb_dir = os.path.join(ep_dir, "rgb")
    csv_path = os.path.join(ep_dir, "robot", "joint_states.csv")

    # 1. Existence Check
    if not os.path.exists(rgb_dir) or not os.path.exists(csv_path):
        return None

    # 2. Empty File Check (The Fix)
    if os.path.getsize(csv_path) == 0:
        print(f"[Warning] Skipping empty CSV: {csv_path}")
        return None

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    
    try:
        # 3. Safe CSV Loading
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"[Warning] CSV is corrupted/empty: {csv_path}")
            return None

        # Attempt to find columns like pos_0, pos_1...
        joint_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
        
        # Fallback if columns are named differently (e.g. 0, 1, 2...)
        if len(joint_cols) != NUM_JOINTS:
            if df.shape[1] >= NUM_JOINTS:
                joints = df.iloc[:, :NUM_JOINTS].to_numpy(dtype=np.float32)
            else:
                print(f"[Warning] Not enough joint columns in {csv_path}")
                return None
        else:
            joints = df[joint_cols].to_numpy(dtype=np.float32)
            
    except Exception as e:
        print(f"[Error] Failed reading {csv_path}: {e}")
        return None

    # Sync lengths
    min_len = min(len(rgb_files), len(joints))
    if min_len < (SEQ_LEN + FUTURE_STEPS):
        return None

    rgb_files = rgb_files[:min_len]
    joints = joints[:min_len]
    
    # Goal image is the last frame of the episode
    goal_img_path = os.path.join(rgb_dir, rgb_files[-1])

    return {
        "rgb_files": rgb_files,
        "joints": joints,
        "goal_img_path": goal_img_path,
        "base_rgb_dir": rgb_dir,
        "ep_name": os.path.basename(ep_dir)
    }

def save_clip(clip_data, output_dir):
    """
    Saves the data in the format required by the Visual-Absolute model.
    """
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "robot"), exist_ok=True)

    # 1. Save RGB Input Sequence (Images 0-7)
    for i, fname in enumerate(clip_data["seq_rgb_files"]):
        src = os.path.join(clip_data["base_rgb_dir"], fname)
        dst = os.path.join(output_dir, "rgb", f"input_{i:04d}.png")
        shutil.copy2(src, dst)

    # 2. Save Goal Image
    shutil.copy2(clip_data["goal_img_path"], os.path.join(output_dir, "rgb", "goal.png"))

    # 3. Save Robot Data (JSON)
    data_payload = {
        # Input History (Saved for compatibility)
        "joint_seq": clip_data["joint_seq"].tolist(), 
        
        # TARGET OUTPUT: Absolute Future Positions (Next 15 steps)
        "fut_absolute": clip_data["fut_absolute"].tolist(),
    }
    
    with open(os.path.join(output_dir, "robot", "data.json"), "w") as f:
        json.dump(data_payload, f)

def process_and_save(episodes, dest_root, split_name):
    save_dir = os.path.join(dest_root, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Processing {split_name} ({len(episodes)} episodes)...")
    
    for ep in tqdm(episodes):
        ep_data = load_episode_data(ep)
        if ep_data is None:
            continue

        num_frames = len(ep_data["joints"])
        max_start = num_frames - (SEQ_LEN + FUTURE_STEPS) + 1

        for start_idx in range(0, max_start, STRIDE):
            clip_name = f"{ep_data['ep_name']}_clip_{start_idx:05d}"
            clip_dir = os.path.join(save_dir, clip_name)

            # --- Inputs ---
            seq_rgb = ep_data["rgb_files"][start_idx : start_idx + SEQ_LEN]
            seq_joints = ep_data["joints"][start_idx : start_idx + SEQ_LEN]

            # --- Outputs (Absolute) ---
            # Future steps 1 to 15 (indices 8 to 23)
            future_q = ep_data["joints"][start_idx + SEQ_LEN : start_idx + SEQ_LEN + FUTURE_STEPS]

            clip_payload = {
                "seq_rgb_files": seq_rgb,
                "base_rgb_dir": ep_data["base_rgb_dir"],
                "goal_img_path": ep_data["goal_img_path"],
                "joint_seq": seq_joints,
                "fut_absolute": future_q
            }

            save_clip(clip_payload, clip_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Raw data folder")
    parser.add_argument("--dest", type=str, required=True, help="Output processed folder")
    args = parser.parse_args()

    # 1. Find Episodes
    all_episodes = get_episode_dirs(args.source)
    if not all_episodes:
        return

    # 2. Shuffle & Split (80/10/10)
    np.random.seed(42)
    np.random.shuffle(all_episodes)
    
    N = len(all_episodes)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    
    train_eps = all_episodes[:n_train]
    val_eps = all_episodes[n_train : n_train + n_val]
    test_eps = all_episodes[n_train + n_val:]

    print(f"Total: {N} | Train: {len(train_eps)} | Val: {len(val_eps)} | Test: {len(test_eps)}")

    # 3. Process
    process_and_save(train_eps, args.dest, "train")
    process_and_save(val_eps, args.dest, "val")
    process_and_save(test_eps, args.dest, "test")

    print(f"Done! Saved to {args.dest}")

if __name__ == "__main__":
    main()