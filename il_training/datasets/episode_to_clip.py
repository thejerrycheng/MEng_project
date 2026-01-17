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
STRIDE = 1           # Sliding window stride (1 = move one frame at a time)

def get_episode_dirs(root_dir):
    """
    Finds all valid episode directories in the root folder.
    Assumes folders follow the naming convention ending in '_episode_XXXX'.
    """
    dirs = []
    if not os.path.exists(root_dir):
        print(f"Error: Source directory {root_dir} does not exist.")
        return []

    for d in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, d)
        # Check if it's a directory and looks like an episode folder
        if os.path.isdir(full_path) and "_episode_" in d:
            dirs.append(full_path)
    return dirs

def load_episode_data(ep_dir):
    """
    Loads raw data from a single episode directory.
    Returns: rgb_files, depth_files, joints, goal_img_path
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
        # Filter for joint columns (pos_0 ... pos_5)
        # Adjust logic if your CSV headers are different
        joint_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
        
        if len(joint_cols) != NUM_JOINTS:
            # Fallback: if columns aren't named pos_, take first 6
            if df.shape[1] >= NUM_JOINTS:
                 joints = df.iloc[:, :NUM_JOINTS].to_numpy(dtype=np.float32)
            else:
                print(f"Warning: {ep_dir} has incorrect joint columns.")
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

    # We need at least Sequence + Future steps to make ONE clip
    if min_len < (SEQ_LEN + FUTURE_STEPS):
        return None

    rgb_files = rgb_files[:min_len]
    depth_files = depth_files[:min_len]
    joints = joints[:min_len]
    
    # Goal Image is the very last frame of the VALID trajectory
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
    Saves a single processed window (clip) to the disk.
    Structure:
      output_dir/
        rgb/ (contains inputs 0..7 + goal.png)
        depth/ (contains inputs 0..7)
        robot/
          data.json (contains joint_seq, fut_delta)
    """
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "robot"), exist_ok=True)
    if clip_data["seq_depth_files"]:
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    # 1. Copy Input RGB Sequence
    for i, fname in enumerate(clip_data["seq_rgb_files"]):
        src = os.path.join(clip_data["base_rgb_dir"], fname)
        # Rename to input_0000.png, input_0001.png... for easy loading
        dst = os.path.join(output_dir, "rgb", f"input_{i:04d}.png")
        shutil.copy2(src, dst)

    # 2. Copy Input Depth Sequence (Optional)
    for i, fname in enumerate(clip_data["seq_depth_files"]):
        src = os.path.join(clip_data["base_depth_dir"], fname)
        dst = os.path.join(output_dir, "depth", f"input_{i:04d}.png")
        shutil.copy2(src, dst)

    # 3. Copy Goal Image
    # We save it as 'goal.png' in the rgb folder
    shutil.copy2(clip_data["goal_img_path"], os.path.join(output_dir, "rgb", "goal.png"))

    # 4. Save Robot Data (Joints + Delta Actions)
    # We save as JSON for easy loading later
    data_payload = {
        "joint_seq": clip_data["joint_seq"].tolist(),    # Input: (8, 6)
        "fut_delta": clip_data["fut_delta"].tolist(),    # Output: (15, 6)
    }
    
    with open(os.path.join(output_dir, "robot", "data.json"), "w") as f:
        json.dump(data_payload, f)

def process_and_save(episodes, dest_root, split_name):
    """
    Iterates through episodes, creates sliding windows, and saves them.
    """
    save_dir = os.path.join(dest_root, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Processing {split_name} set ({len(episodes)} episodes)...")
    
    clip_count = 0
    
    for ep in tqdm(episodes):
        ep_data = load_episode_data(ep)
        if ep_data is None:
            continue

        num_frames = len(ep_data["joints"])
        
        # Sliding Window Logic
        # We need `SEQ_LEN` frames for input AND `FUTURE_STEPS` frames for the label
        # Max start index = Total - (8 + 15) + 1
        max_start = num_frames - (SEQ_LEN + FUTURE_STEPS) + 1

        for start_idx in range(0, max_start, STRIDE):
            # Create unique clip folder name
            clip_name = f"{ep_data['ep_name']}_clip_{start_idx:05d}"
            clip_dir = os.path.join(save_dir, clip_name)

            # --- Extract Data for this Window ---
            
            # 1. Input Sequence (0 to 8)
            seq_rgb = ep_data["rgb_files"][start_idx : start_idx + SEQ_LEN]
            seq_depth = []
            if ep_data["depth_files"]:
                seq_depth = ep_data["depth_files"][start_idx : start_idx + SEQ_LEN]
            seq_joints = ep_data["joints"][start_idx : start_idx + SEQ_LEN]

            # 2. Future Action Calculation (8 to 23)
            # Current state is the LAST step of the input sequence (Index 7)
            # We want to predict change relative to this state
            current_q = ep_data["joints"][start_idx + SEQ_LEN - 1]
            
            # Future states are the next 15 steps
            future_q = ep_data["joints"][start_idx + SEQ_LEN : start_idx + SEQ_LEN + FUTURE_STEPS]
            
            # Calculate Delta (Action = Future - Current)
            # Broadcasting: (15, 6) - (6,)
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
            clip_count += 1

    print(f"Saved {clip_count} clips to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process raw robot data into sliding window training clips.")
    parser.add_argument("--source", type=str, required=True, help="Path to raw data (e.g. /media/jerry/SSD/data_no_obstacle)")
    parser.add_argument("--dest", type=str, required=True, help="Path to save processed data (e.g. /media/jerry/SSD/final_data)")
    args = parser.parse_args()

    # 1. Get List of Episodes
    all_episodes = get_episode_dirs(args.source)
    if not all_episodes:
        print("No episodes found. Exiting.")
        return

    # 2. Shuffle & Split (80/10/10)
    # We split by EPISODE, not by clip, to prevent data leakage.
    np.random.seed(42)
    np.random.shuffle(all_episodes)
    
    N = len(all_episodes)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    
    train_eps = all_episodes[:n_train]
    val_eps = all_episodes[n_train : n_train + n_val]
    test_eps = all_episodes[n_train + n_val:]

    print(f"Total Episodes: {N}")
    print(f"Train: {len(train_eps)} | Val: {len(val_eps)} | Test: {len(test_eps)}")

    # 3. Process
    process_and_save(train_eps, args.dest, "train")
    process_and_save(val_eps, args.dest, "val")
    process_and_save(test_eps, args.dest, "test")

    print("\nProcessing Complete.")
    print(f"Data saved to: {args.dest}")

if __name__ == "__main__":
    main()