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
    rgb_dir = os.path.join(ep_dir, "rgb")
    csv_path = os.path.join(ep_dir, "robot", "joint_states.csv")

    if not os.path.exists(rgb_dir) or not os.path.exists(csv_path): return None
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    
    try:
        df = pd.read_csv(csv_path)
        joint_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
        if len(joint_cols) != NUM_JOINTS:
            if df.shape[1] >= NUM_JOINTS: joints = df.iloc[:, :NUM_JOINTS].to_numpy(dtype=np.float32)
            else: return None
        else:
            joints = df[joint_cols].to_numpy(dtype=np.float32)
    except: return None

    min_len = min(len(rgb_files), len(joints))
    if min_len < (SEQ_LEN + FUTURE_STEPS): return None

    rgb_files = rgb_files[:min_len]
    joints = joints[:min_len]
    goal_img_path = os.path.join(rgb_dir, rgb_files[-1])

    return {
        "rgb_files": rgb_files,
        "joints": joints,
        "goal_img_path": goal_img_path,
        "base_rgb_dir": rgb_dir,
        "ep_name": os.path.basename(ep_dir)
    }

def save_clip_visual(clip_data, output_dir):
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "robot"), exist_ok=True)

    # 1. Save RGB Sequence
    for i, fname in enumerate(clip_data["seq_rgb_files"]):
        src = os.path.join(clip_data["base_rgb_dir"], fname)
        dst = os.path.join(output_dir, "rgb", f"input_{i:04d}.png")
        shutil.copy2(src, dst)

    # 2. Save Goal Image
    shutil.copy2(clip_data["goal_img_path"], os.path.join(output_dir, "rgb", "goal.png"))

    # 3. Save Robot Data (ONLY TARGETS)
    # We DO NOT save "joint_seq" (input history) here to force visual usage
    data_payload = {
        "fut_absolute": clip_data["fut_absolute"].tolist(),  # OUTPUT: Future (Absolute)
    }
    
    with open(os.path.join(output_dir, "robot", "data.json"), "w") as f:
        json.dump(data_payload, f)

def process_and_save(episodes, dest_root, split_name):
    save_dir = os.path.join(dest_root, split_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing {split_name} ({len(episodes)} episodes)...")
    
    for ep in tqdm(episodes):
        ep_data = load_episode_data(ep)
        if ep_data is None: continue

        num_frames = len(ep_data["joints"])
        max_start = num_frames - (SEQ_LEN + FUTURE_STEPS) + 1

        for start_idx in range(0, max_start, STRIDE):
            clip_name = f"{ep_data['ep_name']}_clip_{start_idx:05d}"
            clip_dir = os.path.join(save_dir, clip_name)

            seq_rgb = ep_data["rgb_files"][start_idx : start_idx + SEQ_LEN]
            
            # We don't need input joints, only future targets
            future_q = ep_data["joints"][start_idx + SEQ_LEN : start_idx + SEQ_LEN + FUTURE_STEPS]

            clip_payload = {
                "seq_rgb_files": seq_rgb,
                "base_rgb_dir": ep_data["base_rgb_dir"],
                "goal_img_path": ep_data["goal_img_path"],
                "fut_absolute": future_q  # Absolute Targets
            }
            save_clip_visual(clip_payload, clip_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    args = parser.parse_args()

    all_episodes = get_episode_dirs(args.source)
    np.random.seed(42); np.random.shuffle(all_episodes)
    
    N = len(all_episodes)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    
    process_and_save(all_episodes[:n_train], args.dest, "train")
    process_and_save(all_episodes[n_train:n_train+n_val], args.dest, "val")
    process_and_save(all_episodes[n_train+n_val:], args.dest, "test")

if __name__ == "__main__":
    main()