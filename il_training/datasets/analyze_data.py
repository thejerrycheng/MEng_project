import os
import argparse
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def analyze_split(root_dir, split_name):
    split_path = os.path.join(root_dir, split_name)
    if not os.path.exists(split_path):
        return None

    clip_dirs = sorted(glob.glob(os.path.join(split_path, "*_clip_*")))
    
    stats = {
        "count": len(clip_dirs),
        "unique_episodes": set(),
        "tags": defaultdict(int),
        "joint_sums": np.zeros(6),
        "joint_sq_sums": np.zeros(6),
        "total_frames": 0
    }

    print(f"Analyzing {split_name} ({len(clip_dirs)} clips)...")

    for clip in tqdm(clip_dirs):
        folder_name = os.path.basename(clip)
        
        # 1. Identify Source Episode
        # Expected format: {prefix}_episode_{number}_clip_{index}
        # We split by '_clip_' to get the base episode name
        base_name = folder_name.split("_clip_")[0]
        stats["unique_episodes"].add(base_name)

        # 2. Identify Tags/Types (Naive keyword search)
        if "obstacle" in folder_name.lower():
            stats["tags"]["Obstacle"] += 1
        else:
            stats["tags"]["Free Space"] += 1

        # 3. Load Data for Joint Stats
        json_path = os.path.join(clip, "robot", "data.json")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # data['joint_seq'] is (8, 6)
            joints = np.array(data['joint_seq'])
            
            # We only count the *current* frame (last of input) to avoid double counting 
            # sliding windows, OR we average everything. 
            # For Mean/Std calculation of the dataset, usually sampling every clip is fine.
            stats["joint_sums"] += np.sum(joints, axis=0)
            stats["joint_sq_sums"] += np.sum(joints ** 2, axis=0)
            stats["total_frames"] += len(joints)
            
        except Exception:
            continue

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Path to processed data (e.g., /media/jerry/SSD/final_data_mixed)")
    args = parser.parse_args()

    print(f"Scanning Dataset: {args.data_root}\n")

    splits = ["train", "val", "test"]
    global_stats = {
        "total_clips": 0,
        "total_episodes": 0,
        "joint_all": [],
        "split_counts": {}
    }

    # Aggregate Data
    all_joint_sums = np.zeros(6)
    all_joint_sq_sums = np.zeros(6)
    total_frame_count = 0

    for split in splits:
        res = analyze_split(args.data_root, split)
        if res:
            num = res["count"]
            eps = len(res["unique_episodes"])
            
            global_stats["split_counts"][split] = num
            global_stats["total_clips"] += num
            global_stats["total_episodes"] += eps
            
            all_joint_sums += res["joint_sums"]
            all_joint_sq_sums += res["joint_sq_sums"]
            total_frame_count += res["total_frames"]

            print(f"  [{split.upper()}] Clips: {num} | Source Episodes: {eps}")
            for tag, count in res["tags"].items():
                print(f"      - {tag}: {count}")

    # Calculate Global Mean/Std for Joints
    # Total samples = total_frame_count (since we summed over all 8 frames of every clip)
    joint_means = all_joint_sums / total_frame_count
    joint_stds = np.sqrt((all_joint_sq_sums / total_frame_count) - (joint_means ** 2))

    print("\n" + "="*40)
    print("DATASET STATISTICS SUMMARY")
    print("="*40)
    print(f"Total Samples (Clips):   {global_stats['total_clips']}")
    print(f"Total Unique Episodes:   {global_stats['total_episodes']}")
    print(f"Avg Clips per Episode:   {global_stats['total_clips'] / max(1, global_stats['total_episodes']):.1f}")
    
    print("\n[Joint Statistics (radians)]")
    joints = ["J1 (Base)", "J2 (Shoulder)", "J3 (Elbow)", "J4 (Wrist 1)", "J5 (Wrist 2)", "J6 (Wrist 3)"]
    for i, name in enumerate(joints):
        print(f"  {name:<15}: Mean={joint_means[i]:.4f}, Std={joint_stds[i]:.4f}")

    # --- LATEX OUTPUT ---
    print("\n" + "="*40)
    print("LATEX TABLE SNIPPET")
    print("="*40)
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Split} & \textbf{Episodes} & \textbf{Clips (Samples)} & \textbf{Ratio} \\")
    print(r"\midrule")
    
    # Re-calculate explicitly for table consistency
    t_c = global_stats['split_counts'].get('train', 0)
    v_c = global_stats['split_counts'].get('val', 0)
    te_c = global_stats['split_counts'].get('test', 0)
    total = t_c + v_c + te_c
    
    print(f"Train & - & {t_c} & {t_c/total*100:.1f}\\% \\\\")
    print(f"Validation & - & {v_c} & {v_c/total*100:.1f}\\% \\\\")
    print(f"Test & - & {te_c} & {te_c/total*100:.1f}\\% \\\\")
    print(r"\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{global_stats['total_episodes']}}} & \\textbf{{{total}}} & 100\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Dataset composition details.}")
    print(r"\label{tab:dataset_stats}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()