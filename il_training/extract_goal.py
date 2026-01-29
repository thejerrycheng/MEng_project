import os
import shutil

# ================= CONFIGURATION =================
SOURCE_ROOT = "/media/jerry/SSD/final_data_mixed/test"
DEST_DIR = os.path.expanduser("~/Desktop/goal_images")
# =================================================

def get_start_index(directory, prefix="goal"):
    """Finds the next available number 'N' to start naming from."""
    i = 1
    while True:
        if not os.path.exists(os.path.join(directory, f"{prefix}{i}.png")):
            return i
        i += 1

def get_trajectory_id(folder_name):
    """
    Extracts the unique trajectory ID from the folder name.
    Example: 'diverse_2_..._episode_0002_clip_00034' -> 'diverse_2_..._episode_0002'
    """
    if "_clip_" in folder_name:
        return folder_name.split("_clip_")[0]
    return folder_name  # Fallback if naming structure is different

def main():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created directory: {DEST_DIR}")

    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source path not found: {SOURCE_ROOT}")
        return

    # Sort to ensure we process clips in order (usually clip_00000 first)
    clip_folders = sorted([d for d in os.listdir(SOURCE_ROOT) 
                           if os.path.isdir(os.path.join(SOURCE_ROOT, d))])

    print(f"Scanning {len(clip_folders)} clips for unique trajectories...")
    
    processed_trajectories = set()
    current_index = get_start_index(DEST_DIR)
    count = 0

    for clip_name in clip_folders:
        traj_id = get_trajectory_id(clip_name)
        
        # 1. Check if we already have a goal for this trajectory
        if traj_id in processed_trajectories:
            continue  # Skip this clip, it's a duplicate trajectory

        # 2. If new, look for the goal image
        source_image_path = os.path.join(SOURCE_ROOT, clip_name, "rgb", "goal.png")
        
        if os.path.exists(source_image_path):
            new_filename = f"goal{current_index}.png"
            dest_path = os.path.join(DEST_DIR, new_filename)
            
            try:
                shutil.copy2(source_image_path, dest_path)
                print(f"[New Trajectory] {traj_id} -> {new_filename}")
                
                # Mark this trajectory as done so we don't copy again for clip_00001, etc.
                processed_trajectories.add(traj_id)
                
                count += 1
                current_index += 1
            except Exception as e:
                print(f"[Error] Failed to copy {source_image_path}: {e}")
        else:
            # If the first clip doesn't have it, we might want to check others, 
            # but usually they all share it. We'll skip for now to be safe.
            pass

    print("-" * 40)
    print(f"Done! Extracted {count} unique goal images.")
    print(f"Total clips scanned: {len(clip_folders)}")
    print(f"Total unique trajectories found: {len(processed_trajectories)}")

if __name__ == "__main__":
    main()