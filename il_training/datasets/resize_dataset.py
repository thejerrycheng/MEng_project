import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import multiprocessing

# Configuration
TARGET_SIZE = (224, 224)
EXTENSIONS = ['*.png', '*.jpg', '*.jpeg']

def resize_file(file_path):
    """
    Worker function to resize a single image.
    """
    try:
        with Image.open(file_path) as img:
            # Skip if already resized (saves time on re-runs)
            if img.size == TARGET_SIZE:
                return
            
            # High-quality resize
            img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Save over original (optimize=True reduces size further without quality loss)
            img_resized.save(file_path, optimize=True, quality=95)
    except Exception as e:
        print(f"Error resizing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resize dataset images in-place to 224x224.")
    parser.add_argument("--root_dir", type=str, required=True, 
                        help="Path to the dataset root folder (e.g. ~/Desktop/data/final_visual_data)")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of CPU threads to use")
    
    args = parser.parse_args()
    
    # 1. Gather all image files recursively
    print(f"Scanning {args.root_dir} for images...")
    files = []
    for ext in EXTENSIONS:
        # We look inside **/rgb/*.png to catch inputs and goal.png
        files.extend(glob.glob(os.path.join(args.root_dir, "**", "rgb", ext), recursive=True))
    
    print(f"Found {len(files)} images. Resizing to {TARGET_SIZE}...")
    
    # 2. Resize in parallel (Using Multiprocessing for speed)
    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # list(tqdm(...)) forces the iterator to run and shows progress
            list(tqdm(pool.imap_unordered(resize_file, files), total=len(files)))
    else:
        # Single threaded fallback
        for f in tqdm(files):
            resize_file(f)
            
    print("Done! Dataset is now optimized.")

if __name__ == "__main__":
    main()