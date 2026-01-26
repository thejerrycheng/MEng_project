import os
import glob
from PIL import Image
from tqdm import tqdm
import argparse

def resize_images_in_place(folder, size=(224, 224)):
    # Find all png files recursively
    files = glob.glob(os.path.join(folder, "**/*.png"), recursive=True)
    print(f"Found {len(files)} images in {folder}. Resizing to {size}...")

    for f in tqdm(files):
        try:
            with Image.open(f) as img:
                # Check if already resized to avoid double work
                if img.size == size:
                    continue
                
                # Resize and Overwrite
                img_resized = img.resize(size, Image.LANCZOS)
                img_resized.save(f, optimize=True, quality=95)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, 
                        help="Path to dataset (e.g. ~/Desktop/final_RGB_goal)")
    args = parser.parse_args()
    
    resize_images_in_place(args.target_dir)