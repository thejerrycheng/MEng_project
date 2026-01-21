import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging

class IRISClipDataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224)):
        """
        Args:
            root_dir: Path to the split folder (e.g., /media/jerry/SSD/final_data/train)
            image_size: Tuple for resizing images (224, 224) for ResNet.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        
        logging.info(f"Scanning for clips in: {root_dir}")
        
        # 1. Try finding folders with specific pattern
        self.clip_dirs = sorted(glob.glob(os.path.join(root_dir, "*_clip_*")))
        
        # 2. If empty, grab ANY subdirectories (fallback)
        if len(self.clip_dirs) == 0:
            logging.warning(f"No folders with '_clip_' found in {root_dir}. Looking for ANY subfolders...")
            self.clip_dirs = [
                os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))
            ]
            self.clip_dirs = sorted(self.clip_dirs)

        # 3. Check again
        if len(self.clip_dirs) == 0:
            # Helpful error message listing contents of directory
            try:
                contents = os.listdir(root_dir)
                logging.error(f"Directory contents of {root_dir}: {contents[:5]} ...")
            except Exception:
                logging.error(f"Could not read directory {root_dir}")
                
            raise FileNotFoundError(
                f"No subfolders found in {root_dir}. \n"
                f"Make sure you point to the parent folder containing the clips directly.\n"
                f"Example: If your path is .../train/episode_0_clip_0, your root_dir must be .../train"
            )

        logging.info(f"Found {len(self.clip_dirs)} valid clips.")

        # Standard ImageNet normalization for ResNet
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, idx):
        clip_path = self.clip_dirs[idx]
        
        # 1. Load Robot Data (JSON)
        json_path = os.path.join(clip_path, "robot", "data.json")
        if not os.path.exists(json_path):
             # Skip broken clips or raise error
             raise FileNotFoundError(f"Missing data.json in {clip_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        joint_seq = torch.tensor(data["joint_seq"], dtype=torch.float32)
        fut_delta = torch.tensor(data["fut_delta"], dtype=torch.float32)

        # 2. Load RGB Sequence (Inputs)
        rgb_tensors = []
        # Robust loading: check how many inputs we actually have if < 8
        num_frames = len(joint_seq) 
        
        for i in range(num_frames):
            img_path = os.path.join(clip_path, "rgb", f"input_{i:04d}.png")
            if not os.path.exists(img_path):
                 raise FileNotFoundError(f"Missing image {img_path}")
            
            img = Image.open(img_path).convert("RGB")
            rgb_tensors.append(self.transform(img))
            
        rgb_seq = torch.stack(rgb_tensors)

        # 3. Load Goal Image
        goal_path = os.path.join(clip_path, "rgb", "goal.png")
        if not os.path.exists(goal_path):
             # Fallback: use last frame
             goal_path = os.path.join(clip_path, "rgb", f"input_{num_frames-1:04d}.png")
        
        goal_img = Image.open(goal_path).convert("RGB")
        goal_tensor = self.transform(goal_img)

        return rgb_seq, joint_seq, goal_tensor, fut_delta