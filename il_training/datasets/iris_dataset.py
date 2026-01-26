import os
import json
import glob
import torch
import logging
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class IRISClipDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir: Path to the split folder (e.g., ~/Desktop/final_RGB_goal/train)
        """
        self.root_dir = root_dir
        
        # 1. Fast Directory Scanning
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # Look for folders containing '_clip_'
        self.clip_dirs = sorted(glob.glob(os.path.join(root_dir, "*_clip_*")))
        
        # Fallback: If glob finds nothing, take all subdirectories
        if len(self.clip_dirs) == 0:
            self.clip_dirs = sorted([
                os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))
            ])

        if len(self.clip_dirs) == 0:
            raise FileNotFoundError(f"No clip folders found in {root_dir}. Check your path.")

        logging.info(f"Found {len(self.clip_dirs)} clips in {root_dir}")

        # 2. Optimized Transform (NO RESIZING)
        # We assume images on disk are already 224x224 from your 'resize_dataset.py' script.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, idx):
        clip_path = self.clip_dirs[idx]
        
        # --- A. Load Robot Data (JSON) ---
        json_path = os.path.join(clip_path, "robot", "data.json")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            # Return dummy data if file is corrupted (prevents training crash)
            logging.error(f"Error reading {json_path}")
            return torch.zeros(8, 3, 224, 224), torch.zeros(8, 6), torch.zeros(3, 224, 224), torch.zeros(15, 6)

        # 1. Get Future Target (Robustly check for Absolute OR Delta)
        if "fut_absolute" in data:
            target_actions = torch.tensor(data["fut_absolute"], dtype=torch.float32)
        elif "fut_delta" in data:
            target_actions = torch.tensor(data["fut_delta"], dtype=torch.float32)
        else:
            # Fallback if key missing
            target_actions = torch.zeros((15, 6), dtype=torch.float32)

        # 2. Get Joint History
        if "joint_seq" in data:
            joint_seq = torch.tensor(data["joint_seq"], dtype=torch.float32)
        else:
            joint_seq = torch.zeros((8, 6), dtype=torch.float32) # Default SEQ_LEN=8

        # --- B. Load RGB Sequence ---
        rgb_tensors = []
        # We assume standard sequence length of 8 input frames
        for i in range(8):
            img_path = os.path.join(clip_path, "rgb", f"input_{i:04d}.png")
            
            # Use 'with' to ensure file handle is closed quickly
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    rgb_tensors.append(self.transform(img))
            else:
                # Padding if frame missing
                rgb_tensors.append(torch.zeros(3, 224, 224))
            
        rgb_seq = torch.stack(rgb_tensors)

        # --- C. Load Goal Image ---
        goal_path = os.path.join(clip_path, "rgb", "goal.png")
        if os.path.exists(goal_path):
            with Image.open(goal_path) as img:
                goal_tensor = self.transform(img)
        else:
            # If no goal, use the last frame of input
            goal_tensor = rgb_seq[-1]

        # Return: RGB, Joints, Goal, Targets
        return rgb_seq, joint_seq, goal_tensor, target_actions