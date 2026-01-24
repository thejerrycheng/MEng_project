import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class IRISClipDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clips = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = os.path.join(self.root_dir, self.clips[idx])
        
        # 1. Load Robot Data (JSON)
        json_path = os.path.join(clip_path, "robot", "data.json")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # --- FIX STARTS HERE ---
        # Robustly check for 'fut_absolute' (new) or 'fut_delta' (old)
        if "fut_absolute" in data:
            target_actions = torch.tensor(data["fut_absolute"], dtype=torch.float32)
        elif "fut_delta" in data:
            target_actions = torch.tensor(data["fut_delta"], dtype=torch.float32)
        else:
            raise KeyError(f"Neither 'fut_absolute' nor 'fut_delta' found in {json_path}")
        
        # Safe check for 'joint_seq' (RGB-only datasets might skip this, but usually we save it for safety)
        if "joint_seq" in data:
             joints = torch.tensor(data["joint_seq"], dtype=torch.float32)
        else:
             # If missing (e.g. pure visual dataset), return dummy zeros
             # Shape: (Seq_Len, 6) -> Assuming Seq_Len=8 from your config
             joints = torch.zeros((8, 6), dtype=torch.float32)
        # --- FIX ENDS HERE ---

        # 2. Load Images
        rgb_dir = os.path.join(clip_path, "rgb")
        
        # Input Sequence
        rgb_seq = []
        # We assume files are named input_0000.png to input_0007.png
        # The number of input files should match the sequence length used during creation
        input_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith("input_")])
        
        for img_file in input_files:
            img_path = os.path.join(rgb_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            rgb_seq.append(self.transform(img))
            
        rgb_tensor = torch.stack(rgb_seq) # (S, C, H, W)

        # 3. Load Goal Image
        goal_path = os.path.join(rgb_dir, "goal.png")
        if os.path.exists(goal_path):
            goal_img = Image.open(goal_path).convert("RGB")
            goal_tensor = self.transform(goal_img)
        else:
            # If no goal image exists (e.g. RGB-only tasks), return a blank tensor
            goal_tensor = torch.zeros_like(rgb_tensor[0])

        # Return: (RGB_Seq, Joint_History, Goal_Image, Target_Actions)
        return rgb_tensor, joints, goal_tensor, target_actions