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
        
        # Fast directory scanning
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset path does not exist: {root_dir}")
            
        self.clips = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        # OPTIMIZATION: Removed Resize. 
        # We assume images on disk are already 224x224.
        self.transform = transforms.Compose([
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

        # Robust key checking
        if "fut_absolute" in data:
            target_actions = torch.tensor(data["fut_absolute"], dtype=torch.float32)
        elif "fut_delta" in data:
            target_actions = torch.tensor(data["fut_delta"], dtype=torch.float32)
        else:
            raise KeyError(f"Missing target data in {json_path}")
        
        # Load joint history if available, else zero
        if "joint_seq" in data:
             joints = torch.tensor(data["joint_seq"], dtype=torch.float32)
        else:
             joints = torch.zeros((8, 6), dtype=torch.float32)

        # 2. Load RGB Sequence
        rgb_dir = os.path.join(clip_path, "rgb")
        rgb_seq = []
        
        # We assume files are named input_0000.png ... input_0007.png
        # Using a fixed range is faster than listdir for every item
        for i in range(8): # Assuming SEQ_LEN=8
            img_path = os.path.join(rgb_dir, f"input_{i:04d}.png")
            
            # Using 'with' ensures file is closed immediately
            with Image.open(img_path) as img:
                # OPTIONAL SAFETY CHECK (Can comment out for max speed)
                # if img.size != (224, 224):
                #     img = img.resize((224, 224))
                rgb_seq.append(self.transform(img))
            
        rgb_tensor = torch.stack(rgb_seq) 

        # 3. Load Goal Image
        goal_path = os.path.join(rgb_dir, "goal.png")
        if os.path.exists(goal_path):
            with Image.open(goal_path) as img:
                goal_tensor = self.transform(img)
        else:
            goal_tensor = torch.zeros_like(rgb_tensor[0])

        return rgb_tensor, joints, goal_tensor, target_actions