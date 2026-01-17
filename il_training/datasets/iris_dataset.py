import os
import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torchvision import transforms

# Standard ImageNet normalization for ResNet backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class EpisodeWindowDataset(Dataset):
    def __init__(self, data_path, seq_len, future_steps):
        """
        Args:
            data_path: Path to the .pkl file containing the list of episodes
            seq_len: History length
            future_steps: Prediction horizon
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        with open(data_path, "rb") as f:
            self.episodes = pickle.load(f)
            
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.samples = []
        
        # Pre-compute valid windows
        for ei, ep in enumerate(self.episodes):
            T = ep["joints"].shape[0]
            # Ensure we have enough frames for history + future
            max_start = T - (seq_len + future_steps) + 1
            for s in range(max_start):
                self.samples.append((ei, s))

        # Transforms: Resize -> Tensor -> Normalize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)), # Matches your model input size
            transforms.ToTensor(),         # Scales to [0,1]
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ei, s = self.samples[idx]
        ep = self.episodes[ei]

        rgb_seq = []
        joint_seq = []

        # 1. Load History (Sequence)
        for i in range(self.seq_len):
            img_path = os.path.join(ep["rgb_dir"], ep["rgb_files"][s + i])
            
            if not os.path.exists(img_path):
                 raise FileNotFoundError(f"Image missing: {img_path}")

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms (includes normalization)
            img_tensor = self.transform(img)
            rgb_seq.append(img_tensor)
            
            joint_seq.append(ep["joints"][s + i])

        rgb_seq = torch.stack(rgb_seq)
        joint_seq = torch.tensor(np.array(joint_seq), dtype=torch.float32)

        # 2. Compute Future Deltas (Ground Truth for Training)
        # Current position at end of history
        q_curr = ep["joints"][s + self.seq_len - 1]
        
        # Future positions
        q_future = ep["joints"][s + self.seq_len : s + self.seq_len + self.future_steps]
        
        # We predict Delta q
        fut_delta = torch.tensor(q_future - q_curr[None, :], dtype=torch.float32)

        # 3. Load Goal Image (Last image of the trajectory)
        # We assume the last file in 'rgb_files' represents the goal state
        goal_img_path = os.path.join(ep["rgb_dir"], ep["rgb_files"][-1])
        
        if not os.path.exists(goal_img_path):
             raise FileNotFoundError(f"Goal Image missing: {goal_img_path}")

        goal_img_raw = cv2.imread(goal_img_path)
        goal_img_raw = cv2.cvtColor(goal_img_raw, cv2.COLOR_BGR2RGB)
        
        # Apply SAME transforms as the input sequence
        goal_image = self.transform(goal_img_raw)

        # Return: 
        # 1. Sequence Vision (Input)
        # 2. Sequence Joints (Input)
        # 3. Goal Image (Input Condition)
        # 4. Future Action Delta (Ground Truth Target)
        return rgb_seq, joint_seq, goal_image, fut_delta