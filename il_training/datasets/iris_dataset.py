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
            transforms.Resize((128, 128)),
            transforms.ToTensor(), # Scales to [0,1]
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
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms (includes normalization)
            img_tensor = self.transform(img)
            rgb_seq.append(img_tensor)
            
            joint_seq.append(ep["joints"][s + i])

        rgb_seq = torch.stack(rgb_seq)
        joint_seq = torch.tensor(np.array(joint_seq), dtype=torch.float32)

        # 2. Compute Future Deltas (Ground Truth)
        # Current position at end of history
        q_curr = ep["joints"][s + self.seq_len - 1]
        
        # Future positions
        q_future = ep["joints"][s + self.seq_len : s + self.seq_len + self.future_steps]
        
        # We predict Delta q
        fut_delta = torch.tensor(q_future - q_curr[None, :], dtype=torch.float32)

        # 3. Goal (XYZ)
        goal_xyz = torch.tensor(ep["goal_xyz"], dtype=torch.float32)
        
        # 4. Goal (Joints - optional, for auxiliary loss)
        goal_joint = torch.tensor(ep["goal_joint"], dtype=torch.float32)

        return rgb_seq, joint_seq, goal_xyz, fut_delta, goal_joint