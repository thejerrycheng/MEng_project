import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use 'Agg' for a non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import logging
import cv2

# ---------------------
# Logging Setup
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# ---------------------
# Argument Parser
# ---------------------
parser = argparse.ArgumentParser(
    description="Test the imitation learning model and visualize results")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the trained model")
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to the folder containing test samples (npz files)")
args = parser.parse_args()

# ---------------------
# Processed Dataset Definition (sequential order is preserved)
# ---------------------
class ProcessedDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # Use sorted() to ensure temporal ordering
        self.files = sorted([os.path.join(dataset_dir, f)
                             for f in os.listdir(dataset_dir) if f.endswith('.npz')])
        if not self.files:
            raise RuntimeError(f"No .npz files found in {dataset_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        # Each sample is saved with keys: rgb_seq, joint_seq, future_joints
        rgb_seq = torch.from_numpy(sample['rgb_seq']).float()       # (seq_len, C, H, W)
        joint_seq = torch.from_numpy(sample['joint_seq']).float()   # (seq_len, joints)
        future_joints = torch.from_numpy(sample['future_joints']).float()  # (future_steps, joints)
        return rgb_seq, joint_seq, future_joints

# ---------------------
# VisionJointPlanner Model Definition
# ---------------------
FUTURE_STEPS = 5  # Ensure this matches your training settings

class VisionJointPlanner(nn.Module):
    def __init__(self, future_steps=FUTURE_STEPS):
        super(VisionJointPlanner, self).__init__()
        # Use a pretrained ResNet18 model
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        # Fuse vision features with joint data (assumed 6 joints)
        self.conv1 = nn.Conv1d(in_channels=self.feature_dim + 6,
                               out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256,
                               out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, future_steps * 6)

    def forward(self, rgb_seq, joint_seq):
        # rgb_seq: (B, S, C, H, W)
        B, S, C, H, W = rgb_seq.shape
        rgb_seq = rgb_seq.view(B * S, C, H, W)
        vision_features = self.feature_extractor(rgb_seq).view(B, S, self.feature_dim)
        fused_input = torch.cat([vision_features, joint_seq], dim=2).permute(0, 2, 1)
        x = torch.relu(self.conv1(fused_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.mean(dim=2)  # Global average pooling over S
        x = self.fc(x)
        return x.view(B, FUTURE_STEPS, 6)

# ---------------------
# Load Test Dataset (sequential order preserved)
# ---------------------
logging.info("Loading test dataset from processed files (sequential order)...")
dataset = ProcessedDataset(args.dataset_dir)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # No shuffling!
logging.info(f"Test dataset loaded with {len(dataset)} samples.")

# ---------------------
# Load Trained Model
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionJointPlanner().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
logging.info(f"Model loaded from {args.model_path}.")

# ---------------------
# Forward Kinematics Functions
# ---------------------
def dh_transform(theta, d, alpha, a):
    """Compute the transformation matrix using DH parameters."""
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def forward_kinematics(joint_angles):
    """
    Compute the forward kinematics of the xArm Lite 6.
    
    Parameters:
        joint_angles: (N, 6) numpy array of joint angles (in degrees)
        
    Returns:
        (N, 3) numpy array of end-effector (x, y, z) positions.
    """
    dh_params = [
        [0,    243.3,   0,   0],
        [-90,   0,    -90,   0],
        [-90,   0,    180, 200],
        [0,   227.6,   90,  87],
        [0,     0,     90,   0],
        [0,    61.5,  -90,   0]
    ]
    
    if joint_angles.shape[1] != len(dh_params):
        raise ValueError(f"Expected {len(dh_params)} joint angles, but got {joint_angles.shape[1]}.")
    
    positions = []
    for angles in joint_angles:
        T = np.eye(4)
        for i in range(len(dh_params)):
            theta_offset, d, alpha, a = dh_params[i]
            theta = angles[i] + theta_offset
            T = np.dot(T, dh_transform(theta, d, alpha, a))
        positions.append(T[:3, 3])
    return np.array(positions)

# ---------------------
# Run Inference and Collect Results (sequential)
# ---------------------
logging.info("Running inference on test dataset (sequential order)...")
predicted_trajectories = []
ground_truth_trajectories = []

with torch.no_grad():
    for rgb_imgs, joint_seq, future_joints in test_loader:
        rgb_imgs = rgb_imgs.to(device)
        joint_seq = joint_seq.to(device)
        # Predict future joint states: (1, FUTURE_STEPS, 6)
        predicted_future_joints = model(rgb_imgs, joint_seq)
        predicted_future_joints = predicted_future_joints.cpu().numpy().squeeze(0)  # (FUTURE_STEPS, 6)
        future_joints_np = future_joints.cpu().numpy().squeeze(0)  # (FUTURE_STEPS, 6)
        
        pred_traj = forward_kinematics(predicted_future_joints)
        gt_traj = forward_kinematics(future_joints_np)
        
        predicted_trajectories.append(pred_traj)
        ground_truth_trajectories.append(gt_traj)

# Concatenate trajectories along the time axis
predicted_trajectories = np.concatenate(predicted_trajectories, axis=0)
ground_truth_trajectories = np.concatenate(ground_truth_trajectories, axis=0)
logging.info("Inference complete. Visualizing results...")

# ---------------------
# Visualization
# ---------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(ground_truth_trajectories[:, 0],
        ground_truth_trajectories[:, 1],
        ground_truth_trajectories[:, 2],
        label="Ground Truth", color="blue")
ax.plot(predicted_trajectories[:, 0],
        predicted_trajectories[:, 1],
        predicted_trajectories[:, 2],
        label="Predicted Trajectory", linestyle="dashed", color="red")

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Robot End-Effector Trajectory Comparison")
ax.legend()
plt.show()
