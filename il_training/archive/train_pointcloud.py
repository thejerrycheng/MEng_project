import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import ast
from pointnet_model import PointNet

# Paths to data
pointcloud_data_folder = "../bag_reader/scripts/pointcloud_data"
joint_data_file = "../bag_reader/scripts/joint_states.csv"

# Hyperparameters
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 50
SEQ_LEN = 5  # Number of past frames to consider for temporal dependency

# Load joint state data
joint_data = pd.read_csv(joint_data_file)

# Convert joint positions from string to numerical values
joint_data.iloc[:, 1] = joint_data.iloc[:, 1].apply(lambda x: ast.literal_eval(x))
joint_positions = np.array(joint_data.iloc[:, 1].tolist())
joint_timestamps = joint_data.iloc[:, 0].values

# Interpolate joint state data to match point cloud sequences
pc_files = sorted(os.listdir(pointcloud_data_folder))

# Create interpolation functions for each joint
interp_funcs = [interp1d(joint_timestamps, joint_positions[:, i], kind='linear', fill_value='extrapolate') for i in range(joint_positions.shape[1])]

# Generate interpolated joint positions for point cloud sequences
pc_timestamps = np.linspace(joint_timestamps[0], joint_timestamps[-1], len(pc_files))
interpolated_joints = np.stack([interp_funcs[i](pc_timestamps) for i in range(joint_positions.shape[1])], axis=1)

class PointCloudDataset(Dataset):
    def __init__(self, pc_folder, interpolated_joints, seq_len=SEQ_LEN):
        self.pc_folder = pc_folder
        self.seq_len = seq_len
        self.joint_positions = interpolated_joints
        self.pc_files = sorted(os.listdir(self.pc_folder))

    def __len__(self):
        return len(self.pc_files) - self.seq_len

    def __getitem__(self, idx):
        point_clouds = []
        for i in range(self.seq_len):
            pc_path = os.path.join(self.pc_folder, self.pc_files[idx + i])
            pc = np.load(pc_path)  # Assume point cloud is stored in .npy format
            point_clouds.append(torch.tensor(pc, dtype=torch.float32))
        
        point_clouds = torch.stack(point_clouds)  # Shape: (seq_len, N, 3)
        joint_angles = torch.tensor(self.joint_positions[idx + self.seq_len], dtype=torch.float32)
        return point_clouds, joint_angles

# Define model
class ImitationPolicyPointNet(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, output_dim=6):
        super(ImitationPolicyPointNet, self).__init__()
        self.pointnet = PointNet()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        B, S, N, C = x.shape  # (Batch, Sequence, Num_Points, Channels)
        x = x.view(B * S, N, C)  # Flatten sequence into batch
        x = self.pointnet(x)  # Pass through PointNet
        x = x.view(B, S, -1)  # Reshape back to sequence format
        x, _ = self.lstm(x)  # LSTM for temporal dependencies
        x = self.fc(x[:, -1, :])  # Take last timestep output
        return x

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PointCloudDataset(pointcloud_data_folder, interpolated_joints)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ImitationPolicyPointNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Training loop
losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for pointclouds, joint_targets in dataloader:
        pointclouds, joint_targets = pointclouds.to(device), joint_targets.to(device)
        optimizer.zero_grad()
        outputs = model(pointclouds)
        loss = criterion(outputs, joint_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "il_training/imitation_policy_pointnet.pth")

# Plot training loss
plt.figure()
plt.plot(range(1, EPOCHS+1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.grid()
plt.savefig("il_training/loss_plot_pointnet.png")
plt.show()
