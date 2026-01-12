import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
from scipy.interpolate import interp1d
import ast

# Paths to data
depth_data_folder = "../bag_reader/scripts/depth_data/data_20250209_203309"
joint_data_file = "../bag_reader/scripts/robot_data/data_20250209_203309/joint_states.csv"
models_dir = "models"
plots_dir = "plots"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 50
SEQ_LEN = 5  # Number of past frames to consider for temporal dependency
PATIENCE = 5  # Early stopping patience

# Load joint state data
joint_data = pd.read_csv(joint_data_file)

# Convert joint positions from string to numerical values
joint_data.iloc[:, 1] = joint_data.iloc[:, 1].apply(lambda x: ast.literal_eval(x))
joint_positions = np.array(joint_data.iloc[:, 1].tolist())
joint_timestamps = joint_data.iloc[:, 0].values

# Interpolate joint state data to match depth images
depth_files = sorted(os.listdir(depth_data_folder))

# Create interpolation functions for each joint
interp_funcs = [interp1d(joint_timestamps, joint_positions[:, i], kind='linear', fill_value='extrapolate') for i in range(joint_positions.shape[1])]
print("The interpolation function is made... interpolation started")

# Generate interpolated joint positions for depth images
depth_timestamps = np.linspace(joint_timestamps[0], joint_timestamps[-1], len(depth_files))
interpolated_joints = np.stack([interp_funcs[i](depth_timestamps) for i in range(joint_positions.shape[1])], axis=1)

# Preprocessing pipeline for depth images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class ImitationDataset(Dataset):
    def __init__(self, depth_folder, interpolated_joints, seq_len=SEQ_LEN):
        self.depth_folder = depth_folder
        self.seq_len = seq_len
        self.joint_positions = interpolated_joints
        self.image_files = sorted(os.listdir(self.depth_folder))

    def __len__(self):
        return len(self.image_files) - self.seq_len

    def __getitem__(self, idx):
        depth_images = []
        joint_angles_seq = []
        for i in range(self.seq_len):
            img_path = os.path.join(self.depth_folder, self.image_files[idx + i])
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = transform(img)
            depth_images.append(img)
            joint_angles_seq.append(torch.tensor(self.joint_positions[idx + i], dtype=torch.float32))
        
        depth_images = torch.stack(depth_images)  # Shape: (seq_len, C, H, W)
        joint_angles_seq = torch.stack(joint_angles_seq)  # Shape: (seq_len, 6)
        next_joint_angles = torch.tensor(self.joint_positions[idx + self.seq_len], dtype=torch.float32)  # Shape: (6,)
        return depth_images, joint_angles_seq, next_joint_angles

# Define model

# import torch
# import torch.nn as nn
# import torchvision.models as models

class CNNImitationPolicy(nn.Module):
    def __init__(self, seq_len=5, output_dim=6):
        super(CNNImitationPolicy, self).__init__()

        # 3D CNN for processing temporal depth image sequence
        self.conv3d = nn.Sequential(
            # Input shape: (Batch, Channels=1, Seq_len, H=128, W=128)
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),  # (B, 32, S=5, 64, 64)
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # (B, 64, S=5, 32, 32)
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)), # (B, 128, S=5, 16, 16)
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)), # (B, 256, S=5, 8, 8)
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Global Average Pooling → Output (B, 256, 1, 1, 1)
        )

        # Fully connected layers for depth features
        self.fc_depth = nn.Sequential(
            nn.Linear(256, 256),  # Instead of flattening, directly use 256 features
            nn.ReLU()
        )

        # 1D CNN for processing temporal joint angles
        self.conv1d_joint = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (B, 16, S)
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (B, 32, S)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Pooling → (B, 32, 1)
        )

        # Fully connected layers for joint angles
        self.fc_joint = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Final fully connected layer to predict next joint angles
        self.fc_output = nn.Linear(256 + 64, output_dim)

    def forward(self, depth, joint_angles):
        B, S, C, H, W = depth.shape  # (Batch, Sequence, Channels, Height, Width)
        
        # 3D CNN expects (Batch, Channels, Sequence, H, W), so we swap axes
        depth = depth.view(B, 1, S, H, W)  # Convert to (B, 1, S, H, W)
        depth_features = self.conv3d(depth)  # Extract features: (B, 256, 1, 1, 1)
        depth_features = depth_features.view(B, -1)  # (B, 256)
        depth_features = self.fc_depth(depth_features)  # (B, 256)

        # Joint angles processing (B, S, 6) → (B, 6, S) for Conv1D
        joint_angles = joint_angles.permute(0, 2, 1)  # Swap to (B, 6, S)
        joint_features = self.conv1d_joint(joint_angles)  # Extracted features (B, 32, 1)
        joint_features = joint_features.view(B, -1)  # Flatten to (B, 32)
        joint_features = self.fc_joint(joint_features)  # (B, 64)

        # Concatenate CNN features and joint features
        x = torch.cat((depth_features, joint_features), dim=1)  # (B, 256+64)

        # Predict next joint angles
        x = self.fc_output(x)  # (B, 6)

        return x


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ImitationDataset(depth_data_folder, interpolated_joints)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNImitationPolicy().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Early stopping
best_loss = float("inf")
stopping_counter = 0

# Training loop
losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for depth_imgs, joint_angles_seq, next_joint_angles in dataloader:
        depth_imgs, joint_angles_seq, next_joint_angles = depth_imgs.to(device), joint_angles_seq.to(device), next_joint_angles.to(device)
        optimizer.zero_grad()
        outputs = model(depth_imgs, joint_angles_seq)
        loss = criterion(outputs, next_joint_angles)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(models_dir, "depth_best_model_cnn.pth"))
    else:
        stopping_counter += 1
    
    # # Early stopping condition
    # if stopping_counter >= PATIENCE:
    #     print("Early stopping triggered.")
    #     break

# Save final model
torch.save(model.state_dict(), os.path.join(models_dir, "depth_final_model_cnn.pth"))

# Plot training loss
plt.figure()
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.grid()
plt.savefig(os.path.join(plots_dir, "loss_plot_depth.png"))
plt.show()
