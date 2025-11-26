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
rgb_data_folder = "../bag_reader/scripts/rgb_data/data_20250209_203309_rgb"
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

# Interpolate joint state data to match images
rgb_files = sorted(os.listdir(rgb_data_folder))
depth_files = sorted(os.listdir(depth_data_folder))
assert len(rgb_files) == len(depth_files), "RGB and depth images must have the same number of frames."

# Create interpolation functions for each joint
interp_funcs = [interp1d(joint_timestamps, joint_positions[:, i], kind='linear', fill_value='extrapolate') for i in range(joint_positions.shape[1])]
print("The interpolation function is made... interpolation started")

# Generate interpolated joint positions for images
timestamps = np.linspace(joint_timestamps[0], joint_timestamps[-1], len(rgb_files))
interpolated_joints = np.stack([interp_funcs[i](timestamps) for i in range(joint_positions.shape[1])], axis=1)

# Preprocessing pipeline for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class ImitationDataset(Dataset):
    def __init__(self, rgb_folder, depth_folder, interpolated_joints, seq_len=SEQ_LEN):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.seq_len = seq_len
        self.joint_positions = interpolated_joints
        self.image_files = sorted(os.listdir(self.rgb_folder))

    def __len__(self):
        return len(self.image_files) - self.seq_len

    def __getitem__(self, idx):
        rgb_images, depth_images, joint_angles_seq = [], [], []
        for i in range(self.seq_len):
            rgb_path = os.path.join(self.rgb_folder, self.image_files[idx + i])
            depth_path = os.path.join(self.depth_folder, self.image_files[idx + i])
            
            # Load RGB image
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None:
                raise FileNotFoundError(f"Failed to load RGB image: {rgb_path}")
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = transform(rgb_img)
            
            # Load depth image
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise FileNotFoundError(f"Failed to load depth image: {depth_path}")
            depth_img = depth_img.astype(np.float32) / 65535.0  # Normalize 16-bit depth
            depth_img = (depth_img * 255).astype(np.uint8)  # Convert to 8-bit for compatibility
            depth_img = np.expand_dims(depth_img, axis=-1)  # Ensure it's a single channel
            depth_img = transform(depth_img)
            
            rgb_images.append(rgb_img)
            depth_images.append(depth_img)
            joint_angles_seq.append(torch.tensor(self.joint_positions[idx + i], dtype=torch.float32))
        
        rgb_images = torch.stack(rgb_images)  # Shape: (seq_len, C, H, W)
        depth_images = torch.stack(depth_images)  # Shape: (seq_len, C, H, W)
        joint_angles_seq = torch.stack(joint_angles_seq)  # Shape: (seq_len, 6)
        next_joint_angles = torch.tensor(self.joint_positions[idx + self.seq_len], dtype=torch.float32)  # Shape: (6,)
        return rgb_images, depth_images, joint_angles_seq, next_joint_angles

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ImitationDataset(rgb_data_folder, depth_data_folder, interpolated_joints)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Debugging: Check if images are loaded properly
for i in range(3):
    sample = dataset[i]
    print(f"Sample {i}: RGB shape {sample[0].shape}, Depth shape {sample[1].shape}, Joint angles shape {sample[2].shape}")

# Model, optimizer, and loss function would follow here

model = ImitationPolicyRGBD().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Training loop
best_loss = float("inf")
stopping_counter = 0
losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for rgb_imgs, depth_imgs, joint_angles_seq, next_joint_angles in dataloader:
        rgb_imgs, depth_imgs, joint_angles_seq, next_joint_angles = rgb_imgs.to(device), depth_imgs.to(device), joint_angles_seq.to(device), next_joint_angles.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_imgs, depth_imgs, joint_angles_seq)
        loss = criterion(outputs, next_joint_angles)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(models_dir, "rgbd_best_model.pth"))
    else:
        stopping_counter += 1
    if stopping_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

torch.save(model.state_dict(), os.path.join(models_dir, "rgbd_final_model.pth"))
plt.figure()
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.grid()
plt.savefig(os.path.join(plots_dir, "loss_plot_rgbd.png"))
plt.show()
