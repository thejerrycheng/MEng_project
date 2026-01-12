import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import cv2
from scipy.interpolate import interp1d

# ---------------------
# Argument Parser
# ---------------------
parser = argparse.ArgumentParser(description="Train an imitation learning model using RGB images + robot data")
parser.add_argument("--bag", type=str, required=True, help="Name of the rosbag to process (without extension)")
args = parser.parse_args()

# ---------------------
# Paths & Directories
# ---------------------
rgb_dir = f"../bag_reader/scripts/rgb_data/{args.bag}_rgb/"
robot_data_file = f"../bag_reader/scripts/robot_data/{args.bag}/joint_states.csv"

models_dir = "models"
plots_dir = "plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ---------------------
# Hyperparameters
# ---------------------
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 50
SEQ_LEN = 5            # Number of past frames to consider
FUTURE_STEPS = 10      # Number of future timesteps to predict
PATIENCE = 5           # Early stopping patience

# ---------------------
# Transforms
# ---------------------
rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------------
# Dataset Definition
# ---------------------
class ImitationDataset(Dataset):
    def __init__(self, rgb_dir, robot_data_file, seq_len=SEQ_LEN, future_steps=FUTURE_STEPS):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        
        # 1) Load the CSV of joint states (timestamps + positions)
        #    Assuming columns: [timestamp, joint_positions_string]
        joint_data = pd.read_csv(robot_data_file)
        
        # Convert string lists to actual Python lists
        # Example: "[1.2, 0.4, ...]" -> [1.2, 0.4, ...]
        joint_data.iloc[:, 1] = joint_data.iloc[:, 1].apply(eval)
        
        # Extract numpy arrays
        self.joint_timestamps = joint_data.iloc[:, 0].values  # shape: (N,)
        self.joint_positions = np.array(joint_data.iloc[:, 1].tolist())  # shape: (N, num_joints)
        
        # 2) Gather the sorted list of RGB image files
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.rgb_dir = rgb_dir
        
        # 3) Interpolate joint positions to match the number of frames
        #    We'll assume timestamps are from self.joint_timestamps[0] to self.joint_timestamps[-1]
        interp_funcs = [
            interp1d(self.joint_timestamps, self.joint_positions[:, i], kind='linear', fill_value='extrapolate')
            for i in range(self.joint_positions.shape[1])
        ]
        
        frame_timestamps = np.linspace(self.joint_timestamps[0], self.joint_timestamps[-1], len(self.rgb_files))
        self.interpolated_joints = np.stack([f(frame_timestamps) for f in interp_funcs], axis=1)
        # shape: (num_frames, num_joints)

    def __len__(self):
        # We can only sample where we have SEQ_LEN past frames + FUTURE_STEPS future frames
        return len(self.rgb_files) - self.seq_len - self.future_steps

    def __getitem__(self, idx):
        """
        Returns:
            rgb_seq: Tensor of shape (SEQ_LEN, 3, H, W)
            joint_seq: Tensor of shape (SEQ_LEN, num_joints)
            future_joints: Tensor of shape (FUTURE_STEPS, num_joints)
        """
        # Collect the past SEQ_LEN frames
        rgb_seq = []
        joint_seq = []
        
        for i in range(self.seq_len):
            img_file = self.rgb_files[idx + i]
            img_path = os.path.join(self.rgb_dir, img_file)
            
            # Read and transform
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = rgb_transform(img)
            rgb_seq.append(img)
            
            # Corresponding joint
            joint_seq.append(self.interpolated_joints[idx + i])
        
        # Convert to Tensors
        rgb_seq = torch.stack(rgb_seq)  # (SEQ_LEN, 3, H, W)
        joint_seq = torch.tensor(np.array(joint_seq), dtype=torch.float32)  # (SEQ_LEN, num_joints)
        
        # Collect the future FUTURE_STEPS joint positions
        future_joints = self.interpolated_joints[idx + self.seq_len : idx + self.seq_len + self.future_steps]
        future_joints = torch.tensor(future_joints, dtype=torch.float32)  # (FUTURE_STEPS, num_joints)
        
        return rgb_seq, joint_seq, future_joints

# ---------------------
# Build Dataset
# ---------------------
dataset = ImitationDataset(rgb_dir, robot_data_file)

# Split into train/val/test (80/10/10)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------
# Model Definition
# ---------------------

class VisionJointPlanner(nn.Module):
    def __init__(self, future_steps=FUTURE_STEPS):
        super().__init__()

        # Feature Extractor - ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # Output of ResNet18's penultimate layer

        # Transformer input dimension (Must be divisible by `num_heads`)
        self.transformer_dim = self.feature_dim + 6  # (512 + 6 = 518)
        
        # Fix: Make `num_heads` a divisor of `self.transformer_dim`
        num_heads = 2  # 518 is divisible by 2, so we use `num_heads=2`

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            batch_first=True  # ✅ Fix: Use batch_first=True for better performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)


        # Fully Connected layers
        self.fc1 = nn.Linear(self.transformer_dim, 256)
        self.fc2 = nn.Linear(256, future_steps * 6)  # Predict 10 timesteps * 6 joints = 60 outputs

    def forward(self, rgb_seq, joint_seq):
        B, S, C, H, W = rgb_seq.shape
        
        # Extract image features from ResNet18
        rgb_seq = rgb_seq.view(B * S, C, H, W)
        vision_features = self.feature_extractor(rgb_seq).view(B, S, self.feature_dim)

        # Concatenate visual + joint features
        fused_input = torch.cat([vision_features, joint_seq], dim=2)  # (B, S, 518)

        # Transformer expects (S, B, E)
        fused_input = fused_input.permute(1, 0, 2)  # (S, B, 518)

        transformed = self.transformer_encoder(fused_input)

        # Take last timestep for prediction
        last_timestep = transformed[-1]  # (B, 518)
        x = torch.relu(self.fc1(last_timestep))
        x = self.fc2(x)
        return x.view(B, FUTURE_STEPS, 6)

# ---------------------
# Training Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionJointPlanner().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float("inf")
stopping_counter = 0
train_losses, val_losses = [], []

# ---------------------
# Training Loop
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    
    for rgb_imgs, joint_seq, future_joints in train_loader:
        rgb_imgs = rgb_imgs.to(device)
        joint_seq = joint_seq.to(device)
        future_joints = future_joints.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb_imgs, joint_seq)
        loss = criterion(outputs, future_joints)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
    
    train_loss = running_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for rgb_imgs, joint_seq, future_joints in val_loader:
            rgb_imgs = rgb_imgs.to(device)
            joint_seq = joint_seq.to(device)
            future_joints = future_joints.to(device)
            
            outputs = model(rgb_imgs, joint_seq)
            loss = criterion(outputs, future_joints)
            running_val_loss += loss.item()
    
    val_loss = running_val_loss / len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(models_dir, f"{args.bag}_best_model.pth"))
    else:
        stopping_counter += 1
        if stopping_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ---------------------
# Plot Training & Val Loss
# ---------------------
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, f"loss_plot_{args.bag}.png"))
plt.show()

# ---------------------
# (Optional) Test Loop
# ---------------------
# You can evaluate on test_loader if desired:
# model.load_state_dict(torch.load(os.path.join(models_dir, f"{args.bag}_best_model.pth")))
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for rgb_imgs, joint_seq, future_joints in test_loader:
#         rgb_imgs = rgb_imgs.to(device)
#         joint_seq = joint_seq.to(device)
#         future_joints = future_joints.to(device)
#         outputs = model(rgb_imgs, joint_seq)
#         loss = criterion(outputs, future_joints)
#         test_loss += loss.item()
# test_loss /= len(test_loader)
# print(f"Test Loss: {test_loss:.4f}")
