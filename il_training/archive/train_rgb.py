import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or try 'Agg' for a non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import cv2
from scipy.interpolate import interp1d
import logging

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
parser = argparse.ArgumentParser(description="Train an imitation learning model using CNN with RGB images + robot data")
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
SEQ_LEN = 5           # Number of past frames to consider
FUTURE_STEPS = 5      # Number of future timesteps to predict
PATIENCE = 5          # Early stopping patience

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
        
        logging.info(f"Loading robot data from {robot_data_file}")
        joint_data = pd.read_csv(robot_data_file)
        
        logging.info("Processing joint state data...")
        joint_data.iloc[:, 1] = joint_data.iloc[:, 1].apply(eval)
        self.joint_timestamps = joint_data.iloc[:, 0].values
        self.joint_positions = np.array(joint_data.iloc[:, 1].tolist())
        
        logging.info(f"Loading RGB images from {rgb_dir}")
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.rgb_dir = rgb_dir
        
        logging.info("Interpolating joint positions to match image frames...")
        interp_funcs = [
            interp1d(self.joint_timestamps, self.joint_positions[:, i], kind='linear', fill_value='extrapolate')
            for i in range(self.joint_positions.shape[1])
        ]
        
        frame_timestamps = np.linspace(self.joint_timestamps[0], self.joint_timestamps[-1], len(self.rgb_files))
        self.interpolated_joints = np.stack([f(frame_timestamps) for f in interp_funcs], axis=1)
        
        logging.info(f"Dataset initialized with {len(self)} samples.")

    def __len__(self):
        return len(self.rgb_files) - self.seq_len - self.future_steps

    def __getitem__(self, idx):
        rgb_seq = []
        joint_seq = []
        
        for i in range(self.seq_len):
            img_file = self.rgb_files[idx + i]
            img_path = os.path.join(self.rgb_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = rgb_transform(img)
            rgb_seq.append(img)
            joint_seq.append(self.interpolated_joints[idx + i])
        
        rgb_seq = torch.stack(rgb_seq)
        joint_seq = torch.tensor(np.array(joint_seq), dtype=torch.float32)
        
        future_joints = self.interpolated_joints[idx + self.seq_len : idx + self.seq_len + self.future_steps]
        future_joints = torch.tensor(future_joints, dtype=torch.float32)
        
        return rgb_seq, joint_seq, future_joints

# ---------------------
# Build Dataset
# ---------------------
logging.info("Initializing dataset...")
dataset = ImitationDataset(rgb_dir, robot_data_file)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

logging.info(f"Dataset split: {train_size} train, {val_size} val, {test_size} test.")

# ---------------------
# CNN Model Definition
# ---------------------
class VisionJointPlanner(nn.Module):
    def __init__(self, future_steps=FUTURE_STEPS):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.conv1 = nn.Conv1d(in_channels=self.feature_dim + 6, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, future_steps * 6)

    def forward(self, rgb_seq, joint_seq):
        B, S, C, H, W = rgb_seq.shape
        rgb_seq = rgb_seq.view(B * S, C, H, W)
        vision_features = self.feature_extractor(rgb_seq).view(B, S, self.feature_dim)

        fused_input = torch.cat([vision_features, joint_seq], dim=2).permute(0, 2, 1)
        x = torch.relu(self.conv1(fused_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.mean(dim=2)  # Global Average Pooling
        x = self.fc(x)
        return x.view(B, FUTURE_STEPS, 6)

logging.info("Model architecture defined.")

# ---------------------
# Training Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionJointPlanner().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float("inf")
train_losses, val_losses = [], []

# ---------------------
# Training Loop
# ---------------------
logging.info("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    
    for rgb_imgs, joint_seq, future_joints in train_loader:
        rgb_imgs, joint_seq, future_joints = rgb_imgs.to(device), joint_seq.to(device), future_joints.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_imgs, joint_seq)
        loss = criterion(outputs, future_joints)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for rgb_imgs, joint_seq, future_joints in val_loader:
            outputs = model(rgb_imgs.to(device), joint_seq.to(device))
            loss = criterion(outputs, future_joints.to(device))
            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    logging.info(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the model if we have a new best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(models_dir, f"best_loss_{args.bag}.pth")
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

# Save final model after training
final_model_path = os.path.join(models_dir, f"final_{args.bag}.pth")
torch.save(model.state_dict(), final_model_path)
logging.info(f"Final model saved to {final_model_path}")

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.savefig(os.path.join(plots_dir, f"loss_plot_{args.bag}.png"))
plt.show()
