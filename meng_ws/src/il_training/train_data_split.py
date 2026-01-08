import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for a non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
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
parser = argparse.ArgumentParser(
    description="Train an imitation learning model using CNN with RGB images + robot data from continuous expert clips")
parser.add_argument("--bag", type=str, required=True, help="Name of the rosbag to process (without extension)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
parser.add_argument("--seq_len", type=int, default=5, help="Number of past frames to consider.")
parser.add_argument("--future_steps", type=int, default=5, help="Number of future timesteps to predict.")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
args = parser.parse_args()

# Additional hyperparameter for continuity loss weight.
lambda_cont = 0.1  # You can adjust this weight as needed.

# ---------------------
# Paths & Directories
# ---------------------
base_clip_dir = os.path.join("saved_clips", args.bag)

models_dir = "models"
plots_dir = "plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

processed_dir = "processed_dataset"
train_processed_dir = os.path.join(processed_dir, "training")
val_processed_dir = os.path.join(processed_dir, "validation")
test_processed_dir = os.path.join(processed_dir, "testing")
os.makedirs(train_processed_dir, exist_ok=True)
os.makedirs(val_processed_dir, exist_ok=True)
os.makedirs(test_processed_dir, exist_ok=True)

# ---------------------
# Transforms
# ---------------------
rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------------
# Dataset Definition: ClipWindowDataset
# ---------------------
def load_clips(base_clip_dir):
    clip_dirs = [os.path.join(base_clip_dir, d) for d in os.listdir(base_clip_dir)
                 if os.path.isdir(os.path.join(base_clip_dir, d))]
    clips = []
    for clip_dir in clip_dirs:
        rgb_folder = os.path.join(clip_dir, "rgb")
        joint_file = os.path.join(clip_dir, "joint_states.json")
        if not os.path.exists(rgb_folder) or not os.path.exists(joint_file):
            logging.warning(f"Skipping {clip_dir}: missing 'rgb' folder or 'joint_states.json'")
            continue
        with open(joint_file, 'r') as f:
            joint_data = json.load(f)
        joint_states = []
        for state in joint_data:
            if "position" in state:
                joint_states.append(state["position"])
        joint_states = np.array(joint_states)
        rgb_files = sorted(os.listdir(rgb_folder))
        clip = {
            "clip_dir": clip_dir,
            "rgb_folder": rgb_folder,
            "rgb_files": rgb_files,
            "joint_states": joint_states
        }
        clips.append(clip)
    logging.info(f"Loaded {len(clips)} clip(s) from {base_clip_dir}")
    return clips

class ClipWindowDataset(Dataset):
    def __init__(self, clips, seq_len, future_steps):
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.samples = []  # Each sample: (clip, start_index)
        for clip in clips:
            L = min(len(clip["rgb_files"]), len(clip["joint_states"]))
            for start in range(0, L - (seq_len + future_steps) + 1):
                self.samples.append((clip, start))
        logging.info(f"Total sliding-window samples generated: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        clip, start = self.samples[idx]
        rgb_seq = []
        joint_seq = []
        for i in range(self.seq_len):
            img_file = clip["rgb_files"][start + i]
            img_path = os.path.join(clip["rgb_folder"], img_file)
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = rgb_transform(img)
            rgb_seq.append(img)
            joint_seq.append(clip["joint_states"][start + i])
        rgb_seq = torch.stack(rgb_seq)
        joint_seq = torch.tensor(np.array(joint_seq), dtype=torch.float32)
        future_joints = clip["joint_states"][start + self.seq_len : start + self.seq_len + self.future_steps]
        future_joints = torch.tensor(np.array(future_joints), dtype=torch.float32)
        return rgb_seq, joint_seq, future_joints

# ---------------------
# Build Dataset & Sequential Split (Split by clip)
# ---------------------
clips = load_clips(base_clip_dir)
num_clips = len(clips)
if num_clips == 0:
    raise RuntimeError("No valid clip directories found!")
train_clips = clips[:int(0.8 * num_clips)]
val_clips = clips[int(0.8 * num_clips):int(0.9 * num_clips)]
test_clips = clips[int(0.9 * num_clips):]

train_dataset = ClipWindowDataset(train_clips, seq_len=args.seq_len, future_steps=args.future_steps)
val_dataset = ClipWindowDataset(val_clips, seq_len=args.seq_len, future_steps=args.future_steps)
test_dataset = ClipWindowDataset(test_clips, seq_len=args.seq_len, future_steps=args.future_steps)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

logging.info(f"Dataset split by clips: {len(train_clips)} train clips, {len(val_clips)} val clips, {len(test_clips)} test clips.")
logging.info(f"Sliding-window samples: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test.")

# ---------------------
# (Optional) Save Processed Dataset to Disk
# ---------------------
def save_split_dataset(dataset, split_dir):
    logging.info(f"Saving {len(dataset)} samples to {split_dir}...")
    for i, sample in enumerate(dataset):
        rgb_seq, joint_seq, future_joints = sample
        rgb_seq_np = rgb_seq.numpy()      
        joint_seq_np = joint_seq.numpy()  
        future_joints_np = future_joints.numpy()  
        filename = os.path.join(split_dir, f"sample_{i:05d}.npz")
        np.savez(filename, rgb_seq=rgb_seq_np, joint_seq=joint_seq_np, future_joints=future_joints_np)
    logging.info("Finished saving processed dataset.")

save_split_dataset(train_dataset, train_processed_dir)
save_split_dataset(val_dataset, val_processed_dir)
save_split_dataset(test_dataset, test_processed_dir)

# ---------------------
# CNN Model Definition
# ---------------------
class VisionJointPlanner(nn.Module):
    def __init__(self, future_steps):
        super().__init__()
        self.future_steps = future_steps
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.conv1 = nn.Conv1d(in_channels=self.feature_dim + 6, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, self.future_steps * 6)

    def forward(self, rgb_seq, joint_seq):
        B, S, C, H, W = rgb_seq.shape
        rgb_seq = rgb_seq.view(B * S, C, H, W)
        vision_features = self.feature_extractor(rgb_seq).view(B, S, self.feature_dim)
        fused_input = torch.cat([vision_features, joint_seq], dim=2).permute(0, 2, 1)
        x = torch.relu(self.conv1(fused_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.mean(dim=2)
        x = self.fc(x)
        return x.view(B, self.future_steps, 6)

logging.info("Model architecture defined.")

# ---------------------
# Forward Kinematics Functions
# ---------------------
def dh_transform(theta, d, alpha, a):
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics(joint_angles):
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
# Training Setup and Checkpointing
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionJointPlanner(future_steps=args.future_steps).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

best_val_loss = float("inf")
train_losses, val_losses = [], []

# ---------------------
# Training Loop with Continuity Loss
# ---------------------
logging.info("Starting training...")
for epoch in range(args.epochs):
    model.train()
    running_train_loss = 0.0
    
    for rgb_imgs, joint_seq, future_joints in train_loader:
        rgb_imgs, joint_seq, future_joints = rgb_imgs.to(device), joint_seq.to(device), future_joints.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_imgs, joint_seq)
        mse_loss = criterion(outputs, future_joints)
        # Ensure the last input joint state is on the same device as outputs.
        # change the continuity loss to use the last joint state in the sequence as the mean squared error.
        mse_cont = nn.MSELoss()
        continuity_loss = lambda_cont * mse_cont(outputs[:, 0, :], joint_seq[:, -1, :].to(outputs.device))
        loss = mse_loss + continuity_loss
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
            mse_loss = criterion(outputs, future_joints.to(device))
            mse_cont = nn.MSELoss()
            continuity_loss = lambda_cont * mse_cont(outputs[:, 0, :], joint_seq[:, -1, :].to(outputs.device))
            loss = mse_loss + continuity_loss
            running_val_loss += loss.item()
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    logging.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save the model if new best validation loss is achieved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(models_dir, f"best_loss_mse_{args.bag}.pth")
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

# Save final model after training
final_model_path = os.path.join(models_dir, f"final_mse_{args.bag}.pth")
torch.save(model.state_dict(), final_model_path)
logging.info(f"Final model saved to {final_model_path}")

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.savefig(os.path.join(plots_dir, f"loss_plot_mse_{args.bag}.png"))
plt.show()
