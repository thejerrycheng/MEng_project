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
])import rosbag
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import json
import ast
import tkinter as tk
from tkinter import simpledialog
from scipy.interpolate import interp1d

def read_images_from_rosbag(bag_file, topic):
    bridge = CvBridge()
    images = []
    timestamps = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                images.append(cv_image)
                timestamps.append(t.to_sec())
            except Exception as e:
                # print(f"Error converting image:", e)
                print("We have problem here ")
    return images, timestamps

def read_joint_states_from_rosbag(bag_file, topic):
    timestamps = []
    positions = []
    velocities = []
    torques = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                timestamps.append(t.to_sec())
                positions.append(msg.position)
                velocities.append(msg.velocity)
                torques.append(msg.effort)
            except Exception as e:
                print(f"Error processing joint state message: {e}")
    
    return np.array(timestamps), np.array(positions), np.array(velocities), np.array(torques)

def interpolate_joint_states(image_timestamps, joint_timestamps, positions, velocities, torques):
    if len(joint_timestamps) == 0:
        print("Warning: No joint states available, using default zero values.")
        return [{"position": [0.0] * 6, "velocity": [0.0] * 6, "effort": [0.0] * 6} for _ in range(len(image_timestamps))]
    
    num_joints = positions.shape[1]
    interpolated_joint_states = []
    
    for i, timestamp in enumerate(image_timestamps):
        interp_joint_state = {
            "position": [], "velocity": [], "effort": []
        }
        for data, key in zip([positions, velocities, torques], ["position", "velocity", "effort"]):
            interp_funcs = [interp1d(joint_timestamps, data[:, j], kind='linear', fill_value='extrapolate') for j in range(num_joints)]
            interp_joint_state[key] = [interp_func(timestamp) for interp_func in interp_funcs]
        interpolated_joint_states.append(interp_joint_state)
    
    return interpolated_joint_states

def normalize_depth_image(image):
    normalized_image = np.clip(image, 0, 10000)
    normalized_image = (normalized_image / 10000.0) * 255
    normalized_image = np.uint8(normalized_image)
    return cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

def display_images(depth_images, color_images, index):    
    color_image = color_images[index].copy()
    target_size = (600, 400)
    color_image = cv2.resize(color_image, target_size, interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 255, 255)
    thickness = 2
    position = (10, 30)
    cv2.putText(color_image, f"RGB Frame: {index}", position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    return color_image

def save_clip(depth_images, color_images, joint_states, start_index, end_index):
    clip_folder = "saved_clips"
    os.makedirs(clip_folder, exist_ok=True)
    clip_path = os.path.join(clip_folder, f"clip_{start_index}_to_{end_index}")
    os.makedirs(clip_path, exist_ok=True)
    depth_clip_folder = os.path.join(clip_path, "depth")
    color_clip_folder = os.path.join(clip_path, "rgb")
    joint_states_file = os.path.join(clip_path, "joint_states.json")
    os.makedirs(depth_clip_folder, exist_ok=True)
    os.makedirs(color_clip_folder, exist_ok=True)
    for i in range(start_index, end_index + 1):
        cv2.imwrite(os.path.join(depth_clip_folder, f"{i}.png"), normalize_depth_image(depth_images[i]))
        cv2.imwrite(os.path.join(color_clip_folder, f"{i}.png"), color_images[i])
    with open(joint_states_file, 'w') as f:
        json.dump(joint_states[start_index:end_index+1], f, indent=4)
    print(f"Saved clip from frame {start_index} to {end_index} at {clip_path}")

def main():
    parser = argparse.ArgumentParser(description='Read and display frames from a ROS bag file.')
    parser.add_argument('--bag', required=True, help='Path to the ROS bag file')
    args = parser.parse_args()
    bag_file = args.bag
    depth_topic = "/camera/depth/image_rect_raw"
    color_topic = "/camera/color/image_raw"
    joint_topic = "/ufactory/joint_states"
    depth_images, depth_timestamps = read_images_from_rosbag(bag_file, depth_topic)
    color_images, color_timestamps = read_images_from_rosbag(bag_file, color_topic)
    joint_timestamps, positions, velocities, torques = read_joint_states_from_rosbag(bag_file, joint_topic)
    interpolated_joint_states = interpolate_joint_states(depth_timestamps, joint_timestamps, positions, velocities, torques)
    index = 0
    total_frames = len(depth_images)
    clip_start = None
    while True:
        combined_image = display_images(depth_images, color_images, index)
        cv2.imshow('Frame Viewer', combined_image)
        key = cv2.waitKey(0) & 0xFF
        if key == 81:
            index = max(0, index - 1)
        elif key == 83:
            index = min(total_frames - 1, index + 1)
        elif key == ord('x'):
            if clip_start is None:
                clip_start = index
                print(f"Beginning of the clip selected at frame {clip_start}")
            else:
                clip_end = index
                print(f"End of the clip selected at frame {clip_end}")
        elif key == 13 and clip_start is not None:
            save_clip(depth_images, color_images, interpolated_joint_states, clip_start, index)
            clip_start = None
        elif key == ord('c'):
            clip_start = None
            print("Clip selection cleared.")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


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
class ImitationPolicy(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, output_dim=6):
        super(ImitationPolicy, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        self.lstm = nn.LSTM(input_size=518, hidden_size=256, num_layers=2, batch_first=True)  # Input: (batch, seq_len, 518)
        self.fc = nn.Linear(256, output_dim)  # Output: (batch, 6)

    def forward(self, depth, joint_angles):
        B, S, C, H, W = depth.shape  # (Batch, Sequence, Channels, Height, Width)
        depth = depth.view(B * S, C, H, W)  # Flatten sequence into batch
        depth_features = self.backbone(depth)  # Extract features (B*S, 512)
        depth_features = depth_features.view(B, S, -1)  # Reshape back to sequence format (B, S, 512)
        x = torch.cat((depth_features, joint_angles), dim=2)  # Concatenate depth and joint angles (B, S, 518)
        x, _ = self.lstm(x)  # LSTM for temporal dependencies (B, S, 256)
        x = self.fc(x[:, -1, :])  # Take last timestep output (B, 6)
        return x

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ImitationDataset(depth_data_folder, interpolated_joints)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ImitationPolicy().to(device)
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
        torch.save(model.state_dict(), os.path.join(models_dir, "depth_best_model.pth"))
    else:
        stopping_counter += 1
    
    # # Early stopping condition
    # if stopping_counter >= PATIENCE:
    #     print("Early stopping triggered.")
    #     break

# Save final model
torch.save(model.state_dict(), os.path.join(models_dir, "depth_final_model.pth"))

# Plot training loss
plt.figure()
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.grid()
plt.savefig(os.path.join(plots_dir, "loss_plot_depth.png"))
plt.show()
