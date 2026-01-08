import os
import argparse
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use 'Agg' for a non-interactive backend
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
    description="Test the imitation learning model using CNN with grayscale images + robot data from continuous expert clips")
parser.add_argument("--bag", type=str, required=True, help="Name of the rosbag to process (without extension)")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing.")
parser.add_argument("--seq_len", type=int, default=5, help="Number of past frames to consider.")
parser.add_argument("--future_steps", type=int, default=5, help="Number of future timesteps to predict.")
args = parser.parse_args()

# ---------------------
# Paths & Directories
# ---------------------
# Base directory for clips: saved_clips/{bag}/clip_<start>_to_<end>
base_clip_dir = os.path.join("saved_clips", args.bag)

# ---------------------
# Transforms
# ---------------------
rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    # Uncomment the next line if you wish to convert to grayscale:
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------------------
# Dataset Definition: ClipWindowDataset
# ---------------------
def load_clips(base_clip_dir):
    # List all clip directories in sorted order (preserving temporal order)
    clip_dirs = sorted([os.path.join(base_clip_dir, d) for d in os.listdir(base_clip_dir)
                        if os.path.isdir(os.path.join(base_clip_dir, d))])
    clips = []
    for clip_dir in clip_dirs:
        rgb_folder = os.path.join(clip_dir, "rgb")
        joint_file = os.path.join(clip_dir, "joint_states.json")
        if not os.path.exists(rgb_folder) or not os.path.exists(joint_file):
            logging.warning(f"Skipping {clip_dir}: missing 'rgb' folder or 'joint_states.json'")
            continue
        with open(joint_file, 'r') as f:
            joint_data = json.load(f)
        # Extract joint positions (each state must have key "position")
        joint_states = [state["position"] for state in joint_data if "position" in state]
        joint_states = np.array(joint_states)
        rgb_files = sorted(os.listdir(rgb_folder))
        clip = {
            "clip_id": os.path.basename(clip_dir),
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
        rgb_seq = torch.stack(rgb_seq)  # shape: (seq_len, C, H, W)
        joint_seq = torch.tensor(np.array(joint_seq), dtype=torch.float32)
        future_joints = clip["joint_states"][start + self.seq_len : start + self.seq_len + self.future_steps]
        future_joints = torch.tensor(np.array(future_joints), dtype=torch.float32)
        return rgb_seq, joint_seq, future_joints, clip["clip_id"], start

# ---------------------
# Build Test Dataset (Using all clips sequentially)
# ---------------------
clips = load_clips(base_clip_dir)
test_dataset = ClipWindowDataset(clips, seq_len=args.seq_len, future_steps=args.future_steps)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
logging.info(f"Test dataset built from {len(clips)} clip(s) with {len(test_dataset)} sliding-window samples.")

# ---------------------
# VisionJointPlanner Model Definition
# ---------------------
FUTURE_STEPS = args.future_steps
class VisionJointPlanner(nn.Module):
    def __init__(self, future_steps):
        super(VisionJointPlanner, self).__init__()
        self.future_steps = future_steps
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.conv1 = nn.Conv1d(in_channels=self.feature_dim + 6,
                               out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256,
                               out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=64, kernel_size=3, padding=1)
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
    """
    Compute the forward kinematics of the xArm Lite 6.
    
    Parameters:
      joint_angles: (N, 6) numpy array (angles in degrees)
      
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
# Load Trained Model
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionJointPlanner(future_steps=args.future_steps).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
logging.info(f"Model loaded from {args.model_path}.")

# ---------------------
# Inference: Group predictions by clip and reconstruct continuous trajectories
# ---------------------
logging.info("Running inference on test dataset (sequential order)...")
# Dictionaries to store end-effector positions and joint angles per clip.
pred_by_clip = {}
gt_by_clip = {}
joint_pred_by_clip = {}
joint_gt_by_clip = {}
# Also, store continuity errors per sample.
continuity_errors = {}

with torch.no_grad():
    for rgb_imgs, joint_seq, future_joints, clip_id, start in test_loader:
        clip_id = clip_id[0]  # string
        start = start[0]      # integer
        
        rgb_imgs = rgb_imgs.to(device)
        joint_seq = joint_seq.to(device)
        predicted_future = model(rgb_imgs, joint_seq)  # (1, FUTURE_STEPS, 6)
        predicted_future = predicted_future.cpu().numpy().squeeze(0)  # (FUTURE_STEPS, 6)
        future_joints_np = future_joints.cpu().numpy().squeeze(0)       # (FUTURE_STEPS, 6)
        
        # Compute continuity error: difference between last input joint and first predicted joint.
        last_input_joint = joint_seq[-1].cpu().numpy()  # shape (6,)
        continuity_error = np.linalg.norm(predicted_future[0] - last_input_joint)
        # Store continuity error per sliding-window sample.
        if clip_id not in continuity_errors:
            continuity_errors[clip_id] = []
        continuity_errors[clip_id].append(continuity_error)
        
        pred_positions = forward_kinematics(predicted_future)  # (FUTURE_STEPS, 3)
        gt_positions = forward_kinematics(future_joints_np)      # (FUTURE_STEPS, 3)
        
        for i in range(args.future_steps):
            frame_idx = int(start) + args.seq_len + i
            if clip_id not in pred_by_clip:
                pred_by_clip[clip_id] = {}
                gt_by_clip[clip_id] = {}
                joint_pred_by_clip[clip_id] = {}
                joint_gt_by_clip[clip_id] = {}
            if frame_idx not in pred_by_clip[clip_id]:
                pred_by_clip[clip_id][frame_idx] = []
                gt_by_clip[clip_id][frame_idx] = []
                joint_pred_by_clip[clip_id][frame_idx] = []
                joint_gt_by_clip[clip_id][frame_idx] = []
            pred_by_clip[clip_id][frame_idx].append(pred_positions[i])
            gt_by_clip[clip_id][frame_idx].append(gt_positions[i])
            joint_pred_by_clip[clip_id][frame_idx].append(predicted_future[i])
            joint_gt_by_clip[clip_id][frame_idx].append(future_joints_np[i])

# Reconstruct continuous trajectories by averaging overlapping predictions.
continuous_preds = {}
continuous_gt = {}
continuous_joint_pred = {}
continuous_joint_gt = {}

for clip_id in pred_by_clip:
    frame_indices = sorted(pred_by_clip[clip_id].keys())
    pred_list = []
    gt_list = []
    joint_pred_list = []
    joint_gt_list = []
    for fi in frame_indices:
        pred_avg = np.mean(pred_by_clip[clip_id][fi], axis=0)
        gt_avg = np.mean(gt_by_clip[clip_id][fi], axis=0)
        joint_pred_avg = np.mean(joint_pred_by_clip[clip_id][fi], axis=0)
        joint_gt_avg = np.mean(joint_gt_by_clip[clip_id][fi], axis=0)
        pred_list.append(pred_avg)
        gt_list.append(gt_avg)
        joint_pred_list.append(joint_pred_avg)
        joint_gt_list.append(joint_gt_avg)
    continuous_preds[clip_id] = np.array(pred_list)
    continuous_gt[clip_id] = np.array(gt_list)
    continuous_joint_pred[clip_id] = np.array(joint_pred_list)
    continuous_joint_gt[clip_id] = np.array(joint_gt_list)

# ---------------------
# Visualization: Plot continuous end-effector trajectories and joint angle comparisons for each clip
# ---------------------
for clip_id in continuous_preds:
    avg_cont_err = np.mean(continuity_errors[clip_id]) if clip_id in continuity_errors else None
    
    # Plot end-effector trajectory comparison
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(continuous_gt[clip_id][:, 0],
            continuous_gt[clip_id][:, 1],
            continuous_gt[clip_id][:, 2],
            label="Ground Truth EE", color="blue")
    ax.plot(continuous_preds[clip_id][:, 0],
            continuous_preds[clip_id][:, 1],
            continuous_preds[clip_id][:, 2],
            label="Predicted EE", linestyle="dashed", color="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_str = f"End-Effector Trajectory for Clip {clip_id}"
    if avg_cont_err is not None:
        title_str += f"\nAvg. Continuity Error: {avg_cont_err:.2f}"
    ax.set_title(title_str)
    ax.legend()
    plt.show()
    
    # Plot joint angle comparisons: 6 subplots, one per joint.
    fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
    time_steps = np.arange(len(continuous_joint_gt[clip_id]))
    for joint in range(6):
        axes[joint].plot(time_steps, continuous_joint_gt[clip_id][:, joint],
                         label="GT Joint {}".format(joint+1), color="blue")
        axes[joint].plot(time_steps, continuous_joint_pred[clip_id][:, joint],
                         label="Pred Joint {}".format(joint+1), linestyle="dashed", color="red")
        axes[joint].set_ylabel(f"Joint {joint+1}")
        axes[joint].legend(loc="upper right")
    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f"Joint Angle Comparison for Clip {clip_id}")
    plt.tight_layout()
    plt.show()

logging.info("Visualization complete.")
