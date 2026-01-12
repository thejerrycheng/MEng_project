import os
import argparse
import logging
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------
# Logging
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# ---------------------
# Args
# ---------------------
parser = argparse.ArgumentParser(
    description="Train ACT (Transformer encoder–decoder) on processed IRIS episodes"
)
parser.add_argument("--data_root", type=str, default="processed_data")
parser.add_argument("--bag_prefix", type=str, required=True)
parser.add_argument("--use_depth", action="store_true")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seq_len", type=int, default=8)
parser.add_argument("--future_steps", type=int, default=15)
parser.add_argument("--patience", type=int, default=8)

parser.add_argument("--lambda_cont", type=float, default=0.05)
parser.add_argument("--lambda_goal", type=float, default=0.2)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--enc_layers", type=int, default=4)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)

args = parser.parse_args()
NUM_JOINTS = 6

torch.backends.cudnn.benchmark = True

# ---------------------
# Dirs
# ---------------------
models_dir = "models"
plots_dir = "plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ---------------------
# Transforms
# ---------------------
rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def depth_to_tensor(depth_img_16u):
    d = depth_img_16u.astype(np.float32)
    d = np.clip(d, 0.0, 10000.0) / 10000.0
    d = cv2.resize(d, (128, 128), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(d).unsqueeze(0)


# ---------------------
# Episode discovery
# ---------------------
def list_episode_dirs(data_root, bag_prefix):
    return sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if d.startswith(bag_prefix + "_episode_")
    ])


def load_episode(ep_dir):
    rgb_dir = os.path.join(ep_dir, "rgb")
    depth_dir = os.path.join(ep_dir, "depth")
    robot_csv = os.path.join(ep_dir, "robot", "joint_states.csv")

    if not os.path.isdir(rgb_dir) or not os.path.isfile(robot_csv):
        return None

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    if len(rgb_files) == 0:
        return None

    df = pd.read_csv(robot_csv)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    pos_cols = pos_cols[:NUM_JOINTS]

    joints = df[pos_cols].to_numpy(dtype=np.float32)

    T = min(len(rgb_files), joints.shape[0])
    rgb_files = rgb_files[:T]
    joints = joints[:T]

    depth_files = None
    if os.path.isdir(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])
        depth_files = depth_files[:T] if depth_files else None

    goal_joint = joints[-1].copy()

    return {
        "rgb_dir": rgb_dir,
        "depth_dir": depth_dir if depth_files else None,
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "joints": joints,
        "goal_joint": goal_joint
    }


# ---------------------
# Dataset
# ---------------------
class EpisodeWindowDataset(Dataset):
    def __init__(self, episodes, seq_len, future_steps, use_depth=False):
        self.episodes = episodes
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.use_depth = use_depth
        self.samples = []

        for ei, ep in enumerate(episodes):
            T = ep["joints"].shape[0]
            max_start = T - (seq_len + future_steps) + 1
            for s in range(max_start):
                self.samples.append((ei, s))

        logging.info(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, ep, idx):
        path = os.path.join(ep["rgb_dir"], ep["rgb_files"][idx])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_transform(img)

    def _read_depth(self, ep, idx):
        if ep["depth_dir"] is None:
            return torch.zeros(1, 128, 128)
        path = os.path.join(ep["depth_dir"], ep["depth_files"][idx])
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d.ndim == 3:
            d = d[:, :, 0]
        return depth_to_tensor(d)

    def __getitem__(self, idx):
        ei, start = self.samples[idx]
        ep = self.episodes[ei]

        rgb_seq, depth_seq, joint_seq = [], [], []

        for i in range(self.seq_len):
            fi = start + i
            rgb_seq.append(self._read_rgb(ep, fi))
            if self.use_depth:
                depth_seq.append(self._read_depth(ep, fi))
            joint_seq.append(ep["joints"][fi])

        rgb_seq = torch.stack(rgb_seq)
        joint_seq = torch.from_numpy(np.asarray(joint_seq)).float()

        q_last = ep["joints"][start + self.seq_len - 1]
        fut = ep["joints"][start + self.seq_len : start + self.seq_len + self.future_steps]
        fut_delta = fut - q_last[None, :]
        future = torch.tensor(fut_delta, dtype=torch.float32)

        goal = torch.tensor(ep["goal_joint"], dtype=torch.float32)

        if self.use_depth:
            depth_seq = torch.stack(depth_seq)
            return rgb_seq, depth_seq, joint_seq, goal, future

        return rgb_seq, joint_seq, goal, future


# ---------------------
# ACT Model (RGB)
# ---------------------
class ACT_RGB(nn.Module):
    def __init__(self, seq_len, future_steps, d_model, nhead, enc_layers, dec_layers, ff_dim, dropout):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.rgb_proj = nn.Linear(512, d_model)
        self.joint_proj = nn.Linear(NUM_JOINTS, d_model)
        self.goal_proj = nn.Linear(NUM_JOINTS, d_model)

        self.enc_pos = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, joint_seq, goal):
        B, S, C, H, W = rgb_seq.shape
        x = self.backbone(rgb_seq.view(B*S, C, H, W)).view(B, S, 512)
        rgb_tok = self.rgb_proj(x)
        joint_tok = self.joint_proj(joint_seq)
        enc = rgb_tok + joint_tok

        goal_tok = self.goal_proj(goal).unsqueeze(1)
        enc = torch.cat([enc, goal_tok], dim=1)
        enc = enc + self.enc_pos.unsqueeze(0)

        q = self.action_queries.unsqueeze(0).repeat(B, 1, 1)
        dec = self.transformer(enc, q)
        return self.head(dec)


# ---------------------
# Training
# ---------------------
def run_epoch(model, loader, device, optimizer=None, lambda_cont=0.05, lambda_goal=0.2, use_depth=False):
    train = optimizer is not None
    model.train(train)
    mse = nn.MSELoss()
    total, n = 0, 0

    for rgb_seq, joint_seq, goal, future in loader:
        rgb_seq = rgb_seq.to(device)
        joint_seq = joint_seq.to(device)
        goal = goal.to(device)
        future = future.to(device)

        pred = model(rgb_seq, joint_seq, goal)

        loss_mse = mse(pred, future)
        loss_cont = mse(pred[:, 0, :], torch.zeros_like(pred[:, 0, :]))

        q_last = joint_seq[:, -1, :]
        q_pred_last = q_last + pred[:, -1, :]
        loss_goal = mse(q_pred_last, goal)

        loss = loss_mse + lambda_cont * loss_cont + lambda_goal * loss_goal

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total += loss.item()
        n += 1

    return total / max(n, 1)


# ---------------------
# Load Episodes
# ---------------------
episode_dirs = list_episode_dirs(args.data_root, args.bag_prefix)
episodes = [load_episode(d) for d in episode_dirs]
episodes = [e for e in episodes if e is not None]

logging.info(f"Loaded {len(episodes)} episodes")

N = len(episodes)
n_train = max(1, int(0.8*N))
n_val = max(1, int(0.1*N))

train_eps = episodes[:n_train]
val_eps = episodes[n_train:n_train+n_val]
test_eps = episodes[n_train+n_val:]

train_ds = EpisodeWindowDataset(train_eps, args.seq_len, args.future_steps)
val_ds   = EpisodeWindowDataset(val_eps, args.seq_len, args.future_steps)
test_ds  = EpisodeWindowDataset(test_eps, args.seq_len, args.future_steps)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

# ---------------------
# Run Training
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ACT_RGB(args.seq_len, args.future_steps,
                args.d_model, args.nhead,
                args.enc_layers, args.dec_layers,
                args.ff_dim, args.dropout).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)

best_val = 1e9
for epoch in range(args.epochs):
    tr = run_epoch(model, train_loader, device, optimizer, args.lambda_cont, args.lambda_goal)
    va = run_epoch(model, val_loader, device, None, args.lambda_cont, args.lambda_goal)

    logging.info(f"Epoch {epoch+1:03d} | Train {tr:.6f} | Val {va:.6f}")

    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), f"models/best_act_{args.bag_prefix}.pth")

torch.save(model.state_dict(), f"models/final_act_{args.bag_prefix}.pth")
logging.info("Training complete.")
