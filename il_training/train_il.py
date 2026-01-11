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
parser.add_argument("--data_root", type=str, default="processed_data",
                    help="Root folder containing <bag>_episode_XXXX folders")
parser.add_argument("--bag_prefix", type=str, required=True,
                    help="Prefix of episode folders, e.g. test3_20260110_212918")

parser.add_argument("--use_depth", action="store_true",
                    help="Use RGB+Depth (two encoders) instead of RGB only")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--epochs", type=int, default=50)

parser.add_argument("--seq_len", type=int, default=8,
                    help="Number of past frames used as input")
parser.add_argument("--future_steps", type=int, default=15,
                    help="Number of future steps predicted (action chunk length)")

parser.add_argument("--patience", type=int, default=8,
                    help="Early stopping patience (val loss)")

parser.add_argument("--lambda_cont", type=float, default=0.05,
                    help="Continuity loss weight")
parser.add_argument("--lambda_goal", type=float, default=0.2,
                    help="Goal loss weight (penalize last predicted step vs goal)")

parser.add_argument("--num_workers", type=int, default=4)

# ACT model sizes
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--enc_layers", type=int, default=4)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)

args = parser.parse_args()

NUM_JOINTS = 6

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

# depth: read 16-bit png; normalize to [0,1] then 1-channel tensor
def depth_to_tensor(depth_img_16u):
    d = depth_img_16u.astype(np.float32)
    d = np.clip(d, 0.0, 10000.0) / 10000.0
    d = cv2.resize(d, (128, 128), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(d).unsqueeze(0)  # (1,H,W)


# ---------------------
# Episode discovery
# ---------------------
def list_episode_dirs(data_root, bag_prefix):
    dirs = []
    for name in sorted(os.listdir(data_root)):
        if name.startswith(bag_prefix + "_episode_"):
            ep = os.path.join(data_root, name)
            if os.path.isdir(ep):
                dirs.append(ep)
    return dirs


def load_episode(ep_dir):
    rgb_dir = os.path.join(ep_dir, "rgb")
    depth_dir = os.path.join(ep_dir, "depth")
    robot_csv = os.path.join(ep_dir, "robot", "joint_states.csv")

    if not os.path.isdir(rgb_dir) or not os.path.isfile(robot_csv):
        logging.warning(f"Skipping {ep_dir}: missing rgb/ or robot/joint_states.csv")
        return None

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(".png")])
    if len(rgb_files) == 0:
        logging.warning(f"Skipping {ep_dir}: no rgb frames")
        return None

    df = pd.read_csv(robot_csv)

    # Identify position columns
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    if len(pos_cols) == 0:
        # fallback: any numeric cols excluding timestamp
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        pos_cols = numeric_cols

    if len(pos_cols) == 0:
        logging.warning(f"Skipping {ep_dir}: no joint position columns found in {robot_csv}")
        return None

    pos_cols = pos_cols[:NUM_JOINTS]
    joints = df[pos_cols].to_numpy(dtype=np.float32)  # (T,6)

    timestamps = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df.columns else None

    # Align length with rgb frames
    T = min(len(rgb_files), joints.shape[0])
    rgb_files = rgb_files[:T]
    joints = joints[:T]
    timestamps = timestamps[:T] if timestamps is not None else None

    depth_files = None
    if os.path.isdir(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(".png")])
        if depth_files:
            depth_files = depth_files[:T]
        else:
            depth_files = None

    # Per-user constraint: assume each rosbag has the same goal
    # We use goal joint configuration = last joint vector in this episode.
    goal_joint = joints[-1].copy()  # (6,)

    return {
        "ep_dir": ep_dir,
        "rgb_dir": rgb_dir,
        "depth_dir": depth_dir if depth_files is not None else None,
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "joints": joints,
        "timestamps": timestamps,
        "pos_cols": pos_cols,
        "goal_joint": goal_joint,
    }


# ---------------------
# Dataset (sliding windows)
# ---------------------
class EpisodeWindowDataset(Dataset):
    def __init__(self, episodes, seq_len, future_steps, use_depth=False):
        self.episodes = episodes
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.use_depth = use_depth
        self.samples = []  # (ep_idx, start)

        for ep_idx, ep in enumerate(self.episodes):
            T = ep["joints"].shape[0]
            max_start = T - (seq_len + future_steps) + 1
            for start in range(max_start):
                self.samples.append((ep_idx, start))

        logging.info(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, ep, frame_idx):
        img_path = os.path.join(ep["rgb_dir"], ep["rgb_files"][frame_idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Missing RGB: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_transform(img)

    def _read_depth(self, ep, frame_idx):
        if ep["depth_dir"] is None or ep["depth_files"] is None:
            return torch.zeros(1, 128, 128)
        dep_path = os.path.join(ep["depth_dir"], ep["depth_files"][frame_idx])
        d = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        if d is None:
            return torch.zeros(1, 128, 128)
        if d.ndim == 3:
            d = d[:, :, 0]
        return depth_to_tensor(d)

    def __getitem__(self, idx):
        ep_idx, start = self.samples[idx]
        ep = self.episodes[ep_idx]

        rgb_seq = []
        depth_seq = []
        joint_seq = []

        for i in range(self.seq_len):
            fi = start + i
            rgb_seq.append(self._read_rgb(ep, fi))
            if self.use_depth:
                depth_seq.append(self._read_depth(ep, fi))
            joint_seq.append(ep["joints"][fi])

        rgb_seq = torch.stack(rgb_seq, dim=0)  # (S,3,H,W)
        joint_seq = torch.tensor(np.asarray(joint_seq), dtype=torch.float32)  # (S,6)

        fut = ep["joints"][start + self.seq_len : start + self.seq_len + self.future_steps]
        future_joints = torch.tensor(np.asarray(fut), dtype=torch.float32)  # (F,6)

        goal = torch.tensor(ep["goal_joint"], dtype=torch.float32)  # (6,)

        if self.use_depth:
            depth_seq = torch.stack(depth_seq, dim=0)  # (S,1,H,W)
            return rgb_seq, depth_seq, joint_seq, goal, future_joints

        return rgb_seq, joint_seq, goal, future_joints


# ---------------------
# ACT Model (Transformer encoder–decoder)
# ---------------------
class ACT_RGB(nn.Module):
    def __init__(self, seq_len, future_steps, d_model=128, nhead=4,
                 enc_layers=4, dec_layers=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.d_model = d_model

        # Vision backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        self.rgb_dim = 512

        # Projections into token space
        self.rgb_proj = nn.Linear(self.rgb_dim, d_model)
        self.joint_proj = nn.Linear(NUM_JOINTS, d_model)
        self.goal_proj = nn.Linear(NUM_JOINTS, d_model)

        # Positional embeddings for encoder tokens (S + 1 goal token)
        self.enc_pos = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.02)

        # Learnable action query tokens for decoder (F)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        # Head to joints
        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, joint_seq, goal):
        # rgb_seq: (B,S,3,H,W), joint_seq: (B,S,6), goal: (B,6)
        B, S, C, H, W = rgb_seq.shape

        x = rgb_seq.view(B * S, C, H, W)
        feat = self.rgb_backbone(x).view(B, S, self.rgb_dim)  # (B,S,512)

        rgb_tok = self.rgb_proj(feat)                         # (B,S,D)
        joint_tok = self.joint_proj(joint_seq)                # (B,S,D)
        enc_tok = rgb_tok + joint_tok                         # (B,S,D)

        goal_tok = self.goal_proj(goal).unsqueeze(1)          # (B,1,D)
        enc_tok = torch.cat([enc_tok, goal_tok], dim=1)       # (B,S+1,D)

        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)         # (B,S+1,D)

        # Decoder queries
        q = self.action_queries.unsqueeze(0).repeat(B, 1, 1)  # (B,F,D)

        dec_out = self.transformer(enc_tok, q)                # (B,F,D)
        pred = self.head(dec_out)                             # (B,F,6)
        return pred


class ACT_RGBD(nn.Module):
    def __init__(self, seq_len, future_steps, d_model=128, nhead=4,
                 enc_layers=4, dec_layers=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.d_model = d_model

        # RGB backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.rgb_dim = 512

        # Depth backbone (lightweight)
        self.depth_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.depth_dim = 64

        # Projections into token space
        self.rgb_proj = nn.Linear(self.rgb_dim, d_model)
        self.depth_proj = nn.Linear(self.depth_dim, d_model)
        self.joint_proj = nn.Linear(NUM_JOINTS, d_model)
        self.goal_proj = nn.Linear(NUM_JOINTS, d_model)

        self.enc_pos = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, depth_seq, joint_seq, goal):
        # rgb_seq: (B,S,3,H,W), depth_seq: (B,S,1,H,W), joint_seq: (B,S,6), goal: (B,6)
        B, S, _, H, W = rgb_seq.shape

        # RGB tokens
        rgb = rgb_seq.view(B * S, 3, H, W)
        rgb_feat = self.rgb_backbone(rgb).view(B, S, self.rgb_dim)   # (B,S,512)
        rgb_tok = self.rgb_proj(rgb_feat)                            # (B,S,D)

        # Depth tokens
        d = depth_seq.view(B * S, 1, H, W)
        d_feat = self.depth_backbone(d).view(B, S, self.depth_dim)   # (B,S,64)
        d_tok = self.depth_proj(d_feat)                              # (B,S,D)

        joint_tok = self.joint_proj(joint_seq)                       # (B,S,D)

        # Fuse tokens by sum (stable)
        enc_tok = rgb_tok + d_tok + joint_tok                        # (B,S,D)

        goal_tok = self.goal_proj(goal).unsqueeze(1)                 # (B,1,D)
        enc_tok = torch.cat([enc_tok, goal_tok], dim=1)              # (B,S+1,D)
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        q = self.action_queries.unsqueeze(0).repeat(B, 1, 1)         # (B,F,D)
        dec_out = self.transformer(enc_tok, q)                       # (B,F,D)
        pred = self.head(dec_out)                                    # (B,F,6)
        return pred


# ---------------------
# Train / Eval
# ---------------------
def run_epoch(model, loader, device, optimizer=None,
              lambda_cont=0.05, lambda_goal=0.2, use_depth=False):
    train = optimizer is not None
    model.train(train)

    mse = nn.MSELoss()
    total = 0.0
    n = 0

    for batch in loader:
        if use_depth:
            rgb_seq, depth_seq, joint_seq, goal, future = batch
            rgb_seq = rgb_seq.to(device, non_blocking=True)
            depth_seq = depth_seq.to(device, non_blocking=True)
            joint_seq = joint_seq.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            pred = model(rgb_seq, depth_seq, joint_seq, goal)
        else:
            rgb_seq, joint_seq, goal, future = batch
            rgb_seq = rgb_seq.to(device, non_blocking=True)
            joint_seq = joint_seq.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            pred = model(rgb_seq, joint_seq, goal)

        # Core imitation loss on chunk
        loss_mse = mse(pred, future)

        # Continuity: first predicted step should match last observed joint in the input window
        loss_cont = mse(pred[:, 0, :], joint_seq[:, -1, :])

        # Goal loss: last predicted step should approach the goal joint configuration
        # (assume each rosbag/episode set shares same goal; goal token is fed to network)
        loss_goal = mse(pred[:, -1, :], goal)

        loss = loss_mse + lambda_cont * loss_cont + lambda_goal * loss_goal

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        n += 1

    return total / max(1, n)


# ---------------------
# Main data setup
# ---------------------
episode_dirs = list_episode_dirs(args.data_root, args.bag_prefix)
if len(episode_dirs) == 0:
    raise RuntimeError(f"No episodes found under {args.data_root} with prefix {args.bag_prefix}")

episodes = []
for ep_dir in episode_dirs:
    ep = load_episode(ep_dir)
    if ep is not None:
        episodes.append(ep)

if len(episodes) == 0:
    raise RuntimeError("No valid episodes loaded.")

logging.info(f"Loaded {len(episodes)} episode(s). Example joint columns: {episodes[0]['pos_cols']}")

# Split by episode (no leakage)
N = len(episodes)
n_train = max(1, int(0.8 * N))
n_val = max(1, int(0.1 * N))

train_eps = episodes[:n_train]
val_eps = episodes[n_train:n_train + n_val]
test_eps = episodes[n_train + n_val:]

train_ds = EpisodeWindowDataset(train_eps, args.seq_len, args.future_steps, use_depth=args.use_depth)
val_ds = EpisodeWindowDataset(val_eps, args.seq_len, args.future_steps, use_depth=args.use_depth)
test_ds = EpisodeWindowDataset(test_eps, args.seq_len, args.future_steps, use_depth=args.use_depth)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

logging.info(f"Split: {len(train_eps)} train eps, {len(val_eps)} val eps, {len(test_eps)} test eps")
logging.info(f"Samples: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")


# ---------------------
# Model / Optim
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.use_depth:
    model = ACT_RGBD(
        seq_len=args.seq_len,
        future_steps=args.future_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout
    ).to(device)
else:
    model = ACT_RGB(
        seq_len=args.seq_len,
        future_steps=args.future_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout
    ).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)

best_val = float("inf")
tag = f"act_{args.bag_prefix}_{'rgbd' if args.use_depth else 'rgb'}_S{args.seq_len}_F{args.future_steps}"
best_path = os.path.join(models_dir, f"best_{tag}.pth")
final_path = os.path.join(models_dir, f"final_{tag}.pth")

train_losses, val_losses = [], []
bad_epochs = 0

logging.info("Starting training...")
for epoch in range(args.epochs):
    tr = run_epoch(
        model, train_loader, device,
        optimizer=optimizer,
        lambda_cont=args.lambda_cont,
        lambda_goal=args.lambda_goal,
        use_depth=args.use_depth
    )
    va = run_epoch(
        model, val_loader, device,
        optimizer=None,
        lambda_cont=args.lambda_cont,
        lambda_goal=args.lambda_goal,
        use_depth=args.use_depth
    )

    train_losses.append(tr)
    val_losses.append(va)

    logging.info(f"Epoch [{epoch+1:03d}/{args.epochs}]  Train: {tr:.6f}  Val: {va:.6f}")

    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), best_path)
        logging.info(f"✅ Saved best model: {best_path} (val={best_val:.6f})")
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info("Early stopping triggered.")
            break

torch.save(model.state_dict(), final_path)
logging.info(f"Saved final model: {final_path}")

# Test eval (load best)
model.load_state_dict(torch.load(best_path, map_location=device))
te = run_epoch(
    model, test_loader, device,
    optimizer=None,
    lambda_cont=args.lambda_cont,
    lambda_goal=args.lambda_goal,
    use_depth=args.use_depth
)
logging.info(f"Test loss (best model): {te:.6f}")

# Plot losses
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(plots_dir, f"loss_{tag}.png")
plt.savefig(plot_path, dpi=200)
logging.info(f"Saved plot: {plot_path}")