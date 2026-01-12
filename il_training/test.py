import torch
import yaml
import numpy as np

from datasets.iris_dataset import list_episode_dirs, load_episode, EpisodeWindowDataset
from models.transformer_model import ACT_RGB
from losses.loss import batch_fk

# -------------------------------------------------
# Load config
# -------------------------------------------------
cfg = yaml.safe_load(open("config.yaml"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Load episodes
# -------------------------------------------------
dirs = list_episode_dirs(cfg["data_root"], cfg["bag_prefix"])
episodes = [load_episode(d) for d in dirs]
episodes = [e for e in episodes if e is not None]

if len(episodes) == 0:
    raise RuntimeError("No episodes found!")

# Test set = last 10% of episodes
N = len(episodes)
test_eps = episodes[int(0.9 * N):]

print(f"Loaded {len(test_eps)} test episodes")

test_ds = EpisodeWindowDataset(test_eps, cfg["seq_len"], cfg["future_steps"])
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers"]
)

print(f"Total test windows: {len(test_ds)}")

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = ACT_RGB(
    seq_len=cfg["seq_len"],
    future_steps=cfg["future_steps"],
    d_model=cfg["d_model"],
    nhead=cfg["nhead"],
    enc_layers=cfg["enc_layers"],
    dec_layers=cfg["dec_layers"],
    ff_dim=cfg["ff_dim"],
    dropout=cfg["dropout"]
).to(device)

ckpt_path = "outputs/models/best_model.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

print(f"Loaded checkpoint: {ckpt_path}")

# -------------------------------------------------
# Run Evaluation
# -------------------------------------------------
cart_errors = []

with torch.no_grad():
    for rgb, joint, goal_xyz, future in test_loader:
        rgb = rgb.to(device)
        joint = joint.to(device)
        goal_xyz = goal_xyz.to(device)

        pred = model(rgb, joint, goal_xyz)

        # last predicted joint
        q_last = joint[:, -1, :]
        q_pred_last = q_last + pred[:, -1, :]

        # FK → Cartesian
        xyz_pred = batch_fk(q_pred_last)

        # Euclidean error
        err = torch.norm(xyz_pred - goal_xyz, dim=1)
        cart_errors.append(err.cpu())

cart_errors = torch.cat(cart_errors).numpy()

# -------------------------------------------------
# Report
# -------------------------------------------------
print("\n===== TEST RESULTS =====")
print(f"Cartesian Goal Error:")
print(f"  Mean: {cart_errors.mean():.4f} m")
print(f"  Std : {cart_errors.std():.4f} m")
print(f"  Min : {cart_errors.min():.4f} m")
print(f"  Max : {cart_errors.max():.4f} m")

print("\nDone.")
