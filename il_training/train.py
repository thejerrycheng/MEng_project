#!/usr/bin/env python3
import os, math, yaml, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets.iris_dataset import list_episode_dirs, load_episode, EpisodeWindowDataset
from models.transformer_model import ACT_RGB
from losses.loss import act_loss, batch_fk


# -------------------------------------------------
# Load Config
# -------------------------------------------------
cfg = yaml.safe_load(open("configs/train.yaml"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)


# -------------------------------------------------
# Load Episodes from SSD
# -------------------------------------------------
dirs = list_episode_dirs(cfg["data"]["data_root"], cfg["data"]["bag_prefix"])
episodes = [load_episode(d) for d in dirs]
episodes = [e for e in episodes if e is not None]

if len(episodes) == 0:
    raise RuntimeError("No episodes found!")

print(f"Loaded {len(episodes)} episodes")

# -------------------------------------------------
# Split by Episode (no leakage)
# -------------------------------------------------
N = len(episodes)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)

train_eps = episodes[:n_train]
val_eps   = episodes[n_train:n_train+n_val]
test_eps  = episodes[n_train+n_val:]

print(f"Split -> Train:{len(train_eps)}  Val:{len(val_eps)}  Test:{len(test_eps)}")

# -------------------------------------------------
# Create Datasets
# -------------------------------------------------
train_ds = EpisodeWindowDataset(train_eps, cfg["model"]["seq_len"], cfg["model"]["future_steps"])
val_ds   = EpisodeWindowDataset(val_eps,   cfg["model"]["seq_len"], cfg["model"]["future_steps"])
test_ds  = EpisodeWindowDataset(test_eps,  cfg["model"]["seq_len"], cfg["model"]["future_steps"])

train_loader = DataLoader(
    train_ds,
    batch_size=cfg["train"]["batch_size"],
    shuffle=True,
    num_workers=cfg["train"]["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=cfg["train"]["batch_size"],
    shuffle=False,
    num_workers=cfg["train"]["num_workers"],
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=cfg["train"]["batch_size"],
    shuffle=False,
    num_workers=cfg["train"]["num_workers"],
    pin_memory=True
)

print(f"Windows -> Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")


# -------------------------------------------------
# Build Model
# -------------------------------------------------
model = ACT_RGB(**cfg["model"]).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg["train"]["lr"],
    weight_decay=cfg["train"]["weight_decay"],
    betas=(0.9, 0.95)
)


# -------------------------------------------------
# Warmup + Cosine Scheduler
# -------------------------------------------------
total_steps = cfg["train"]["epochs"] * len(train_loader)
warmup_steps = cfg["scheduler"]["warmup_steps"]
min_lr = cfg["scheduler"]["min_lr"]
base_lr = cfg["train"]["lr"]

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return max(min_lr / base_lr, cosine)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------
# One Epoch Function
# -------------------------------------------------
def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0

    for rgb, joint, goal_xyz, future in loader:
        rgb = rgb.to(device, non_blocking=True)
        joint = joint.to(device, non_blocking=True)
        goal_xyz = goal_xyz.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)

        pred = model(rgb, joint, goal_xyz)

        loss = act_loss(
            pred, future, joint, goal_xyz,
            cfg["loss"]["lambda_cont"],
            cfg["loss"]["lambda_goal"]
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
best_val = float("inf")
train_curve, val_curve = [], []
bad_epochs = 0

print("\n===== START TRAINING =====")

for epoch in range(cfg["train"]["epochs"]):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader, train=False)

    train_curve.append(train_loss)
    val_curve.append(val_loss)

    print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "outputs/models/best_model.pth")
        bad_epochs = 0
        print("   ✅ Saved new best model")
    else:
        bad_epochs += 1

    if bad_epochs >= cfg["train"]["patience"]:
        print("Early stopping triggered.")
        break


# -------------------------------------------------
# Save final model
# -------------------------------------------------
torch.save(model.state_dict(), "outputs/models/final_model.pth")
print("Saved final model.")


# -------------------------------------------------
# Plot Loss Curves
# -------------------------------------------------
plt.figure()
plt.plot(train_curve, label="Train")
plt.plot(val_curve, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/loss_curve.png", dpi=200)
print("Saved loss plot.")


# -------------------------------------------------
# Test Evaluation (FK Goal Error)
# -------------------------------------------------
print("\n===== RUNNING TEST EVALUATION =====")

model.load_state_dict(torch.load("outputs/models/best_model.pth"))
model.eval()

cart_errors = []

with torch.no_grad():
    for rgb, joint, goal_xyz, future in test_loader:
        rgb = rgb.to(device)
        joint = joint.to(device)
        goal_xyz = goal_xyz.to(device)

        pred = model(rgb, joint, goal_xyz)

        q_last = joint[:, -1, :]
        q_pred_last = q_last + pred[:, -1, :]
        xyz_pred = batch_fk(q_pred_last)

        err = torch.norm(xyz_pred - goal_xyz, dim=1)
        cart_errors.append(err.cpu())

cart_errors = torch.cat(cart_errors).numpy()

print(f"\nTest Cartesian Goal Error:")
print(f"Mean: {cart_errors.mean():.4f} m")
print(f"Std : {cart_errors.std():.4f} m")

print("\n===== DONE =====")
