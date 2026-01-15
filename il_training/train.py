import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import both models
from models.transformer_model import ACT_RGB
from models.cnn_model import VanillaBC
from datasets.iris_dataset import EpisodeWindowDataset
from losses.loss import ACTLoss

# ---------------------
# Logging Setup
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

def count_parameters(model):
    """Counts trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_plots(history, plots_dir, name):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curve: {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"loss_{name}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_data", help="Dir containing .pkl files")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    
    # Model Selection
    parser.add_argument("--model", type=str, required=True, choices=['transformer', 'cnn'], 
                        help="Choose architecture: 'transformer' (ACT) or 'cnn' (Vanilla BC)")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Common Model Params
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    
    # Transformer Specific
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    
    # CNN Specific
    parser.add_argument("--hidden_dim", type=int, default=1024)
    
    args = parser.parse_args()

    # Directories
    models_dir = "checkpoints"
    plots_dir = "plots"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------------------
    # Data Loading
    # ---------------------
    logging.info(f"Loading Datasets for Model: {args.model.upper()}...")
    train_pkl = os.path.join(args.data_dir, "train_episodes.pkl")
    val_pkl = os.path.join(args.data_dir, "val_episodes.pkl")

    train_ds = EpisodeWindowDataset(train_pkl, args.seq_len, args.future_steps)
    val_ds = EpisodeWindowDataset(val_pkl, args.seq_len, args.future_steps)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # ---------------------
    # Model Instantiation
    # ---------------------
    if args.model == 'transformer':
        logging.info("Initializing Transformer (ACT) Model...")
        model = ACT_RGB(
            seq_len=args.seq_len,
            future_steps=args.future_steps,
            d_model=args.d_model,
            nhead=args.nhead
        ).to(device)
        
    elif args.model == 'cnn':
        logging.info("Initializing Vanilla CNN (BC) Model...")
        model = VanillaBC(
            seq_len=args.seq_len,
            future_steps=args.future_steps,
            hidden_dim=args.hidden_dim,
            freeze_backbone=False 
        ).to(device)

    # Count Parameters
    total, trainable = count_parameters(model)
    logging.info(f"Model Parameters: {total:,} total | {trainable:,} trainable")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ACTLoss() 

    # ---------------------
    # Training Loop
    # ---------------------
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss_acc = 0
        
        for batch_idx, (rgb, joints, goal_delta, fut_delta, goal_joint_abs) in enumerate(train_loader):
            rgb = rgb.to(device)
            joints = joints.to(device)
            goal_delta = goal_delta.to(device)
            fut_delta = fut_delta.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass is identical for both models
            # (rgb, joint, goal) -> action_delta
            pred_delta = model(rgb, joints, goal_delta)
            
            loss, _ = criterion(pred_delta, fut_delta)
            
            loss.backward()
            optimizer.step()
            
            train_loss_acc += loss.item()

        avg_train = train_loss_acc / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for rgb, joints, goal_delta, fut_delta, goal_joint_abs in val_loader:
                rgb = rgb.to(device)
                joints = joints.to(device)
                goal_delta = goal_delta.to(device)
                fut_delta = fut_delta.to(device)

                pred_delta = model(rgb, joints, goal_delta)
                loss, _ = criterion(pred_delta, fut_delta)
                val_loss_acc += loss.item()

        avg_val = val_loss_acc / len(val_loader)

        # --- Log & Save ---
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_train:.5f} | Val: {avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save using the specific model name and experiment name
            torch.save(model.state_dict(), os.path.join(models_dir, f"best_{args.model}_{args.name}.pth"))
            logging.info("  -> Saved Best Model")

    logging.info("Training Complete.")
    
    # Save Final
    torch.save(model.state_dict(), os.path.join(models_dir, f"final_{args.model}_{args.name}.pth"))
    
    # Save CSV
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(plots_dir, f"history_{args.model}_{args.name}.csv"), index=False)
    
    # Plot
    save_plots(history, plots_dir, f"{args.model}_{args.name}")

if __name__ == "__main__":
    main()