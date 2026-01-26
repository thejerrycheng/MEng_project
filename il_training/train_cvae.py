import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# ---------------------------------------------------------
# Import Custom Modules
# ---------------------------------------------------------
# Ensure these files exist in your 'models', 'datasets', and 'losses' folders
from models.transformer_cvae import ACT_CVAE_Optimized
from datasets.iris_dataset import IRISClipDataset
from losses.loss_kl import ACTCVAELoss

# ---------------------
# Logging Setup
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

def save_plots(history, plots_dir, name):
    """
    Saves loss curves (Train vs Val)
    """
    if len(history['train_loss']) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['val_loss'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curve: {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"loss_{name}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    
    # Experiment Config
    parser.add_argument("--data_roots", type=str, nargs='+', required=True, 
                        help="List of root dirs containing 'train' and 'val' folders")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save models")
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Model Hyperparams
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    
    # CVAE Specific Params
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.01, help="KL Divergence weight")
    parser.add_argument("--smoothness_weight", type=float, default=0.01, help="Smoothness loss weight")
    
    args = parser.parse_args()

    # ---------------------
    # Setup Directories & Device
    # ---------------------
    models_dir = args.checkpoint_dir
    plots_dir = "plots"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------------------
    # Data Loading
    # ---------------------
    logging.info("Loading Datasets...")
    train_datasets = []
    val_datasets = []

    for root in args.data_roots:
        # Check for 'train' and 'val' subfolders
        train_path = os.path.join(root, "train")
        val_path = os.path.join(root, "val")

        if os.path.exists(train_path):
            try:
                ds = IRISClipDataset(train_path)
                train_datasets.append(ds)
            except Exception as e:
                logging.warning(f"Skipping {train_path}: {e}")
        
        if os.path.exists(val_path):
            try:
                ds = IRISClipDataset(val_path)
                val_datasets.append(ds)
            except Exception as e:
                pass

    if not train_datasets:
        raise RuntimeError("No valid training datasets found!")

    full_train_ds = ConcatDataset(train_datasets)
    full_val_ds = ConcatDataset(val_datasets) if val_datasets else None

    # Fallback if no validation set provided
    if full_val_ds is None or len(full_val_ds) == 0:
        logging.info("No validation data found. Splitting training data (90/10)...")
        total_len = len(full_train_ds)
        val_len = int(0.1 * total_len)
        train_len = total_len - val_len
        full_train_ds, full_val_ds = torch.utils.data.random_split(
            full_train_ds, [train_len, val_len]
        )

    logging.info(f"Train samples: {len(full_train_ds)} | Val samples: {len(full_val_ds)}")

    train_loader = DataLoader(full_train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(full_val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # ---------------------
    # Model & Optimization
    # ---------------------
    logging.info("Initializing ACT CVAE Model...")
    
    model = ACT_CVAE_Optimized(
        seq_len=args.seq_len,
        future_steps=args.future_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        latent_dim=args.latent_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Initialize CVAE Loss
    criterion = ACTCVAELoss(beta=args.beta, smoothness_weight=args.smoothness_weight)

    # ---------------------
    # Training Loop
    # ---------------------
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    logging.info(f"--- Starting Training ({args.epochs} epochs) ---")

    for epoch in range(args.epochs):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss_acc = 0
        batch_count = 0
        
        for batch_idx, (rgb, joints, goal_image, fut_delta) in enumerate(train_loader):
            # Move to device
            rgb = rgb.to(device)              # (B, Seq, 3, 224, 224)
            joints = joints.to(device)        # (B, Seq, 6)
            goal_image = goal_image.to(device)# (B, 3, 224, 224)
            fut_delta = fut_delta.to(device)  # (B, Future, 6) -> Ground Truth Actions

            optimizer.zero_grad()
            
            # Forward Pass (Pass fut_delta as 'target_actions' for the CVAE Encoder)
            pred_delta, (mu, logvar) = model(rgb, joints, goal_image, target_actions=fut_delta)
            
            # Loss Calculation
            loss, loss_dict = criterion(pred_delta, fut_delta, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            train_loss_acc += loss.item()
            batch_count += 1
            
            # Optional: Log batch progress
            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                             f"Loss: {loss.item():.4f} | MSE: {loss_dict['mse']:.4f} | KL: {loss_dict['kl']:.4f}")

        avg_train = train_loss_acc / max(batch_count, 1)

        # --- Validate ---
        model.eval()
        val_loss_acc = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for rgb, joints, goal_image, fut_delta in val_loader:
                rgb = rgb.to(device)
                joints = joints.to(device)
                goal_image = goal_image.to(device)
                fut_delta = fut_delta.to(device)

                # During validation, we still pass targets to calculate the KL Loss for metrics
                pred_delta, (mu, logvar) = model(rgb, joints, goal_image, target_actions=fut_delta)
                
                loss, _ = criterion(pred_delta, fut_delta, mu, logvar)
                
                val_loss_acc += loss.item()
                val_batch_count += 1

        avg_val = val_loss_acc / max(val_batch_count, 1)

        # --- Logging & Saving ---
        epoch_duration = time.time() - start_time
        logging.info(f"=== Epoch {epoch+1}/{args.epochs} Done ({epoch_duration:.1f}s) ===")
        logging.info(f"    Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        # Save Best Model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(models_dir, f"best_{args.name}.pth"))
            logging.info("    -> Saved Best Model")

        # Save Latest Model (Every Epoch)
        torch.save(model.state_dict(), os.path.join(models_dir, f"latest_{args.name}.pth"))

        # Save Plots and CSV
        save_plots(history, plots_dir, args.name)
        pd.DataFrame(history).to_csv(os.path.join(plots_dir, f"history_{args.name}.csv"), index=False)

    # Save Final Model
    torch.save(model.state_dict(), os.path.join(models_dir, f"final_{args.name}.pth"))
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()