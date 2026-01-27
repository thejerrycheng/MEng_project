import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from datasets.iris_dataset import IRISClipDataset

# ---------------------
# Logging Setup
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

# ---------------------
# 1. CVAE MODEL MAPPING
# ---------------------
MODEL_MAPPING = {
    # RGB Only (Image -> Absolute)
    "cvae_rgb": "transformer_cvae_rgb_absolute.CVAE_RGB_Only_Absolute",
    
    # Visual Servoing (RGB + Goal -> Absolute)
    "cvae_visual": "transformer_cvae_rgb_goal_absolute.CVAE_RGB_Goal_Absolute",
    
    # Full Context (RGB + Joints + Goal -> Absolute)
    "cvae_full": "transformer_cvae_rgb_joint_goal_absolute.CVAE_RGB_Joints_Goal_Absolute",
    
    # Legacy
    "legacy_cvae": "transformer_cvae.ACT_CVAE_Optimized",
}

# ---------------------
# 2. LOSS MAPPING
# ---------------------
LOSS_MAPPING = {
    "loss_kl": "losses.loss_kl.ACTCVAELoss",
}

def load_class(path_str):
    try:
        if "transformer_" in path_str: 
            module_name, class_name = path_str.split(".")
            full_module_path = f"models.{module_name}"
            module = importlib.import_module(full_module_path)
            return getattr(module, class_name)
        else:
            module_path, class_name = path_str.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
    except Exception as e:
        logging.error(f"Error loading {path_str}: {e}")
        raise e

def save_plots(history, plots_dir, name):
    if len(history['train_loss']) == 0: return
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Total')
    plt.plot(history['val_loss'], label='Val Total')
    plt.title(f'Training Curve: {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"loss_{name}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--data_roots", type=str, nargs='+', required=True)
    parser.add_argument("--name", type=str, required=True)
    
    # --- CHANGED: Saves to Desktop/checkpoints by default ---
    parser.add_argument("--checkpoint_dir", type=str, 
                        default=os.path.join(os.path.expanduser("~"), "Desktop", "checkpoints"))
    
    # Model Selection
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--loss", type=str, default="loss_kl")

    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    
    # Model Architecture Params
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--smoothness_weight", type=float, default=0.01)
    
    args = parser.parse_args()

    # Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # We save plots inside the checkpoint dir to keep things organized
    plots_dir = os.path.join(args.checkpoint_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Saving models to: {args.checkpoint_dir}")

    # ---------------------
    # Data Loading
    # ---------------------
    logging.info("Loading Datasets...")
    train_ds = ConcatDataset([IRISClipDataset(os.path.join(r, "train")) for r in args.data_roots])
    
    if len(train_ds) == 0: raise ValueError("No training data found!")

    val_len = int(0.1 * len(train_ds))
    train_len = len(train_ds) - val_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info(f"Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    # ---------------------
    # Model & Loss Init
    # ---------------------
    ModelClass = load_class(MODEL_MAPPING[args.model])
    logging.info(f"Initializing Model: {ModelClass.__name__}")
    
    model = ModelClass(
        seq_len=args.seq_len, future_steps=args.future_steps,
        d_model=args.d_model, nhead=args.nhead, latent_dim=args.latent_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    LossClass = load_class(LOSS_MAPPING[args.loss])
    criterion = LossClass(beta=args.beta, smoothness_weight=args.smoothness_weight)

    # ---------------------
    # Forward Pass Helper
    # ---------------------
    def run_pass(batch, is_training=True):
        # Unpack from Dataset
        rgb, joints, goal_image, fut_delta = [b.to(device) for b in batch]
        
        # During training, we MUST pass target actions for CVAE Encoder
        targets = fut_delta 

        # --- A. RGB Only CVAE ---
        if args.model == "cvae_rgb":
            pred, (mu, logvar) = model(rgb, target_actions=targets)
        
        # --- B. Visual Goal CVAE ---
        elif args.model == "cvae_visual":
            pred, (mu, logvar) = model(rgb, goal_image, target_actions=targets)

        # --- C. Full Context CVAE ---
        elif args.model == "cvae_full" or args.model == "legacy_cvae":
            pred, (mu, logvar) = model(rgb, joints, goal_image, target_actions=targets)
        
        # Calculate Loss
        loss, loss_dict = criterion(pred, fut_delta, mu, logvar)
        return loss, loss_dict

    # ---------------------
    # Training Loop
    # ---------------------
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')

    logging.info(f"--- Starting Training ({args.name}) ---")

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        
        train_stats = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, loss_dict = run_pass(batch, is_training=True)
            loss.backward()
            optimizer.step()
            
            train_batches += 1
            train_stats["loss"] += loss.item()
            train_stats["mse"] += loss_dict.get("mse", 0.0)
            train_stats["kl"] += loss_dict.get("kl", 0.0)
            
            if batch_idx % args.log_interval == 0:
                logging.info(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                             f"Loss: {loss.item():.4f} | MSE: {loss_dict.get('mse',0):.4f} | KL: {loss_dict.get('kl',0):.4f}")

        # Validation
        model.eval()
        val_stats = {"loss": 0.0, "mse": 0.0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, loss_dict = run_pass(batch, is_training=False)
                val_batches += 1
                val_stats["loss"] += loss.item()
                val_stats["mse"] += loss_dict.get("mse", 0.0)

        # Averages
        avg_t_loss = train_stats["loss"] / train_batches
        avg_v_loss = val_stats["loss"] / max(val_batches, 1)
        avg_t_kl = train_stats["kl"] / train_batches
        
        epoch_dur = time.time() - start_time
        logging.info("-" * 60)
        logging.info(f"=== Epoch {epoch+1} Done ({epoch_dur:.1f}s) ===")
        logging.info(f"    [Train] Total: {avg_t_loss:.5f} | KL: {avg_t_kl:.5f}")
        logging.info(f"    [Val]   Total: {avg_v_loss:.5f}")
        logging.info("-" * 60)

        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)

        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_{args.name}.pth"))
            logging.info(f"    -> Saved Best Model to {args.checkpoint_dir}")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"latest_{args.name}.pth"))
        
        # Save plots
        pd.DataFrame(history).to_csv(os.path.join(plots_dir, f"history_{args.name}.csv"), index=False)
        save_plots(history, plots_dir, args.name)

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"final_{args.name}.pth"))
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()