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
import glob
import re

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split

# Import the Dataset
from datasets.iris_dataset import IRISClipDataset

# ---------------------
# Logging Setup
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# ---------------------
# Dynamic Loading Helpers
# ---------------------
MODEL_MAPPING = {
    "transformer_model": "ACT_RGB",
    "cnn_model": "VanillaBC",
    "transformer_cvae": "ACT_CVAE_Optimized",
}

LOSS_MAPPING = {
    "loss": "ACTLoss",
    "loss_kl": "ACTCVAELoss",
}

def load_class_from_module(module_type, module_name, class_name):
    try:
        full_module_path = f"{module_type}.{module_name}"
        module = importlib.import_module(full_module_path)
        return getattr(module, class_name)
    except Exception as e:
        logging.error(f"Error loading {class_name} from {module_name}: {e}")
        raise e

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
    logging.info(f"Saved plot to {plots_dir}/loss_{name}.png")

def main():
    parser = argparse.ArgumentParser()
    
    # Data & Experiment
    parser.add_argument("--data_roots", type=str, nargs='+', required=True, 
                        help="List of root dirs (e.g. /path/to/data1 /path/to/data2)")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--model", type=str, required=True, help=f"Options: {list(MODEL_MAPPING.keys())}")
    parser.add_argument("--loss", type=str, required=True, help=f"Options: {list(LOSS_MAPPING.keys())}")
    parser.add_argument("--checkpoint_dir", type=str, default="/media/jerry/SSD/checkpoints", 
                        help="Directory to save/load model checkpoints")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200) # Ensure this matches your target total epochs
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Model Params
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    
    # Architecture Params
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--smoothness_weight", type=float, default=0.01)
    
    args = parser.parse_args()

    # Directories
    models_dir = args.checkpoint_dir 
    plots_dir = "plots"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------------------
    # Multi-Source Data Loading
    # ---------------------
    logging.info("--- Data Loading Phase ---")
    train_datasets = []
    val_datasets = []

    for root in args.data_roots:
        train_path = os.path.join(root, "train")
        val_path = os.path.join(root, "val")
        
        logging.info(f"Scanning root: {root}")

        if os.path.exists(train_path):
            try:
                ds = IRISClipDataset(train_path)
                train_datasets.append(ds)
                logging.info(f"  [+] Loaded {len(ds)} train clips from {train_path}")
            except Exception as e:
                logging.warning(f"  [-] Failed to load {train_path}: {e}")

        if os.path.exists(val_path):
            try:
                ds = IRISClipDataset(val_path)
                if len(ds) > 0:
                    val_datasets.append(ds)
                    logging.info(f"  [+] Loaded {len(ds)} val clips from {val_path}")
            except Exception:
                pass 

    if not train_datasets:
        raise RuntimeError("No training datasets loaded!")

    full_train_ds = ConcatDataset(train_datasets)
    full_val_ds = ConcatDataset(val_datasets) if val_datasets else None

    if full_val_ds is None or len(full_val_ds) == 0:
        logging.warning("⚠️  No validation data found! Performing split...")
        total_len = len(full_train_ds)
        train_len = int(0.9 * total_len)
        val_len = total_len - train_len
        full_train_ds, full_val_ds = random_split(
            full_train_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )

    logging.info(f"TOTAL Train Samples: {len(full_train_ds)}")
    logging.info(f"TOTAL Val Samples:   {len(full_val_ds)}")

    train_loader = DataLoader(full_train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(full_val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # ---------------------
    # Model & Loss Init
    # ---------------------
    logging.info("--- Model Initialization ---")
    
    ModelClass = load_class_from_module("models", args.model, MODEL_MAPPING[args.model])
    
    if "CVAE" in ModelClass.__name__:
        model = ModelClass(
            seq_len=args.seq_len, future_steps=args.future_steps,
            d_model=args.d_model, nhead=args.nhead, latent_dim=args.latent_dim
        ).to(device)
    else:
        model = ModelClass(
            seq_len=args.seq_len, future_steps=args.future_steps,
            d_model=args.d_model, nhead=args.nhead
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    LossClass = load_class_from_module("losses", args.loss, LOSS_MAPPING[args.loss])
    if "CVAE" in LossClass.__name__:
        criterion = LossClass(beta=args.beta, smoothness_weight=args.smoothness_weight)
    else:
        criterion = LossClass()

    # ---------------------
    # RESUME LOGIC
    # ---------------------
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    # Load history if exists
    history_path = os.path.join(plots_dir, f"history_{args.name}.csv")
    if os.path.exists(history_path):
        try:
            df_hist = pd.read_csv(history_path)
            history['train_loss'] = df_hist['train_loss'].tolist()
            history['val_loss'] = df_hist['val_loss'].tolist()
            start_epoch = len(history['train_loss'])
            best_val_loss = min(history['val_loss']) if len(history['val_loss']) > 0 else float('inf')
            logging.info(f"Loaded history. Resuming from Epoch {start_epoch}")
        except Exception as e:
            logging.warning(f"Could not load history file: {e}")

    # Load weights
    # We look for 'best_*.pth' to get the weights, but if you have a 'latest' or 'checkpoint' file that saves state, use that.
    # Since the previous script only saved 'best' and 'final', we have to load 'best'.
    # WARNING: Loading 'best' resets the model to the best epoch, not necessarily the *last* epoch run.
    # Ideally, training scripts should save a 'latest.pth'. 
    # Here we will try to load 'best_{name}.pth' as the starting point.
    
    checkpoint_path = os.path.join(models_dir, f"best_{args.name}.pth")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        # Note: If you didn't save optimizer state, we restart optimizer. 
        # Ideally, save {'model': ..., 'optimizer': ..., 'epoch': ...}
        logging.info("Checkpoint loaded successfully.")
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch (unless this was unintended).")
        start_epoch = 0 # Reset if no weights found

    # ---------------------
    # Training Loop
    # ---------------------
    logging.info(f"--- Resuming Training from Epoch {start_epoch+1} ---")
    
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_acc = 0
        batch_count = 0
        
        for batch_idx, (rgb, joints, goal_image, fut_delta) in enumerate(train_loader):
            rgb, joints, goal_image, fut_delta = rgb.to(device), joints.to(device), goal_image.to(device), fut_delta.to(device)
            
            optimizer.zero_grad()
            
            if "CVAE" in ModelClass.__name__:
                output = model(rgb, joints, goal_image, target_actions=fut_delta)
            else:
                output = model(rgb, joints, goal_image)
            
            if isinstance(output, tuple):
                pred, (mu, logvar) = output
                loss, loss_dict = criterion(pred, fut_delta, mu, logvar)
            else:
                pred = output
                loss, loss_dict = criterion(pred, fut_delta)
            
            loss.backward()
            optimizer.step()
            
            train_loss_acc += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                log_str = f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                if isinstance(loss_dict, dict):
                    if 'mse' in loss_dict: log_str += f" | MSE: {loss_dict['mse']:.4f}"
                    if 'kl' in loss_dict: log_str += f" | KL: {loss_dict['kl']:.4f}"
                logging.info(log_str)

        avg_train = train_loss_acc / batch_count

        # Validation
        model.eval()
        val_loss_acc = 0
        val_batches = 0
        with torch.no_grad():
            for rgb, joints, goal_image, fut_delta in val_loader:
                rgb, joints, goal_image, fut_delta = rgb.to(device), joints.to(device), goal_image.to(device), fut_delta.to(device)

                if "CVAE" in ModelClass.__name__:
                    output = model(rgb, joints, goal_image, target_actions=None)
                else:
                    output = model(rgb, joints, goal_image)

                if isinstance(output, tuple):
                    pred, (mu, logvar) = output
                    if mu is None: mu = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
                    if logvar is None: logvar = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
                    loss, _ = criterion(pred, fut_delta, mu, logvar)
                else:
                    pred = output
                    loss, _ = criterion(pred, fut_delta)
                
                val_loss_acc += loss.item()
                val_batches += 1

        avg_val = val_loss_acc / val_batches

        epoch_dur = time.time() - start_time
        start_time = time.time() 
        
        logging.info(f"=== Epoch {epoch+1}/{args.epochs} Completed in {epoch_dur:.1f}s ===")
        logging.info(f"    Train Loss: {avg_train:.5f}")
        logging.info(f"    Val Loss:   {avg_val:.5f}")

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(models_dir, f"best_{args.name}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"    -> New Best Model Saved!")
        
        # Save History every epoch just in case
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(plots_dir, f"history_{args.name}.csv"), index=False)

    torch.save(model.state_dict(), os.path.join(models_dir, f"final_{args.name}.pth"))
    save_plots(history, plots_dir, f"{args.name}")
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()