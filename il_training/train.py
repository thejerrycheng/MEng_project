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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
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
    # --- 1. Original Models (Delta output) ---
    "transformer_cvae": "ACT_CVAE_Optimized",
    "transformer_cvae_resnet_freeze": "ACT_CVAE_Frozen",

    # --- 2. Deterministic Absolute Models ---
    "transformer_absolute": "Transformer_Absolute",
    "transformer_visual_absolute": "Transformer_Visual_Absolute",
    "transformer_rgb_only": "Transformer_RGB_Only_Absolute",

    # --- 3. CVAE Absolute Models (NEW) ---
    # Filename: cvae_absolute_full.py -> Class: CVAE_RGB_Joints_Goal_Absolute
    "cvae_abs_full": "CVAE_RGB_Joints_Goal_Absolute",
    
    # Filename: cvae_absolute_visual.py -> Class: CVAE_RGB_Goal_Absolute
    "cvae_abs_visual": "CVAE_RGB_Goal_Absolute",
    
    # Filename: cvae_absolute_rgb_only.py -> Class: CVAE_RGB_Only_Absolute
    "cvae_abs_rgb": "CVAE_RGB_Only_Absolute",
}

LOSS_MAPPING = {
    "loss": "ACTLoss",           # Original Delta Loss
    "loss_kl": "ACTCVAELoss",    # CVAE Loss (KL + MSE + Smooth)
    "mse": "AbsoluteMotionLoss", # Deterministic Smooth Loss
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
    if len(history['train_loss']) == 0: return
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
    
    # Data & Experiment
    parser.add_argument("--data_roots", type=str, nargs='+', required=True, help="List of root dirs")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--model", type=str, required=True, help=f"Options: {list(MODEL_MAPPING.keys())}")
    parser.add_argument("--loss", type=str, required=True, help=f"Options: {list(LOSS_MAPPING.keys())}")
    parser.add_argument("--checkpoint_dir", type=str, default="/media/jerry/SSD/checkpoints", help="Save dir")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
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
    # Data Loading
    # ---------------------
    train_datasets = []
    val_datasets = []

    for root in args.data_roots:
        train_path = os.path.join(root, "train")
        val_path = os.path.join(root, "val")
        
        if os.path.exists(train_path):
            try:
                ds = IRISClipDataset(train_path)
                train_datasets.append(ds)
            except Exception as e:
                logging.warning(f"Failed to load {train_path}: {e}")

        if os.path.exists(val_path):
            try:
                ds = IRISClipDataset(val_path)
                if len(ds) > 0: val_datasets.append(ds)
            except Exception: pass 

    if not train_datasets: raise RuntimeError("No training datasets loaded!")

    full_train_ds = ConcatDataset(train_datasets)
    full_val_ds = ConcatDataset(val_datasets) if val_datasets else None

    if full_val_ds is None or len(full_val_ds) == 0:
        logging.warning("No validation data found! Splitting...")
        total_len = len(full_train_ds)
        train_len = int(0.9 * total_len)
        val_len = total_len - train_len
        full_train_ds, full_val_ds = random_split(
            full_train_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )

    logging.info(f"Train Samples: {len(full_train_ds)} | Val Samples: {len(full_val_ds)}")

    train_loader = DataLoader(full_train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(full_val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # ---------------------
    # Model Init
    # ---------------------
    ModelClass = load_class_from_module("models", args.model, MODEL_MAPPING[args.model])
    class_name = ModelClass.__name__
    logging.info(f"Initializing Model: {class_name}")

    if "CVAE" in class_name:
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

    # ---------------------
    # Loss Init
    # ---------------------
    LossClass = load_class_from_module("losses", args.loss, LOSS_MAPPING[args.loss])
    if "CVAE" in LossClass.__name__:
        criterion = LossClass(beta=args.beta, smoothness_weight=args.smoothness_weight)
        logging.info("Using CVAE Loss (MSE + KL + Smooth)")
    else:
        criterion = LossClass(smoothness_weight=args.smoothness_weight)
        logging.info("Using Absolute Motion Loss (MSE + Smooth)")

    # ---------------------
    # Resume Logic
    # ---------------------
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    history_path = os.path.join(plots_dir, f"history_{args.name}.csv")
    if os.path.exists(history_path):
        try:
            df_hist = pd.read_csv(history_path)
            history['train_loss'] = df_hist['train_loss'].tolist()
            history['val_loss'] = df_hist['val_loss'].tolist()
            start_epoch = len(history['train_loss'])
            if len(history['val_loss']) > 0: best_val_loss = min(history['val_loss'])
            logging.info(f"Resuming at Epoch {start_epoch + 1}")
        except Exception: pass

    latest_path = os.path.join(models_dir, f"latest_{args.name}.pth")
    best_path = os.path.join(models_dir, f"best_{args.name}.pth")
    load_path = latest_path if os.path.exists(latest_path) else (best_path if os.path.exists(best_path) else None)
        
    if load_path:
        logging.info(f"Resuming weights: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        logging.info("Starting from scratch.")

    # ---------------------
    # Helper: Forward Pass
    # ---------------------
    def run_forward_pass(batch_data, is_training=True):
        rgb, joints, goal_image, target_actions = batch_data
        rgb, joints, goal_image, target_actions = rgb.to(device), joints.to(device), goal_image.to(device), target_actions.to(device)

        # A. CVAE Absolute Models (New)
        if "CVAE_RGB_Only" in class_name:
            # Inputs: RGB + Future Actions (training)
            targets = target_actions if is_training else None
            pred, (mu, logvar) = model(rgb, target_actions=targets)
            if not is_training and mu is None: 
                mu = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
                logvar = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
            loss, loss_dict = criterion(pred, target_actions, mu, logvar)
            return loss, loss_dict
            
        elif "CVAE_RGB_Goal" in class_name:
            # Inputs: RGB + Goal + Future Actions (training)
            targets = target_actions if is_training else None
            pred, (mu, logvar) = model(rgb, goal_image, target_actions=targets)
            if not is_training and mu is None: 
                mu = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
                logvar = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
            loss, loss_dict = criterion(pred, target_actions, mu, logvar)
            return loss, loss_dict

        elif "CVAE_RGB_Joints" in class_name: # Full CVAE
            targets = target_actions if is_training else None
            pred, (mu, logvar) = model(rgb, joints, goal_image, target_actions=targets)
            if not is_training and mu is None: 
                mu = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
                logvar = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
            loss, loss_dict = criterion(pred, target_actions, mu, logvar)
            return loss, loss_dict

        # B. Deterministic Models
        elif "RGB_Only" in class_name:
            pred = model(rgb)
            loss, loss_dict = criterion(pred, target_actions)
            return loss, loss_dict
            
        elif "Visual_Absolute" in class_name:
            pred = model(rgb, goal_image)
            loss, loss_dict = criterion(pred, target_actions)
            return loss, loss_dict

        elif "Transformer_Absolute" in class_name:
            pred = model(rgb, joints, goal_image)
            loss, loss_dict = criterion(pred, target_actions)
            return loss, loss_dict

        # C. Fallback (Legacy CVAE Delta)
        else:
            targets = target_actions if is_training else None
            pred, (mu, logvar) = model(rgb, joints, goal_image, target_actions=targets)
            if not is_training and mu is None: 
                mu = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
                logvar = torch.zeros((rgb.shape[0], args.latent_dim)).to(device)
            loss, loss_dict = criterion(pred, target_actions, mu, logvar)
            return loss, loss_dict

    # ---------------------
    # Training Loop
    # ---------------------
    logging.info(f"--- Starting Training ---")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_acc = 0
        batch_count = 0
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            loss, loss_dict = run_forward_pass(batch_data, is_training=True)
            loss.backward()
            optimizer.step()
            
            train_loss_acc += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                log_str = f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                if 'mse' in loss_dict: log_str += f" | MSE: {loss_dict['mse']:.4f}"
                if 'kl' in loss_dict: log_str += f" | KL: {loss_dict['kl']:.4f}"
                logging.info(log_str)

        avg_train = train_loss_acc / batch_count

        # Validation
        model.eval()
        val_loss_acc = 0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                loss, _ = run_forward_pass(batch_data, is_training=False)
                val_loss_acc += loss.item()
                val_batches += 1

        avg_val = val_loss_acc / val_batches
        epoch_dur = time.time() - start_time
        
        logging.info(f"=== Epoch {epoch+1} Done ({epoch_dur:.1f}s) | Train: {avg_train:.5f} | Val: {avg_val:.5f} ===")

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(plots_dir, f"history_{args.name}.csv"), index=False)
        save_plots(history, plots_dir, f"{args.name}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(models_dir, f"best_{args.name}.pth"))
            logging.info(f"    -> Saved Best Model")

        torch.save(model.state_dict(), os.path.join(models_dir, f"latest_{args.name}.pth"))

    torch.save(model.state_dict(), os.path.join(models_dir, f"final_{args.name}.pth"))
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()