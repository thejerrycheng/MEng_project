import os
import argparse
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from datasets.absolute_utils import IRISClipDataset

# Configure logging to show timestamp
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO
)

# ---------------------------------------------------------
# 1. DETERMINISTIC MODEL MAPPING
# ---------------------------------------------------------
MODEL_MAPPING = {
    # 1. Full Context (RGB + Joints + Goal)
    "det_full": "transformer_det_rgb_joint_goal_absolute.Transformer_Absolute",
    
    # 2. Visual Servoing (RGB + Goal)
    "det_visual": "transformer_det_rgb_goal_absolute.Transformer_Visual_Absolute",
    
    # 3. Behavior Cloning (RGB Only)
    "det_rgb": "transformer_det_rgb_absolute.Transformer_RGB_Only_Absolute",
}

# ---------------------------------------------------------
# 2. LOSS MAPPING
# ---------------------------------------------------------
LOSS_MAPPING = {
    "mse_smooth": "losses.loss_genertic.AbsoluteMotionLoss",
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
        logging.error(f"Failed to load {path_str}: {e}")
        raise e

def save_plots(history, plots_dir, name):
    if len(history['train_loss']) == 0: return
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Total')
    plt.plot(history['val_loss'], label='Val Total')
    plt.title(f'Total Loss: {name}')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: MSE Only (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.title(f'MSE (Position Error): {name}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"loss_{name}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    
    # Paths & Name
    parser.add_argument("--data_roots", type=str, nargs='+', required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--loss", type=str, default="mse_smooth")
    parser.add_argument("--checkpoint_dir", type=str, default="/media/jerry/SSD/checkpoints")
    
    # Training Params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--log_interval", type=int, default=10, help="Log progress every N batches")
    
    # Model Params
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    logging.info("Loading Datasets...")
    train_ds = ConcatDataset([IRISClipDataset(os.path.join(r, "train")) for r in args.data_roots])
    
    if len(train_ds) == 0:
        raise ValueError("No training data found!")

    val_len = int(0.1 * len(train_ds))
    train_len = len(train_ds) - val_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info(f"Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    # --- Model & Loss ---
    ModelClass = load_class(MODEL_MAPPING[args.model])
    
    model = ModelClass(
        seq_len=args.seq_len,
        future_steps=args.future_steps,
        d_model=args.d_model,
        nhead=args.nhead
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    LossClass = load_class(LOSS_MAPPING[args.loss])
    criterion = LossClass() 

    # --- Forward Pass Switch ---
    def run_pass(batch):
        # Unpack Data
        rgb, joints, goal, actions = [b.to(device) for b in batch]
        
        # 1. RGB ONLY
        if args.model == "det_rgb":
            pred = model(rgb)
        # 2. Visual Goal
        elif args.model == "det_visual":
            pred = model(rgb, goal)
        # 3. Full Context
        elif args.model == "det_full":
            pred = model(rgb, joints, goal)

        loss, loss_dict = criterion(pred, actions)
        return loss, loss_dict

    # --- Training Loop ---
    # Enhanced history tracking
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    best_loss = float('inf')

    logging.info(f"Starting training: {args.name}")

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        
        # Accumulators for this epoch
        train_stats = {"loss": 0.0, "mse": 0.0, "smooth": 0.0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, loss_dict = run_pass(batch)
            loss.backward()
            optimizer.step()
            
            # Accumulate stats
            train_batches += 1
            train_stats["loss"] += loss.item()
            train_stats["mse"] += loss_dict.get("mse", 0.0)
            train_stats["smooth"] += loss_dict.get("smooth", 0.0)
            
            # Detailed Logging within Epoch
            if batch_idx % args.log_interval == 0:
                logging.info(
                    f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"MSE: {loss_dict.get('mse', 0.0):.4f} | "
                    f"Smooth: {loss_dict.get('smooth', 0.0):.4f}"
                )
        
        # Validation Loop
        model.eval()
        val_stats = {"loss": 0.0, "mse": 0.0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, loss_dict = run_pass(batch)
                val_batches += 1
                val_stats["loss"] += loss.item()
                val_stats["mse"] += loss_dict.get("mse", 0.0)
        
        # Averages
        avg_t_loss = train_stats["loss"] / train_batches
        avg_t_mse = train_stats["mse"] / train_batches
        avg_t_smooth = train_stats["smooth"] / train_batches
        
        avg_v_loss = val_stats["loss"] / max(val_batches, 1)
        avg_v_mse = val_stats["mse"] / max(val_batches, 1)
        
        epoch_dur = time.time() - start_time
        
        # Epoch Summary Log
        logging.info("-" * 60)
        logging.info(f"=== Epoch {epoch+1} Completed ({epoch_dur:.1f}s) ===")
        logging.info(f"    [Train] Total: {avg_t_loss:.5f} | MSE: {avg_t_mse:.5f} | Smooth: {avg_t_smooth:.5f}")
        logging.info(f"    [Val]   Total: {avg_v_loss:.5f} | MSE: {avg_v_mse:.5f}")
        logging.info("-" * 60)
        
        # History Update
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['train_mse'].append(avg_t_mse)
        history['val_mse'].append(avg_v_mse)

        # Save Best
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            logging.info(f"*** New Best Model (Val Loss: {best_loss:.5f}) - Saving... ***")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_{args.name}.pth"))
        
        # Save Artifacts
        df = pd.DataFrame(history)
        df.to_csv(f"plots/history_{args.name}.csv", index=False)
        save_plots(history, "plots", args.name)
        
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"latest_{args.name}.pth"))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"final_{args.name}.pth"))

if __name__ == "__main__":
    main()