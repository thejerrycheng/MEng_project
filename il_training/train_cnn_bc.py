import os
import argparse
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split

# --- Import Custom Modules ---
from models.cnn_model import VanillaBC_Visual_Absolute
from datasets.iris_dataset import IRISClipDataset
from losses.loss_genertic import AbsoluteMotionLoss

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO
)

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
    
    # Plot 2: MSE Only
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.title(f'MSE (Position): {name}')
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
    parser.add_argument("--checkpoint_dir", type=str, 
                        default=os.path.join(os.path.expanduser("~"), "Desktop", "checkpoints"))
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    
    # Model Params
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(args.checkpoint_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

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
    # Model Init
    # ---------------------
    logging.info("Initializing Vanilla CNN-BC Model...")
    
    model = VanillaBC_Visual_Absolute(
        seq_len=args.seq_len,
        future_steps=args.future_steps,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = AbsoluteMotionLoss(smoothness_weight=0.01)
    
    # Mixed Precision Scaler (Updated for PyTorch 2.0+)
    scaler = torch.amp.GradScaler('cuda')

    # ---------------------
    # Training Loop
    # ---------------------
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    best_loss = float('inf')

    logging.info(f"--- Starting Training ({args.name}) ---")

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        
        train_stats = {"loss": 0.0, "mse": 0.0, "smooth": 0.0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Unpack
            rgb, joints, goal_image, actions = [b.to(device) for b in batch]
            
            # Forward Pass with Mixed Precision (Updated)
            with torch.amp.autocast('cuda'):
                pred = model(rgb, joints, goal_image)
                loss, loss_dict = criterion(pred, actions)

            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Stats
            train_batches += 1
            train_stats["loss"] += loss.item()
            train_stats["mse"] += loss_dict.get("mse", 0.0)
            train_stats["smooth"] += loss_dict.get("smooth", 0.0)
            
            if batch_idx % args.log_interval == 0:
                logging.info(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                             f"Loss: {loss.item():.4f} | MSE: {loss_dict.get('mse', 0.0):.4f}")

        # Validation
        model.eval()
        val_stats = {"loss": 0.0, "mse": 0.0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb, joints, goal_image, actions = [b.to(device) for b in batch]
                
                # Validation doesn't strictly need autocast, but helps consistency
                with torch.amp.autocast('cuda'):
                    pred = model(rgb, joints, goal_image)
                    loss, loss_dict = criterion(pred, actions)
                
                val_batches += 1
                val_stats["loss"] += loss.item()
                val_stats["mse"] += loss_dict.get("mse", 0.0)

        # Averages
        avg_t_loss = train_stats["loss"] / train_batches
        avg_t_mse = train_stats["mse"] / train_batches
        avg_v_loss = val_stats["loss"] / max(val_batches, 1)
        avg_v_mse = val_stats["mse"] / max(val_batches, 1)
        
        epoch_dur = time.time() - start_time
        logging.info("-" * 60)
        logging.info(f"=== Epoch {epoch+1} Done ({epoch_dur:.1f}s) ===")
        logging.info(f"    [Train] Total: {avg_t_loss:.5f} | MSE: {avg_t_mse:.5f}")
        logging.info(f"    [Val]   Total: {avg_v_loss:.5f} | MSE: {avg_v_mse:.5f}")
        logging.info("-" * 60)
        
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['train_mse'].append(avg_t_mse)
        history['val_mse'].append(avg_v_mse)

        # Checkpointing
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