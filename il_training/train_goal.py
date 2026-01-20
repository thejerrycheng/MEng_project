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

# Import Revised Models
from models.transformer_model import ACT_GoalImage
from models.cnn_model import VanillaBC_GoalImage
from datasets.iris_dataset import EpisodeWindowDataset
from losses.loss import ACTLoss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['transformer', 'cnn'])
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model Params
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=15)
    
    args = parser.parse_args()

    models_dir = "checkpoints"
    plots_dir = "plots"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------------------
    # Data Loading
    # ---------------------
    logging.info("Loading Datasets...")
    train_pkl = os.path.join(args.data_dir, "train_episodes.pkl")
    val_pkl = os.path.join(args.data_dir, "val_episodes.pkl")

    # NOTE: Ensure your EpisodeWindowDataset now yields 'goal_image' instead of 'goal_delta'
    train_ds = EpisodeWindowDataset(train_pkl, args.seq_len, args.future_steps)
    val_ds = EpisodeWindowDataset(val_pkl, args.seq_len, args.future_steps)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # ---------------------
    # Model Setup
    # ---------------------
    if args.model == 'transformer':
        logging.info("Initializing ACT (Goal Image)...")
        model = ACT_GoalImage(
            seq_len=args.seq_len,
            future_steps=args.future_steps
        ).to(device)
        
    elif args.model == 'cnn':
        logging.info("Initializing Vanilla BC (Goal Image)...")
        model = VanillaBC_GoalImage(
            seq_len=args.seq_len,
            future_steps=args.future_steps
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ACTLoss() 

    # ---------------------
    # Training Loop
    # ---------------------
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        model.train()
        train_loss_acc = 0
        
        # NOTE: Updated unpacking to expect 'goal_image'
        for batch_idx, (rgb, joints, goal_image, fut_delta, _) in enumerate(train_loader):
            rgb = rgb.to(device)
            joints = joints.to(device)
            goal_image = goal_image.to(device) # (B, 3, H, W)
            fut_delta = fut_delta.to(device)
            
            optimizer.zero_grad()
            
            # Forward: rgb + joint + goal_image -> action
            pred_delta = model(rgb, joints, goal_image)
            
            loss, _ = criterion(pred_delta, fut_delta)
            
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        avg_train = train_loss_acc / len(train_loader)

        # Validation
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for rgb, joints, goal_image, fut_delta, _ in val_loader:
                rgb = rgb.to(device)
                joints = joints.to(device)
                goal_image = goal_image.to(device)
                fut_delta = fut_delta.to(device)

                pred_delta = model(rgb, joints, goal_image)
                loss, _ = criterion(pred_delta, fut_delta)
                val_loss_acc += loss.item()

        avg_val = val_loss_acc / len(val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_train:.5f} | Val: {avg_val:.5f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(models_dir, f"ckpt_{args.model}_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(models_dir, f"final_{args.model}_{args.name}.pth"))
    
    # Save CSV
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(plots_dir, f"history_{args.model}_{args.name}.csv"), index=False)
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()