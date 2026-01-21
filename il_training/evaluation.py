import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg") # No GUI needed
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import Policy and Dataset
from policy import IRISPolicy
from datasets.iris_dataset import IRISClipDataset

def evaluate(args):
    # 1. Setup Directories
    viz_dir = os.path.join("eval_results", args.name)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 2. Load Policy
    policy = IRISPolicy(args.checkpoint, device='cuda')
    
    # 3. Load Test Dataset
    # We look for the 'test' subfolder inside your data root
    test_data_path = os.path.join(args.data_root, "test")
    if not os.path.exists(test_data_path):
        print(f"Warning: 'test' folder not found in {args.data_root}. Using 'val' instead.")
        test_data_path = os.path.join(args.data_root, "val")
        
    dataset = IRISClipDataset(test_data_path)
    # Batch size 1 makes it easier to plot individual trajectories
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(dataset)} clips from {test_data_path}")
    
    all_mse = []
    
    # 4. Evaluation Loop
    for i, (rgb_seq, joint_seq, goal_img, gt_fut_delta) in tqdm(enumerate(loader), total=len(loader)):
        
        # Move inputs to device handled by Policy, but here we just need raw data to pass to policy
        # The Dataset returns Tensors. We need to convert them back to what Policy expects
        # OR we can just modify policy to accept tensors. 
        # For simplicity, let's use the model inside policy directly since we have tensors.
        
        rgb_seq = rgb_seq.to(policy.device)
        joint_seq = joint_seq.to(policy.device)
        goal_img = goal_img.to(policy.device)
        
        with torch.no_grad():
            # Run Inference
            # target_actions=None ensures we test the 'Mean' (deterministic) capability
            pred_delta, _ = policy.model(rgb_seq, joint_seq, goal_img, target_actions=None)
            
        pred_delta = pred_delta.cpu().numpy().squeeze(0) # (15, 6)
        gt_delta = gt_fut_delta.numpy().squeeze(0)       # (15, 6)
        
        # Compute Error
        mse = np.mean((pred_delta - gt_delta) ** 2)
        all_mse.append(mse)
        
        # 5. Visualization (Save plot for every 50th clip)
        if i % 50 == 0:
            plot_trajectory(pred_delta, gt_delta, i, viz_dir)
            
    # 6. Summary Stats
    avg_mse = np.mean(all_mse)
    print("\n==================================")
    print(f"Evaluation Complete: {args.name}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Plots saved to: {viz_dir}")
    print("==================================")

def plot_trajectory(pred, gt, idx, save_dir):
    """
    Plots the 15-step trajectory for 3 key joints (Joint 0, 1, 2)
    """
    plt.figure(figsize=(12, 4))
    
    # We plot the first 3 joints (usually the main arm movement)
    for joint_idx in range(3):
        plt.subplot(1, 3, joint_idx+1)
        plt.plot(gt[:, joint_idx], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(pred[:, joint_idx], 'r--', label='Prediction', linewidth=2)
        plt.title(f"Joint {joint_idx} Delta")
        plt.xlabel("Time Step (Future)")
        plt.ylabel("Angle Change (rad)")
        if joint_idx == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"traj_clip_{idx}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing /test subfolder")
    parser.add_argument("--name", type=str, default="test_run", help="Name for output folder")
    args = parser.parse_args()
    
    evaluate(args)