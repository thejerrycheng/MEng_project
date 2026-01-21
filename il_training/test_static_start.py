import os
import argparse
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import torchvision.transforms as transforms
import mujoco 

from models.transformer_cvae import ACT_CVAE_Optimized

# --------------------------
# Configuration
# --------------------------
SEQ_LEN = 8
FUTURE_STEPS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# 1. Kinematics Helper
# --------------------------
class IRISKinematics:
    def __init__(self):
        # Configuration extracted from your XML
        self.link_configs = [
            {'pos': [0, 0, 0.2487],         'euler': [0, 0, 0],      'axis': [0, 0, 1]},
            {'pos': [0.0218, 0, 0.059],     'euler': [0, 90, 180],   'axis': [0, 0, 1]},
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0],      'axis': [0, 0, 1]},
            {'pos': [0.02, 0, 0],           'euler': [0, 90, 0],     'axis': [0, 0, 1]},
            {'pos': [0, 0, 0.315],          'euler': [0, -90, 0],    'axis': [0, 0, 1]},
            {'pos': [0.042824, 0, 0],       'euler': [0, 90, 180],   'axis': [0, 0, 1]},
            {'pos': [0, 0, 0],              'euler': [0, 0, 0],      'axis': [0, 0, 0]} 
        ]

    def get_local_transform(self, cfg, q_rad):
        T_pos = np.eye(4)
        T_pos[:3, 3] = cfg['pos']

        R_fixed = np.eye(3)
        if any(cfg['euler']):
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            R_fixed = mat.reshape(3, 3)
        
        T_rot_fixed = np.eye(4)
        T_rot_fixed[:3, :3] = R_fixed

        T_joint = np.eye(4)
        if np.any(cfg['axis']):
            quat_j = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat_j, np.array(cfg['axis']), q_rad)
            mat_j = np.zeros(9)
            mujoco.mju_quat2Mat(mat_j, quat_j)
            R_joint = mat_j.reshape(3, 3)
            T_joint[:3, :3] = R_joint

        return T_pos @ T_rot_fixed @ T_joint

    def forward_traj(self, joint_seq):
        N = len(joint_seq)
        xyz_seq = np.zeros((N, 3))
        
        for t in range(N):
            q_rad = joint_seq[t] 
            T_accumulated = np.eye(4)
            for i in range(6): 
                T_link = self.get_local_transform(self.link_configs[i], q_rad[i])
                T_accumulated = T_accumulated @ T_link
            T_ee = self.get_local_transform(self.link_configs[6], 0)
            T_accumulated = T_accumulated @ T_ee
            xyz_seq[t] = T_accumulated[:3, 3]
        return xyz_seq

# --------------------------
# 2. Helper Functions
# --------------------------
def load_model(checkpoint_path):
    print(f"Loading Model from: {checkpoint_path}")
    model = ACT_CVAE_Optimized(
        seq_len=SEQ_LEN,
        future_steps=FUTURE_STEPS,
        d_model=256,
        nhead=8,
        latent_dim=32
    ).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_first_clip_of_episodes(data_root, num_samples=5):
    all_clips = sorted(glob.glob(os.path.join(data_root, "*_clip_00000")))
    if len(all_clips) == 0:
        print(f"Warning: No start clips found. Picking random clips.")
        all_clips = sorted(glob.glob(os.path.join(data_root, "*_clip_*")))
    
    indices = np.random.choice(len(all_clips), min(len(all_clips), num_samples), replace=False)
    return [all_clips[i] for i in indices]

# --------------------------
# 3. Inference & Plotting (Modified for Static Start)
# --------------------------
def run_inference(model, clip_path, fk_solver):
    with open(os.path.join(clip_path, "robot", "data.json"), "r") as f:
        data = json.load(f)
    
    joint_seq_raw = np.array(data["joint_seq"]) # (8, 6)
    gt_fut_delta = np.array(data["fut_delta"])  # (15, 6)
    
    # --- STATIC JOINT HISTORY ---
    # Instead of loading 8 different joint states, we take the LAST one (current)
    # and repeat it 8 times. This simulates the robot sitting still at the start.
    current_q = joint_seq_raw[-1] 
    joint_seq_static = np.tile(current_q, (SEQ_LEN, 1)) # (8, 6)
    joint_seq_tensor = torch.tensor(joint_seq_static, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # --- STATIC IMAGE HISTORY ---
    # Instead of loading input_0000 -> input_0007, we take input_0007 (Current)
    # and stack it 8 times.
    current_img_path = os.path.join(clip_path, "rgb", f"input_{SEQ_LEN-1:04d}.png")
    current_img_raw = Image.open(current_img_path).convert("RGB")
    current_img_tensor = transform(current_img_raw).to(DEVICE)
    
    # Stack 8 identical images
    rgb_seq = torch.stack([current_img_tensor] * SEQ_LEN).unsqueeze(0) # (1, 8, 3, 224, 224)

    # Load Goal
    goal_path = os.path.join(clip_path, "rgb", "goal.png")
    if not os.path.exists(goal_path):
         goal_path = os.path.join(clip_path, "rgb", f"input_{SEQ_LEN-1:04d}.png")
    goal_img_raw = Image.open(goal_path).convert("RGB")
    goal_tensor = transform(goal_img_raw).unsqueeze(0).to(DEVICE)

    # Inference
    print(f"Running inference with STATIC history on {os.path.basename(clip_path)}...")
    with torch.no_grad():
        pred_delta, _ = model(rgb_seq, joint_seq_tensor, goal_tensor, target_actions=None)
    pred_delta = pred_delta.squeeze(0).cpu().numpy()

    # Reconstruct Absolute Angles & Paths
    pred_abs_q = current_q + pred_delta
    gt_abs_q = current_q + gt_fut_delta
    
    pred_abs_full = np.vstack([current_q, pred_abs_q])
    gt_abs_full = np.vstack([current_q, gt_abs_q])

    gt_path_3d = fk_solver.forward_traj(gt_abs_full)
    pred_path_3d = fk_solver.forward_traj(pred_abs_full)

    return {
        "pred_delta": pred_delta,
        "gt_delta": gt_fut_delta,
        "pred_path": pred_path_3d,
        "gt_path": gt_path_3d,
        "start_img": current_img_raw,
        "goal_img": goal_img_raw,
        "name": os.path.basename(clip_path) + "_STATIC"
    }

def visualize_interactive(res, save_dir):
    gt_path = res['gt_path']
    pred_path = res['pred_path']
    gt_delta = res['gt_delta']
    pred_delta = res['pred_delta']

    # Calculate Velocity (Euclidean distance between steps)
    # This tells us how fast the robot wants to move
    gt_vel = np.linalg.norm(np.diff(gt_path, axis=0), axis=1)
    pred_vel = np.linalg.norm(np.diff(pred_path, axis=0), axis=1)

    plt.ion() 
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Static Start Analysis: {res['name']}", fontsize=16)
    
    gs = fig.add_gridspec(4, 4) # Added 4th row for Velocity

    # --- ROW 0: Visuals ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(res['start_img'])
    ax1.set_title("Start (Static)")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(res['goal_img'])
    ax2.set_title("Goal")
    ax2.axis('off')

    # --- 3D Path ---
    ax_3d = fig.add_subplot(gs[0, 2], projection='3d')
    ax_3d.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'g-', linewidth=2, label='GT (Dynamic)', alpha=0.5)
    ax_3d.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2], 'r-o', linewidth=2, label='Pred (Static)', markersize=3)
    ax_3d.scatter(gt_path[0,0], gt_path[0,1], gt_path[0,2], c='k', s=50, label='Start')
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend()

    # --- Top Down ---
    ax_xy = fig.add_subplot(gs[0, 3])
    ax_xy.plot(gt_path[:, 0], gt_path[:, 1], 'g-', linewidth=2, label='GT', alpha=0.5)
    ax_xy.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='Pred')
    ax_xy.set_title("Top-Down View")
    ax_xy.axis('equal')

    # --- ROW 1: VELOCITY PROFILE (New!) ---
    ax_vel = fig.add_subplot(gs[1, :]) # Span entire row
    ax_vel.plot(gt_vel, 'g-', label='Ground Truth Velocity (Already Moving)', linewidth=2)
    ax_vel.plot(pred_vel, 'r-o', label='Prediction Velocity (From Stop)', linewidth=2)
    ax_vel.set_title("Velocity Profile (Step-to-Step Euclidean Distance)")
    ax_vel.set_ylabel("Speed (m/step)")
    ax_vel.set_xlabel("Time Step")
    ax_vel.grid(True, alpha=0.5)
    ax_vel.legend()

    # --- ROWS 2 & 3: Joint Deltas ---
    time = np.arange(len(gt_delta))
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(time, gt_delta[:, i], 'g-', alpha=0.5)
        ax.plot(time, pred_delta[:, i], 'r--')
        ax.set_title(f"Joint {i}")
        ax.grid(True, alpha=0.3)

    for i in range(3):
        ax = fig.add_subplot(gs[3, i])
        ax.plot(time, gt_delta[:, i+3], 'g-', alpha=0.5)
        ax.plot(time, pred_delta[:, i+3], 'r--')
        ax.set_title(f"Joint {i+3}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"static_vel_{res['name']}.png")
    plt.savefig(save_path)
    print(f"  -> Saved plot to {save_path}")
    
    plt.show() 
    input("Press Enter to see next...")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--data_root", type=str, required=True, help="Path to TEST clips folder")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of scenarios")
    args = parser.parse_args()

    save_dir = "offline_deployment_results"
    os.makedirs(save_dir, exist_ok=True)

    print("Initializing IRIS Kinematics...")
    fk_solver = IRISKinematics()

    model = load_model(args.checkpoint)
    clip_paths = get_first_clip_of_episodes(args.data_root, args.num_samples)

    print(f"\n--- Static Start Testing on {len(clip_paths)} Scenarios ---")
    
    for clip_path in clip_paths:
        try:
            results = run_inference(model, clip_path, fk_solver)
            visualize_interactive(results, save_dir)
        except Exception as e:
            print(f"Failed {clip_path}: {e}")

if __name__ == "__main__":
    main()