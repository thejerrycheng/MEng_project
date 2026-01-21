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
    
    if len(all_clips) == 0:
        raise FileNotFoundError(f"No clips found in {data_root}")
        
    indices = np.random.choice(len(all_clips), min(len(all_clips), num_samples), replace=False)
    return [all_clips[i] for i in indices]

# --------------------------
# 3. Inference & Plotting
# --------------------------
def run_inference(model, clip_path, fk_solver):
    # Load JSON
    with open(os.path.join(clip_path, "robot", "data.json"), "r") as f:
        data = json.load(f)
    
    joint_seq_raw = np.array(data["joint_seq"]) # (8, 6)
    gt_fut_delta = np.array(data["fut_delta"])  # (15, 6)
    current_q = joint_seq_raw[-1] # The 'start' configuration

    # Load Images
    rgb_tensors = []
    first_img_raw = None
    for i in range(SEQ_LEN):
        img_path = os.path.join(clip_path, "rgb", f"input_{i:04d}.png")
        img = Image.open(img_path).convert("RGB")
        if i == SEQ_LEN - 1: first_img_raw = img
        rgb_tensors.append(transform(img).to(DEVICE))
    rgb_seq = torch.stack(rgb_tensors).unsqueeze(0)

    # Load Goal
    goal_path = os.path.join(clip_path, "rgb", "goal.png")
    if not os.path.exists(goal_path):
         goal_path = os.path.join(clip_path, "rgb", f"input_{SEQ_LEN-1:04d}.png")
    goal_img_raw = Image.open(goal_path).convert("RGB")
    goal_tensor = transform(goal_img_raw).unsqueeze(0).to(DEVICE)
    
    joint_seq_tensor = torch.tensor(joint_seq_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred_delta, _ = model(rgb_seq, joint_seq_tensor, goal_tensor, target_actions=None)
    pred_delta = pred_delta.squeeze(0).cpu().numpy()

    # Reconstruct Absolute Angles
    pred_abs_q = current_q + pred_delta
    gt_abs_q = current_q + gt_fut_delta
    
    pred_abs_full = np.vstack([current_q, pred_abs_q])
    gt_abs_full = np.vstack([current_q, gt_abs_q])

    # Forward Kinematics
    gt_path_3d = fk_solver.forward_traj(gt_abs_full)
    pred_path_3d = fk_solver.forward_traj(pred_abs_full)

    return {
        "pred_delta": pred_delta,
        "gt_delta": gt_fut_delta,
        "pred_path": pred_path_3d,
        "gt_path": gt_path_3d,
        "start_img": first_img_raw,
        "goal_img": goal_img_raw,
        "name": os.path.basename(clip_path)
    }

def visualize_interactive(res, save_dir):
    gt_path = res['gt_path']
    pred_path = res['pred_path']
    gt_delta = res['gt_delta']
    pred_delta = res['pred_delta']

    plt.ion() # Interactive mode
    
    # Increase figure width to accommodate the extra plot
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Analysis: {res['name']}", fontsize=16)
    
    # 3 Rows, 4 Columns
    gs = fig.add_gridspec(3, 4)

    # --- ROW 0: Visuals & Trajectories ---
    
    # 1. Start View
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(res['start_img'])
    ax1.set_title("Start")
    ax1.axis('off')

    # 2. Goal View
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(res['goal_img'])
    ax2.set_title("Goal")
    ax2.axis('off')

    # 3. 3D Path (Constrained to one cell)
    ax_3d = fig.add_subplot(gs[0, 2], projection='3d')
    
    
    ax_3d.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'g-', linewidth=2, label='GT', alpha=0.7)
    ax_3d.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2], 'r-o', linewidth=2, label='Pred', markersize=3)
    ax_3d.scatter(gt_path[0,0], gt_path[0,1], gt_path[0,2], c='k', s=50, label='Start')
    
    # Force Equal Aspect Ratio for 3D
    all_x = np.concatenate([gt_path[:, 0], pred_path[:, 0]])
    all_y = np.concatenate([gt_path[:, 1], pred_path[:, 1]])
    all_z = np.concatenate([gt_path[:, 2], pred_path[:, 2]])
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (all_x.max()+all_x.min())*0.5, (all_y.max()+all_y.min())*0.5, (all_z.max()+all_z.min())*0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_3d.set_title("3D Trajectory")

    # 4. Top-Down (X-Y) Projection
    ax_xy = fig.add_subplot(gs[0, 3])
    ax_xy.plot(gt_path[:, 0], gt_path[:, 1], 'g-', linewidth=2, label='GT')
    ax_xy.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='Pred')
    ax_xy.scatter(gt_path[0,0], gt_path[0,1], c='k', s=50, label='Start')
    ax_xy.set_title("Top-Down View (X-Y)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal') # Ensure circles look like circles
    ax_xy.legend(loc='best', fontsize='small')

    # --- ROWS 1 & 2: Joint Deltas ---
    # Plotting first 3 joints
    time = np.arange(len(gt_delta))
    for i in range(3):
        ax = fig.add_subplot(gs[1, i]) # Row 1, Col 0,1,2
        ax.plot(time, gt_delta[:, i], 'g-', label='GT')
        ax.plot(time, pred_delta[:, i], 'r--', label='Pred')
        ax.set_title(f"Joint {i} Delta")
        ax.grid(True, alpha=0.3)

    # Plotting next 3 joints
    for i in range(3):
        ax = fig.add_subplot(gs[2, i]) # Row 2, Col 0,1,2
        ax.plot(time, gt_delta[:, i+3], 'g-')
        ax.plot(time, pred_delta[:, i+3], 'r--')
        ax.set_title(f"Joint {i+3} Delta")
        ax.grid(True, alpha=0.3)

    # Place Legend in the empty bottom-right slot
    ax_leg = fig.add_subplot(gs[2, 3])
    ax_leg.axis('off')
    ax_leg.text(0.1, 0.5, "Green = Ground Truth\nRed = Prediction\nBlack = Start", fontsize=12)

    plt.tight_layout()
    
    # Save BEFORE showing (interactive backends sometimes clear the plot)
    save_path = os.path.join(save_dir, f"eval_{res['name']}.png")
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

    # Output Dir
    save_dir = "offline_deployment_results"
    os.makedirs(save_dir, exist_ok=True)

    print("Initializing IRIS Kinematics...")
    fk_solver = IRISKinematics()

    model = load_model(args.checkpoint)
    clip_paths = get_first_clip_of_episodes(args.data_root, args.num_samples)

    print(f"\n--- Testing on {len(clip_paths)} Scenarios ---")
    print("Graphs will pop up. Interact with them, then close or press Enter to continue.")
    
    for clip_path in clip_paths:
        try:
            results = run_inference(model, clip_path, fk_solver)
            visualize_interactive(results, save_dir)
        except Exception as e:
            print(f"Failed {clip_path}: {e}")

if __name__ == "__main__":
    main()