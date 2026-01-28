import os
import argparse
import time
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import mujoco
import glob  # <--- Added missing import

# Import your models
from models.transformer_cvae import ACT_CVAE_Optimized
# from models.cnn_model import VanillaBC  # Uncomment if you have the class definition

# --------------------------
# Configuration
# --------------------------
SEQ_LEN = 8
FUTURE_STEPS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing (Standard ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# 1. Kinematics (For Cartesian Calculations)
# --------------------------
class IRISKinematics:
    """Calculates EE Position (XYZ) and Orientation (Quat/Mat)"""
    def __init__(self):
        self.link_configs = [
            {'pos': [0, 0, 0.2487], 'euler': [0, 0, 0], 'axis': [0, 0, 1]},
            {'pos': [0.0218, 0, 0.059], 'euler': [0, 90, 180], 'axis': [0, 0, 1]},
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0], 'axis': [0, 0, 1]},
            {'pos': [0.02, 0, 0], 'euler': [0, 90, 0], 'axis': [0, 0, 1]},
            {'pos': [0, 0, 0.315], 'euler': [0, -90, 0], 'axis': [0, 0, 1]},
            {'pos': [0.042824, 0, 0], 'euler': [0, 90, 180], 'axis': [0, 0, 1]},
            {'pos': [0, 0, 0], 'euler': [0, 0, 0], 'axis': [0, 0, 0]} 
        ]

    def _get_transform(self, cfg, q):
        T_pos = np.eye(4); T_pos[:3, 3] = cfg['pos']
        
        quat = np.zeros(4); mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
        mat = np.zeros(9); mujoco.mju_quat2Mat(mat, quat)
        T_rot = np.eye(4); T_rot[:3, :3] = mat.reshape(3,3)
        
        T_joint = np.eye(4)
        if np.any(cfg['axis']):
            q_j = np.zeros(4); mujoco.mju_axisAngle2Quat(q_j, np.array(cfg['axis']), q)
            m_j = np.zeros(9); mujoco.mju_quat2Mat(m_j, q_j)
            T_joint[:3, :3] = m_j.reshape(3,3)
            
        return T_pos @ T_rot @ T_joint

    def forward(self, q):
        T = np.eye(4)
        for i in range(6): T = T @ self._get_transform(self.link_configs[i], q[i])
        T = T @ self._get_transform(self.link_configs[6], 0)
        return T[:3, 3], T[:3, :3] # Return XYZ and Rotation Matrix

# --------------------------
# 2. Metrics Calculator
# --------------------------
def calculate_smoothness(traj_xyz, dt=1.0/50.0):
    """
    Calculates E_smooth: Mean Squared Jerk (3rd derivative).
    traj_xyz: (T, 3) numpy array
    """
    if len(traj_xyz) < 4: return 0.0
    
    # Velocity (1st derivative)
    vel = np.diff(traj_xyz, axis=0) / dt
    # Acceleration (2nd derivative)
    acc = np.diff(vel, axis=0) / dt
    # Jerk (3rd derivative)
    jerk = np.diff(acc, axis=0) / dt
    
    # E_smooth = 1/T * integral(||jerk||^2)
    # We approximate integral with sum * dt
    jerk_norm_sq = np.sum(jerk**2, axis=1)
    e_smooth = np.mean(jerk_norm_sq) 
    
    return e_smooth

def calculate_orientation_error(R_pred, R_gt):
    """
    Approximates framing accuracy using orientation error (Geodesic distance on SO3).
    Returns error in degrees.
    """
    # R_diff = R_pred * R_gt^T
    R_diff = R_pred @ R_gt.T
    tr = np.trace(R_diff)
    theta = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    return np.degrees(theta)

def load_policy(checkpoint_path):
    # NOTE: Since you don't have the CNN trained yet, we assume all checkpoints 
    # passed here are Transformer CVAE. If you mix models later, you need logic here.
    model = ACT_CVAE_Optimized(seq_len=SEQ_LEN, future_steps=FUTURE_STEPS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        print("Tip: If trying to load a CNN, you need to change the model class in load_policy()")
        raise e
    model.eval()
    return model

# --------------------------
# 3. Main Evaluation Loop
# --------------------------
def evaluate_model(model_name, checkpoint_path, test_data_root):
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {model_name}: Checkpoint not found at {checkpoint_path}")
        return None

    fk = IRISKinematics()
    model = load_policy(checkpoint_path)
    
    # Get test clips
    clip_folders = sorted(glob.glob(os.path.join(test_data_root, "*_clip_00000")))
    if not clip_folders:
        # Fallback to random clips if "start" clips don't exist
        clip_folders = sorted(glob.glob(os.path.join(test_data_root, "*_clip_*")))[:50]

    metrics = {
        'goal_err_cm': [],
        'smoothness': [],
        'frame_err_deg': [],
        'inference_ms': []
    }

    print(f"\nEvaluating {model_name} on {len(clip_folders)} trials...")
    
    for clip in tqdm(clip_folders):
        try:
            # --- Load Data ---
            with open(os.path.join(clip, "robot", "data.json")) as f:
                data = json.load(f)
                
            joint_seq = np.array(data['joint_seq'])
            gt_fut_delta = np.array(data['fut_delta'])
            current_q = joint_seq[-1]
            
            # Prepare Inputs
            rgb_stack = []
            for i in range(SEQ_LEN):
                img = Image.open(os.path.join(clip, "rgb", f"input_{i:04d}.png")).convert("RGB")
                rgb_stack.append(transform(img).to(DEVICE))
            
            rgb_in = torch.stack(rgb_stack).unsqueeze(0)
            joint_in = torch.tensor(joint_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            goal_img = Image.open(os.path.join(clip, "rgb", "goal.png")).convert("RGB")
            goal_in = transform(goal_img).unsqueeze(0).to(DEVICE)

            # --- Inference & Timer ---
            start_t = time.time()
            with torch.no_grad():
                # CVAE: target_actions=None -> Deterministic Inference
                pred_delta, _ = model(rgb_in, joint_in, goal_in, target_actions=None)
            
            if DEVICE == "cuda": torch.cuda.synchronize()
            metrics['inference_ms'].append((time.time() - start_t) * 1000)

            # --- Reconstruct Trajectories ---
            pred_delta = pred_delta.squeeze(0).cpu().numpy()
            
            # Integrate to get absolute angles
            pred_abs = current_q + pred_delta
            gt_abs = current_q + gt_fut_delta
            
            # --- Calculate Physics Metrics ---
            # 1. Goal Accuracy (Last Step Euclidean Dist)
            pos_pred_final, rot_pred_final = fk.forward(pred_abs[-1])
            pos_gt_final, rot_gt_final = fk.forward(gt_abs[-1])
            
            # Convert to cm
            e_goal = np.linalg.norm(pos_pred_final - pos_gt_final) * 100 
            metrics['goal_err_cm'].append(e_goal)
            
            # 2. Framing Accuracy (Orientation Error of Goal)
            e_frame = calculate_orientation_error(rot_pred_final, rot_gt_final)
            metrics['frame_err_deg'].append(e_frame)
            
            # 3. Smoothness (Jerk) - Calculate on FULL trajectory
            # Generate full XYZ path for smoothness calculation
            traj_xyz = []
            for q in pred_abs:
                pos, _ = fk.forward(q)
                traj_xyz.append(pos)
            traj_xyz = np.array(traj_xyz)
            
            e_smooth = calculate_smoothness(traj_xyz)
            metrics['smoothness'].append(e_smooth)
            
        except Exception as e:
            # Skip bad clips gracefully
            continue

    if len(metrics['goal_err_cm']) == 0:
        return None

    # --- Aggregate ---
    results = {k: f"{np.mean(v):.2f} ± {np.std(v):.2f}" for k, v in metrics.items()}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True, help="Path to /test folder")
    # You can pass multiple checkpoints to compare!
    parser.add_argument("--models", nargs='+', help="List of checkpoints (e.g. best_vanilla.pth best_cvae.pth)")
    parser.add_argument("--names", nargs='+', help="List of names for the table (e.g. 'Vanilla BC' 'Ours (CVAE)')")
    args = parser.parse_args()

    if not args.models:
        print("Please provide model checkpoints via --models")
        return

    all_results = []
    
    for name, ckpt in zip(args.names, args.models):
        res = evaluate_model(name, ckpt, args.test_data)
        if res is not None:
            res['Method'] = name
            all_results.append(res)
        else:
            print(f"Skipping {name} due to errors or missing file.")

    if not all_results:
        print("No results generated.")
        return

    # --- Generate LaTeX Table ---
    df = pd.DataFrame(all_results)
    # Reorder cols
    cols = ['Method', 'goal_err_cm', 'smoothness', 'frame_err_deg', 'inference_ms']
    df = df[cols]
    
    print("\n" + "="*50)
    print("LATEX TABLE GENERATION")
    print("="*50 + "\n")
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & $E_{\text{goal}}$ (cm) $\downarrow$ & $E_{\text{smooth}}$ (Jerk) $\downarrow$ & $E_{\text{frame}}$ (deg) $\downarrow$ & Time (ms) $\downarrow$ \\")
    print(r"\midrule")
    
    for _, row in df.iterrows():
        print(f"{row['Method']} & {row['goal_err_cm']} & {row['smoothness']} & {row['frame_err_deg']} & {row['inference_ms']} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Quantitative comparison on cinematic shots.}")
    print(r"\label{tab:results}")
    print(r"\end{table}")

    # Save CSV
    df.to_csv("experiment_results.csv", index=False)
    print("\nSaved raw results to experiment_results.csv")

if __name__ == "__main__":
    main()