import os
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging
import sys
import csv
import glob
from ultralytics import YOLO
from tqdm import tqdm
from colorama import Fore, Style, init
import mujoco

# Initialize Colorama
init(autoreset=True)

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Thresholds ---
THRESH_SUCCESS = 0.85      # > 0.85 = SUCCESS
THRESH_SEMI = 0.80         # > 0.80 = SEMI-SUCCESS

# ---------------------------------------------------------
# 2. Kinematics Engine (MuJoCo FK)
# ---------------------------------------------------------
class MujocoFKSolver:
    """Computes Cartesian EE Path from Joint Angles using MuJoCo"""
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            # Try default path
            default_path = os.path.expanduser("~/Desktop/MEng_project/mujoco_sim/assets/iris.xml")
            if os.path.exists(default_path):
                xml_path = default_path
            else:
                raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")

        # Find End-Effector Site/Body
        # Priority: 'ee_site' -> 'ee_mount' -> Last Body
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.is_site = True
        
        if self.ee_id == -1:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
            self.is_site = False
        if self.ee_id == -1:
            self.ee_id = self.model.nbody - 1
            self.is_site = False

    def get_ee_path(self, joint_trajectory):
        """
        Input: (N, 6) Joint Angles
        Output: (N, 3) Cartesian XYZ Coordinates
        """
        n_frames = len(joint_trajectory)
        ee_path = np.zeros((n_frames, 3))
        
        # Determine number of joints in the model vs data
        n_joints_to_set = min(self.model.nq, joint_trajectory.shape[1])

        for i in range(n_frames):
            # 1. Set Joint Angles
            self.data.qpos[:n_joints_to_set] = joint_trajectory[i][:n_joints_to_set]
            
            # 2. Compute Forward Kinematics
            mujoco.mj_kinematics(self.model, self.data)
            
            # 3. Extract Cartesian Position
            if self.is_site:
                ee_path[i] = self.data.site_xpos[self.ee_id]
            else:
                ee_path[i] = self.data.xpos[self.ee_id]
                
        return ee_path

# ---------------------------------------------------------
# 3. Vision Metric Engine
# ---------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.to(DEVICE)
        self.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

# ---------------------------------------------------------
# 4. Main Processor
# ---------------------------------------------------------
class MetricsProcessor:
    def __init__(self, goal_image_path, xml_path, object_id=41):
        # Vision
        print(f"{Fore.CYAN}Loading Vision Models...")
        self.extractor = FeatureExtractor()
        if not os.path.exists(goal_image_path):
            raise FileNotFoundError(f"Goal image not found: {goal_image_path}")
        goal_bgr = cv2.imread(goal_image_path)
        self.goal_emb = self.extractor.get_embedding(goal_bgr)
        self.yolo = YOLO("yolov8n.pt")
        self.target_cls = object_id

        # Kinematics
        print(f"{Fore.CYAN}Loading MuJoCo Model from: {xml_path}")
        try:
            self.fk_solver = MujocoFKSolver(xml_path)
            self.has_fk = True
        except Exception as e:
            print(f"{Fore.RED}[Error] FK Init Failed: {e}")
            self.has_fk = False

    def compute_vision_metrics(self, rgb_folder):
        image_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")) + 
                             glob.glob(os.path.join(rgb_folder, "*.jpg")))
        if not image_files: return None

        framing_errors = []
        in_safe_zone_counts = 0
        
        # Visual Alignment (Final Frame)
        last_img = cv2.imread(image_files[-1])
        final_emb = self.extractor.get_embedding(last_img)
        final_sim_score = torch.sum(self.goal_emb * final_emb).item()

        # Determine Status
        status = "FAIL"
        if final_sim_score > THRESH_SUCCESS:
            status = "SUCCESS"
        elif final_sim_score > THRESH_SEMI:
            status = "SEMI"

        # Framing (All Frames)
        for img_path in image_files:
            frame = cv2.imread(img_path)
            H, W = frame.shape[:2]
            center_img = np.array([W/2, H/2])
            safe_min_x, safe_max_x = W*0.25, W*0.75
            safe_min_y, safe_max_y = H*0.25, H*0.75

            results = self.yolo(frame, verbose=False, conf=0.3)
            detected = False
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == self.target_cls:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        
                        framing_errors.append(np.linalg.norm(np.array([cx, cy]) - center_img))
                        if (safe_min_x < cx < safe_max_x) and (safe_min_y < cy < safe_max_y):
                            in_safe_zone_counts += 1
                        detected = True
                        break
                if detected: break

        avg_framing = np.mean(framing_errors) if framing_errors else 0.0
        srr = (in_safe_zone_counts / len(image_files)) * 100
        return {"vis": final_sim_score, "status": status, "err": avg_framing, "srr": srr}

    def compute_kinematics(self, robot_folder):
        """Loads joints -> computes FK -> computes Cartesian Jerk"""
        if not self.has_fk: return 0.0, 0.0

        joints = None
        # Try loading .npy first, then .csv
        npy_files = glob.glob(os.path.join(robot_folder, "*.npy"))
        if npy_files:
            try:
                data = np.load(max(npy_files, key=os.path.getsize)).astype(np.float64)
                if data.ndim == 2 and data.shape[1] >= 6:
                    joints = data[:, :6]
            except: pass
        
        if joints is None:
            # CSV Fallback
            csv_files = glob.glob(os.path.join(robot_folder, "*.csv"))
            if csv_files:
                try:
                    df = pd.read_csv(csv_files[0])
                    # Look for joint columns
                    cols = [c for c in df.columns if "pos_" in c or "joint" in c]
                    if len(cols) >= 6:
                        joints = df[cols[:6]].to_numpy()
                except: pass

        if joints is None or len(joints) < 5:
            return 0.0, 0.0

        # Check for static robot (std dev check)
        if np.mean(np.std(joints, axis=0)) < 1e-4:
            return 0.0, 0.0

        # 1. Compute Cartesian Path via MuJoCo FK
        ee_path = self.fk_solver.get_ee_path(joints) # Shape (N, 3)
        
        # 2. Path Length
        path_len = np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1))

        # 3. Compute Derivatives (Finite Difference)
        dt = 0.1 # 10Hz assumption
        vel = np.diff(ee_path, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt # Shape (N-3, 3)

        # 4. Metric: Mean Jerk Magnitude
        mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))

        return mean_jerk, path_len

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Episodes folder")
    parser.add_argument("--goal", type=str, required=True, help="Goal image path")
    parser.add_argument("--xml", type=str, default="iris.xml", help="Path or name of robot XML")
    parser.add_argument("--cls", type=int, default=41, help="YOLO Class ID")
    parser.add_argument("--num", type=int, default=10, help="Max episodes to process")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"{Fore.WHITE}--------------------------------------------------")
    print(f"{Fore.YELLOW}  IRIS METRICS (MuJoCo FK Enabled)")
    print(f"{Fore.WHITE}--------------------------------------------------")

    if not os.path.exists(args.root):
        print(f"{Fore.RED}Root not found.")
        return

    # Auto-Name Output
    if args.output is None:
        folder_name = os.path.basename(os.path.normpath(args.root))
        output_filename = f"{folder_name}_metrics.csv"
    else:
        output_filename = args.output

    # Initialize Processor
    try:
        processor = MetricsProcessor(args.goal, args.xml, args.cls)
    except Exception as e:
        print(f"{Fore.RED}Initialization Error: {e}")
        return

    # Find Episodes
    all_episodes = sorted([f.path for f in os.scandir(args.root) 
                           if f.is_dir() and os.path.exists(os.path.join(f.path, 'rgb'))])
    
    # Slice
    episodes_to_process = all_episodes[:args.num]
    print(f"{Fore.GREEN}Processing {len(episodes_to_process)} episodes...\n")

    results = []
    
    pbar = tqdm(episodes_to_process, bar_format="{l_bar}{bar:20}{r_bar}")
    
    for i, ep_path in enumerate(pbar):
        ep_name = os.path.basename(ep_path)
        
        # 1. Vision Metrics
        v_met = processor.compute_vision_metrics(os.path.join(ep_path, "rgb"))
        if not v_met: continue

        # 2. Kinematics Metrics (FK -> Jerk)
        jerk, path_len = 0.0, 0.0
        if os.path.exists(os.path.join(ep_path, "robot")):
            jerk, path_len = processor.compute_kinematics(os.path.join(ep_path, "robot"))

        # Color Logic
        status = v_met["status"]
        if status == "SUCCESS":
            c_suc = Fore.GREEN
        elif status == "SEMI":
            c_suc = Fore.YELLOW
        else:
            c_suc = Fore.RED

        c_jerk = Fore.GREEN if jerk < 5.0 else Fore.YELLOW
        
        short_name = ep_name.split("_episode_")[-1] if "_episode_" in ep_name else ep_name[:15]
        
        # --- FULL REAL-TIME LOGGING ---
        tqdm.write(
            f"{Fore.WHITE}[Ep {short_name}] "
            f"Vis: {v_met['vis']:.3f} | "
            f"Status: {c_suc}{status}{Fore.WHITE} | "
            f"SRR: {v_met['srr']:.0f}% | "
            f"Err: {v_met['err']:.1f}px | "
            f"Jerk: {c_jerk}{jerk:.4f}{Fore.WHITE} | "
            f"Len: {path_len:.2f}m"
        )

        # Record
        results.append({
            "episode": ep_name,
            "visual_alignment": round(v_met["vis"], 4),
            "status": status,
            "success_flag": 1 if status == "SUCCESS" else 0,
            "srr_percent": round(v_met["srr"], 1),
            "framing_error_px": round(v_met["err"], 1),
            "cartesian_jerk": round(jerk, 6),
            "path_length_m": round(path_len, 4)
        })

    # Save
    if results:
        with open(output_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        n_total = len(results)
        n_suc = sum(1 for r in results if r['status'] == "SUCCESS")
        n_semi = sum(1 for r in results if r['status'] == "SEMI")
        
        print(f"\n{Fore.GREEN}================ SUMMARY ================")
        print(f" Total Processed:     {n_total}")
        print(f" Full Success:        {n_suc} ({(n_suc/n_total)*100:.1f}%)")
        print(f" Semi Success:        {n_semi} ({(n_semi/n_total)*100:.1f}%)")
        print(f" Avg Cartesian Jerk:  {np.mean([r['cartesian_jerk'] for r in results]):.4f}")
        print(f" Saved To:            {output_filename}")
        print(f"{Fore.GREEN}=========================================")

if __name__ == "__main__":
    main()