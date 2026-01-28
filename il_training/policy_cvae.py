#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import rospy
import message_filters
import time
import csv
from datetime import datetime
from sensor_msgs.msg import Image as RosImage, JointState
from cv_bridge import CvBridge
from PIL import Image as PILImage
from collections import deque
import threading

# =========================================================================
# CONFIGURATION
# =========================================================================
SEQ_LEN = 8
FUTURE_STEPS = 15
NUM_JOINTS = 6
CONTROL_HZ = 10 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Motion Tuning
LOOKAHEAD_STEPS = 1     # Prediction index to aim for (0=immediate, 2=0.2s ahead)
MAX_STEP_RADIANS = 0.2  # Safety limit for sudden jumps
ENABLE_EMA = True       
EMA_ALPHA = 0.3         # 0.3 = Heavy smoothing, 1.0 = No smoothing

# Default Paths
SSD_GOAL_DIR = os.path.expanduser("~/Desktop/goal_images")

# Image Preprocessing (Resize required for live camera feed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
]) 

# =========================================================================
# VISUAL CRITIC (Similarity Metric)
# =========================================================================
class VisualCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pre-trained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        # Remove classification layer (output: 512 feature vector)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.device = device
        self.to(device)
        self.eval()

    def get_embedding(self, img_tensor):
        """Expects normalized tensor (B, C, H, W)"""
        with torch.no_grad():
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

# =========================================================================
# SHARED HELPER MODULES
# =========================================================================

class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int, temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        flat = feature_map.view(B, C, -1)
        attention = torch.nn.functional.softmax(flat / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=2)
        expected_y = torch.sum(self.pos_y * attention, dim=2)
        return torch.cat([expected_x, expected_y], dim=1)

class CVAEEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, actions):
        B = actions.shape[0]
        flat_actions = actions.view(B, -1)
        x = self.encoder(flat_actions)
        return self.mean_proj(x), self.logvar_proj(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# =========================================================================
# MODEL ARCHITECTURES
# =========================================================================

# --- MODEL 1: FULL (RGB + Joints + Goal) ---
class CVAE_RGB_Joints_Goal_Absolute(nn.Module):
    def __init__(self, seq_len=8, future_steps=15, d_model=256, nhead=8, 
                 enc_layers=4, dec_layers=4, ff_dim=1024, dropout=0.1, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        resnet = models.resnet18(weights=None) 
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.spatial_softmax = SpatialSoftmax(height=7, width=7)

        self.rgb_proj = nn.Linear(1024, d_model)
        self.joint_proj = nn.Linear(NUM_JOINTS, d_model)
        self.fusion_proj = nn.Sequential(nn.Linear(d_model*2, d_model), nn.GELU())
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cvae = CVAEEncoder(future_steps * NUM_JOINTS, 256, latent_dim)

        self.enc_pos = nn.Parameter(torch.zeros(seq_len + 2, d_model)) 
        self.action_queries = nn.Parameter(torch.zeros(future_steps, d_model))
        self.dec_pos = nn.Parameter(torch.zeros(future_steps, d_model))
        
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff_dim, dropout, batch_first=True)
        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, joint_seq, goal_image, target_actions=None):
        B, S, C, H, W = rgb_seq.shape
        z = torch.zeros((B, self.latent_dim), device=rgb_seq.device)
        latent_tok = self.latent_proj(z).unsqueeze(1)

        x_seq = rgb_seq.view(B*S, C, H, W)
        with torch.no_grad(): feat = self.backbone(x_seq)
        rgb_tok = self.rgb_proj(self.spatial_softmax(feat).view(B, S, -1))
        
        with torch.no_grad(): goal_feat = self.backbone(goal_image)
        goal_tok = self.rgb_proj(self.spatial_softmax(goal_feat)).unsqueeze(1)
        
        joint_tok = self.joint_proj(joint_seq)
        
        enc_tok = self.fusion_proj(torch.cat([rgb_tok, joint_tok], dim=-1))
        enc_tok = torch.cat([latent_tok, enc_tok, goal_tok], dim=1) 
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        q = self.action_queries.unsqueeze(0).expand(B, -1, -1) + self.dec_pos.unsqueeze(0)
        out = self.transformer(enc_tok, q)
        return self.head(out), (None, None)

# --- MODEL 2: VISUAL (RGB + Goal Only) ---
class CVAE_RGB_Goal_Absolute(nn.Module):
    def __init__(self, seq_len=8, future_steps=15, d_model=256, nhead=8, 
                 enc_layers=4, dec_layers=4, ff_dim=1024, dropout=0.1, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.spatial_softmax = SpatialSoftmax(height=7, width=7)

        self.rgb_proj = nn.Linear(1024, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cvae = CVAEEncoder(future_steps * NUM_JOINTS, 256, latent_dim)

        self.enc_pos = nn.Parameter(torch.zeros(seq_len + 2, d_model)) 
        self.action_queries = nn.Parameter(torch.zeros(future_steps, d_model))
        self.dec_pos = nn.Parameter(torch.zeros(future_steps, d_model))
        
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff_dim, dropout, batch_first=True)
        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, goal_image, target_actions=None):
        B, S, C, H, W = rgb_seq.shape
        z = torch.zeros((B, self.latent_dim), device=rgb_seq.device)
        latent_tok = self.latent_proj(z).unsqueeze(1)

        x_seq = rgb_seq.view(B*S, C, H, W)
        with torch.no_grad(): feat = self.backbone(x_seq)
        rgb_tok = self.rgb_proj(self.spatial_softmax(feat).view(B, S, -1))
        
        with torch.no_grad(): goal_feat = self.backbone(goal_image)
        goal_tok = self.rgb_proj(self.spatial_softmax(goal_feat)).unsqueeze(1)
        
        enc_tok = torch.cat([latent_tok, rgb_tok, goal_tok], dim=1) 
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        q = self.action_queries.unsqueeze(0).expand(B, -1, -1) + self.dec_pos.unsqueeze(0)
        out = self.transformer(enc_tok, q)
        return self.head(out), (None, None)

# --- MODEL 3: RGB (RGB Only) ---
class CVAE_RGB_Only_Absolute(nn.Module):
    def __init__(self, seq_len=8, future_steps=15, d_model=256, nhead=8, 
                 enc_layers=4, dec_layers=4, ff_dim=1024, dropout=0.1, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.spatial_softmax = SpatialSoftmax(height=7, width=7)

        self.rgb_proj = nn.Linear(1024, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cvae = CVAEEncoder(future_steps * NUM_JOINTS, 256, latent_dim)

        self.enc_pos = nn.Parameter(torch.zeros(seq_len + 1, d_model)) 
        self.action_queries = nn.Parameter(torch.zeros(future_steps, d_model))
        self.dec_pos = nn.Parameter(torch.zeros(future_steps, d_model))
        
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff_dim, dropout, batch_first=True)
        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, target_actions=None):
        B, S, C, H, W = rgb_seq.shape
        z = torch.zeros((B, self.latent_dim), device=rgb_seq.device)
        latent_tok = self.latent_proj(z).unsqueeze(1)

        x_seq = rgb_seq.view(B*S, C, H, W)
        with torch.no_grad(): feat = self.backbone(x_seq)
        rgb_tok = self.rgb_proj(self.spatial_softmax(feat).view(B, S, -1))
        
        enc_tok = torch.cat([latent_tok, rgb_tok], dim=1) 
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        q = self.action_queries.unsqueeze(0).expand(B, -1, -1) + self.dec_pos.unsqueeze(0)
        out = self.transformer(enc_tok, q)
        return self.head(out), (None, None)

# =========================================================================
# CONTROLLER
# =========================================================================

class IRISController:
    def __init__(self, model_path, model_type, goal_image_path=None):
        self.device = DEVICE
        self.bridge = CvBridge()
        self.model_path_str = model_path
        self.model_type = model_type.lower()
        self.goal_name_str = "None"
        
        if self.model_type not in ['full', 'visual', 'rgb']:
            raise ValueError("Invalid type. Choose: 'full', 'visual', or 'rgb'")

        # 1. Load Policy Model
        print(f"Loading Model: {model_path} [Type: {self.model_type}]")
        
        if self.model_type == 'full':
            self.model = CVAE_RGB_Joints_Goal_Absolute(seq_len=SEQ_LEN, future_steps=FUTURE_STEPS)
        elif self.model_type == 'visual':
            self.model = CVAE_RGB_Goal_Absolute(seq_len=SEQ_LEN, future_steps=FUTURE_STEPS)
        elif self.model_type == 'rgb':
            self.model = CVAE_RGB_Only_Absolute(seq_len=SEQ_LEN, future_steps=FUTURE_STEPS)

        self.model.to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("Policy Model Loaded Successfully.")
        except RuntimeError as e:
            print(f"\n[ERROR] Architecture Mismatch! Did you select the correct --type?\n{e}")
            exit(1)

        # 2. Load Visual Critic & Goal
        self.critic = VisualCritic(self.device)
        self.goal_tensor = None
        self.goal_embedding = None

        if self.model_type in ['full', 'visual'] or goal_image_path:
            # We load the goal for the Critic even if the RGB-only model doesn't use it for control
            if not goal_image_path or not os.path.exists(goal_image_path):
                # Only raise error if model REQUIRES goal
                if self.model_type in ['full', 'visual']:
                    raise ValueError("This model type requires a --goal image!")
                else:
                    print("Warning: No goal image provided. Visual Similarity tracking disabled.")
            else:
                print(f"Loading Goal: {goal_image_path}")
                self.goal_name_str = os.path.basename(goal_image_path)
                raw_goal = PILImage.open(goal_image_path).convert("RGB")
                
                # Preprocess for Policy (if needed)
                self.goal_tensor = transform(raw_goal).unsqueeze(0).to(self.device)
                
                # Preprocess for Critic (Calculate Embedding Once)
                self.goal_embedding = self.critic.get_embedding(self.goal_tensor)
                print("Goal Embedding Cached.")

        # 3. Buffers
        self.image_buffer = deque(maxlen=SEQ_LEN)
        self.joint_buffer = deque(maxlen=SEQ_LEN)
        self.lock = threading.Lock()
        
        self.prev_target_q = None 
        self.joint_names = []
        self.latest_similarity = 0.0
        
        self.setup_logging()

        # 4. ROS Setup
        rospy.init_node('iris_neural_policy', anonymous=True)
        self.cmd_pub = rospy.Publisher('/joint_commands_calibrated', JointState, queue_size=1)
        
        image_sub = message_filters.Subscriber('/camera/color/image_raw', RosImage)
        joint_sub = message_filters.Subscriber('/joint_states_calibrated', JointState)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_sub], 10, 0.1)
        ts.registerCallback(self.data_callback)

        print("Waiting for ROS messages...")

    def setup_logging(self):
        model_name = os.path.splitext(os.path.basename(self.model_path_str))[0]
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_filename = f"deployment_{self.model_type}_{model_name}_{date_str}.csv"
        
        self.log_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        header = ["timestamp", "model_type", "goal_image", "similarity_score"]
        header.extend([f"curr_j{i}" for i in range(6)])
        header.extend([f"cmd_j{i}" for i in range(6)])
        header.extend([f"step_diff_j{i}" for i in range(6)])
        
        self.csv_writer.writerow(header)

    def data_callback(self, img_msg, joint_msg):
        with self.lock:
            try:
                if not self.joint_names and joint_msg.name:
                    self.joint_names = joint_msg.name
                    
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
                pil_img = PILImage.fromarray(cv_img)
                img_tensor = transform(pil_img).to(self.device)
                
                # --- Update Similarity Score ---
                # We calculate this on the latest frame (no need for buffer)
                if self.goal_embedding is not None:
                    curr_emb = self.critic.get_embedding(img_tensor.unsqueeze(0))
                    # Cosine Similarity = Dot Product (since normalized)
                    self.latest_similarity = torch.sum(self.goal_embedding * curr_emb).item()

                joints = np.array(joint_msg.position[:6], dtype=np.float32)
                
                self.image_buffer.append(img_tensor)
                self.joint_buffer.append(joints)
            except Exception as e:
                pass

    def get_action(self):
        with self.lock:
            if len(self.image_buffer) < SEQ_LEN:
                if len(self.image_buffer) % 20 == 0: 
                    print(f"Buffering... {len(self.image_buffer)}/{SEQ_LEN}")
                return None
            
            # Prepare Inputs
            img_seq = torch.stack(list(self.image_buffer)).unsqueeze(0)
            joint_seq_np = np.array(list(self.joint_buffer))
            joint_seq = torch.tensor(joint_seq_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            current_physical_q = joint_seq_np[-1]

        # INFERENCE (Type Dependent)
        with torch.no_grad():
            if self.model_type == 'full':
                pred, _ = self.model(img_seq, joint_seq, self.goal_tensor)
            elif self.model_type == 'visual':
                pred, _ = self.model(img_seq, self.goal_tensor)
            elif self.model_type == 'rgb':
                pred, _ = self.model(img_seq)

        pred_joints_seq = pred.squeeze(0).cpu().numpy() # [Future, 6]
        
        # --- LOOKAHEAD & SMOOTHING ---
        step_idx = min(LOOKAHEAD_STEPS, len(pred_joints_seq) - 1)
        raw_target_q = pred_joints_seq[step_idx]
        
        if ENABLE_EMA and (self.prev_target_q is not None):
            smoothed_target_q = (EMA_ALPHA * raw_target_q) + ((1 - EMA_ALPHA) * self.prev_target_q)
        else:
            smoothed_target_q = raw_target_q
            
        # --- SAFETY CLIPPING ---
        delta_from_current = smoothed_target_q - current_physical_q
        safe_delta = np.clip(delta_from_current, -MAX_STEP_RADIANS, MAX_STEP_RADIANS)
        final_cmd_q = current_physical_q + safe_delta

        self.prev_target_q = final_cmd_q
        step_mag = np.max(np.abs(safe_delta))
            
        return final_cmd_q, step_mag, safe_delta, current_physical_q

    def run(self):
        rate = rospy.Rate(CONTROL_HZ)
        print(f"Policy Running [{self.model_type.upper()}]. Lookahead: {LOOKAHEAD_STEPS}")
        counter = 0

        try:
            while not rospy.is_shutdown():
                result = self.get_action()
                if result is not None:
                    target_q, step_mag, step_diff, current_q = result
                    
                    if counter % 10 == 0:
                        sim_status = f"Sim Score: {self.latest_similarity:.4f}"
                        if self.latest_similarity > 0.95:
                            sim_status += " [LOCKED]"
                        elif self.latest_similarity > 0.90:
                            sim_status += " [ALIGNED]"
                            
                        print("-" * 40)
                        print(f"Cmd Joint: {np.round(target_q, 4)}")
                        print(f"Max Jump:  {step_mag:.4f} | {sim_status}")
                    
                    # CSV Log
                    row = [time.time(), self.model_type, self.goal_name_str, self.latest_similarity]
                    row.extend(current_q.tolist()) 
                    row.extend(target_q.tolist())  
                    row.extend(step_diff.tolist()) 
                    self.csv_writer.writerow(row)

                    # Publish
                    msg = JointState()
                    msg.header.stamp = rospy.Time.now()
                    msg.position = target_q.tolist()
                    msg.name = self.joint_names if self.joint_names else ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
                    self.cmd_pub.publish(msg)
                    counter += 1
                rate.sleep()
        finally:
            print("\nClosing Log File...")
            self.log_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--type", type=str, required=True, choices=['full', 'visual', 'rgb'], 
                        help="Model Type: full (RGB+Joints+Goal), visual (RGB+Goal), rgb (RGB only)")
    parser.add_argument("--goal", type=str, default=None, help="Goal image filename or path")
    
    args = parser.parse_args()

    # Handle Goal Path
    goal_path = None
    if args.goal:
        if os.path.exists(args.goal):
            goal_path = args.goal
        else:
            potential_path = os.path.join(SSD_GOAL_DIR, args.goal)
            if not os.path.exists(potential_path): potential_path += ".png"
            if os.path.exists(potential_path): goal_path = potential_path

    try:
        controller = IRISController(args.checkpoint, args.type, goal_path)
        controller.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()