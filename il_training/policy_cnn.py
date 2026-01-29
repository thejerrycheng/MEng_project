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

# Image Preprocessing (Standard ImageNet Stats)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
]) 

# =========================================================================
# MODEL ARCHITECTURE (Your Provided Model)
# =========================================================================

class VanillaBC_Visual_Absolute(nn.Module):
    def __init__(
        self,
        seq_len: int,
        future_steps: int,
        num_joints: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.num_joints = num_joints

        # 1. Vision Backbone (ResNet34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        self.backbone_out_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Feature Fusion & MLP Encoder
        # Inputs: RGB Sequence + Joint History + Goal Feature
        input_dim = (self.seq_len * self.backbone_out_dim) + \
                    (self.seq_len * self.num_joints) + \
                    self.backbone_out_dim 

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3. Prediction Head
        self.head = nn.Linear(hidden_dim // 2, self.future_steps * self.num_joints)

    def forward(self, rgb_seq, joint_seq, goal_image):
        B, S, C, H, W = rgb_seq.shape
        
        # 1. Vision Encoding (Input Sequence)
        x_img = rgb_seq.view(B * S, C, H, W)
        img_feat = self.backbone(x_img)         # (B*S, 512, 1, 1)
        img_feat = torch.flatten(img_feat, 1)   # (B*S, 512)
        img_feat = img_feat.view(B, -1)         # (B, S*512)

        # 2. Vision Encoding (Goal Image)
        goal_feat = self.backbone(goal_image)   # (B, 512, 1, 1)
        goal_feat = torch.flatten(goal_feat, 1) # (B, 512)

        # 3. Joint Encoding
        joint_feat = joint_seq.view(B, -1)

        # 4. Fusion
        combined_feat = torch.cat([img_feat, joint_feat, goal_feat], dim=1)

        # 5. MLP Pass
        x = self.mlp(combined_feat)
        
        # 6. Output
        pred_flat = self.head(x)
        pred_action_abs = pred_flat.view(B, self.future_steps, self.num_joints)
        
        return pred_action_abs

# =========================================================================
# VISUAL CRITIC (For Monitoring Only - Not used by Policy)
# =========================================================================
class VisualCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.device = device
        self.to(device)
        self.eval()

    def get_embedding(self, img_tensor):
        with torch.no_grad():
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

# =========================================================================
# CONTROLLER
# =========================================================================

class IRISController:
    def __init__(self, model_path, goal_image_path):
        self.device = DEVICE
        self.bridge = CvBridge()
        self.model_path_str = model_path
        
        # 1. Check Goal Image (Required for this architecture)
        if not goal_image_path or not os.path.exists(goal_image_path):
            raise ValueError(f"Goal image is REQUIRED for this policy! Path invalid: {goal_image_path}")

        # 2. Load Policy Model
        print(f"Loading CNN Policy: {model_path}")
        self.model = VanillaBC_Visual_Absolute(
            seq_len=SEQ_LEN, 
            future_steps=FUTURE_STEPS, 
            num_joints=NUM_JOINTS
        )
        self.model.to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle if state_dict is nested or flat
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("Policy Model Loaded Successfully.")
        except Exception as e:
            print(f"\n[ERROR] Failed to load model weights:\n{e}")
            exit(1)

        # 3. Load Goal & Critic
        self.critic = VisualCritic(self.device)
        
        print(f"Loading Goal: {goal_image_path}")
        self.goal_name_str = os.path.basename(goal_image_path)
        raw_goal = PILImage.open(goal_image_path).convert("RGB")
        
        # Preprocess for Policy
        self.goal_tensor = transform(raw_goal).unsqueeze(0).to(self.device)
        
        # Preprocess for Critic (Calculate Embedding Once)
        self.goal_embedding = self.critic.get_embedding(self.goal_tensor)
        print("Goal Embedding Cached.")

        # 4. Buffers
        self.image_buffer = deque(maxlen=SEQ_LEN)
        self.joint_buffer = deque(maxlen=SEQ_LEN)
        self.lock = threading.Lock()
        
        self.prev_target_q = None 
        self.joint_names = []
        self.latest_similarity = 0.0
        
        self.setup_logging()

        # 5. ROS Setup
        rospy.init_node('iris_cnn_policy', anonymous=True)
        self.cmd_pub = rospy.Publisher('/joint_commands_calibrated', JointState, queue_size=1)
        
        image_sub = message_filters.Subscriber('/camera/color/image_raw', RosImage)
        joint_sub = message_filters.Subscriber('/joint_states_calibrated', JointState)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_sub], 10, 0.1)
        ts.registerCallback(self.data_callback)

        print("Waiting for ROS messages...")

    def setup_logging(self):
        model_name = os.path.splitext(os.path.basename(self.model_path_str))[0]
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_filename = f"deploy_cnn_{model_name}_{date_str}.csv"
        
        self.log_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        header = ["timestamp", "goal_image", "similarity_score"]
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
                if self.goal_embedding is not None:
                    curr_emb = self.critic.get_embedding(img_tensor.unsqueeze(0))
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
            img_seq = torch.stack(list(self.image_buffer)).unsqueeze(0) # (1, S, C, H, W)
            joint_seq_np = np.array(list(self.joint_buffer))
            joint_seq = torch.tensor(joint_seq_np, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, S, 6)
            current_physical_q = joint_seq_np[-1]

        # INFERENCE
        with torch.no_grad():
            # Pass inputs exactly as defined in VanillaBC_Visual_Absolute.forward
            pred = self.model(img_seq, joint_seq, self.goal_tensor)

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
        print(f"CNN Policy Running. Lookahead: {LOOKAHEAD_STEPS}")
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
                    row = [time.time(), self.goal_name_str, self.latest_similarity]
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
    parser.add_argument("--goal", type=str, required=True, help="Goal image filename or path")
    
    args = parser.parse_args()

    # Handle Goal Path
    goal_path = args.goal
    if not os.path.exists(goal_path):
        # Try checking default dir
        potential_path = os.path.join(SSD_GOAL_DIR, args.goal)
        if not os.path.exists(potential_path): potential_path += ".png"
        if os.path.exists(potential_path): goal_path = potential_path

    try:
        controller = IRISController(args.checkpoint, goal_path)
        controller.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()