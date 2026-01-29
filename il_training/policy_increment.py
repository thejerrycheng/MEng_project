#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import rospy
import message_filters
import time
import csv
from datetime import datetime
from sensor_msgs.msg import Image as RosImage, JointState
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torchvision.transforms as transforms
from collections import deque
import threading

# Import your model
from models.transformer_cvae import ACT_CVAE_Optimized

# --------------------------
# Configuration
# --------------------------
SEQ_LEN = 8
FUTURE_STEPS = 15
CONTROL_HZ = 10 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MOTION TUNING ---
LOOKAHEAD_STEPS = 2     # Look ~0.6s into the future
ACTION_SCALE = 1.0      # Scalar multiplier 
MAX_DELTA = 0.3         # Safety limit

# --- SMOOTHING CONFIG ---
ENABLE_EMA = True       
EMA_ALPHA = 0.3         # Lower = Smoother, Higher = More Responsive

# Paths
SSD_GOAL_DIR = "/media/jerry/SSD/goal_images"

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
]) 

class IRISController:
    def __init__(self, model_path, goal_image_path):
        self.device = DEVICE
        self.bridge = CvBridge()
        self.model_path_str = model_path
        self.goal_name_str = os.path.basename(goal_image_path)
        
        # 1. Load Model
        print(f"Loading Model: {model_path}")
        self.model = ACT_CVAE_Optimized(
            seq_len=SEQ_LEN,
            future_steps=FUTURE_STEPS,
            d_model=256, nhead=8, latent_dim=32
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model Loaded.")

        # 2. Load Goal Image
        print(f"Loading Goal: {goal_image_path}")
        if not os.path.exists(goal_image_path):
            raise FileNotFoundError(f"Goal image not found at {goal_image_path}")
        
        raw_goal = PILImage.open(goal_image_path).convert("RGB")
        self.goal_tensor = transform(raw_goal).unsqueeze(0).to(self.device)

        # 3. Buffers
        self.image_buffer = deque(maxlen=SEQ_LEN)
        self.joint_buffer = deque(maxlen=SEQ_LEN)
        self.lock = threading.Lock()
        
        # Smoothing State
        self.prev_delta = None 
        
        self.joint_names = []
        
        # 4. Logging Setup
        self.setup_logging()

        # 5. ROS Setup
        rospy.init_node('iris_neural_policy', anonymous=True)
         
        self.cmd_pub = rospy.Publisher('/joint_commands_calibrated', JointState, queue_size=1)
        
        image_sub = message_filters.Subscriber('/camera/color/image_raw', RosImage)
        joint_sub = message_filters.Subscriber('/joint_states_calibrated', JointState)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_sub], 10, 0.1)
        ts.registerCallback(self.data_callback)

        print("Waiting for ROS messages...")

    def setup_logging(self):
        # Extract model name from path (e.g. 'best_iris_experiment_combined')
        model_name = os.path.splitext(os.path.basename(self.model_path_str))[0]
        
        # Create Timestamp (e.g. '2026-01-23_18-30-00')
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Construct Filename
        self.csv_filename = f"deployment_{model_name}_{date_str}.csv"
        
        # Create File and Writer
        self.log_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Define Header
        header = ["timestamp", "model_name", "goal_image"]
        
        # Add columns for Current Joint Positions (curr_j0 ... curr_j5)
        header.extend([f"curr_j{i}" for i in range(6)])
        
        # Add columns for Command Joint Positions (cmd_j0 ... cmd_j5)
        header.extend([f"cmd_j{i}" for i in range(6)])
        
        # Add columns for Calculated Deltas (delta_j0 ... delta_j5)
        header.extend([f"delta_j{i}" for i in range(6)])
        
        self.csv_writer.writerow(header)
        print(f"--------------------------------------------------")
        print(f"LOGGING DATA TO: {self.csv_filename}")
        print(f"--------------------------------------------------")

    def data_callback(self, img_msg, joint_msg):
        with self.lock:
            try:
                if not self.joint_names and joint_msg.name:
                    self.joint_names = joint_msg.name
                    print(f"Captured Joint Names: {self.joint_names}")

                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
                pil_img = PILImage.fromarray(cv_img)
                img_tensor = transform(pil_img).to(self.device)
                
                # Assuming incoming joints are already in [J0...J5] order
                joints = np.array(joint_msg.position[:6], dtype=np.float32)
                
                self.image_buffer.append(img_tensor)
                self.joint_buffer.append(joints)
            except Exception as e:
                rospy.logerr(f"Data processing error: {e}")

    def get_action(self):
        with self.lock:
            if len(self.image_buffer) < SEQ_LEN:
                if len(self.image_buffer) % 20 == 0: 
                    print(f"Buffering... {len(self.image_buffer)}/{SEQ_LEN}")
                return None
            
            img_seq = torch.stack(list(self.image_buffer)).unsqueeze(0)
            joint_seq_np = np.array(list(self.joint_buffer))
            joint_seq = torch.tensor(joint_seq_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Capture current state for logging and calculation
            current_q = joint_seq_np[-1]

        # Inference
        with torch.no_grad():
            pred_delta, _ = self.model(img_seq, joint_seq, self.goal_tensor, target_actions=None)
        
        pred_delta = pred_delta.squeeze(0).cpu().numpy() # Shape: [15, 6]
        
        # --- LOOKAHEAD LOGIC ---
        step_idx = min(LOOKAHEAD_STEPS, len(pred_delta) - 1)
        raw_next_delta = pred_delta[step_idx]
        
        # Apply Scaling
        raw_next_delta = raw_next_delta * ACTION_SCALE
        
        # --- SMOOTHING LOGIC ---
        if ENABLE_EMA:
            if self.prev_delta is None:
                smoothed_delta = raw_next_delta
            else:
                smoothed_delta = (EMA_ALPHA * raw_next_delta) + ((1 - EMA_ALPHA) * self.prev_delta)
            
            self.prev_delta = smoothed_delta
            final_delta = smoothed_delta
        else:
            final_delta = raw_next_delta
            
        # Safety Clip
        final_delta = np.clip(final_delta, -MAX_DELTA, MAX_DELTA)
        
        # Calculate Final Target
        target_q = current_q + final_delta
        
        # Log magnitude
        motion_mag = np.max(np.abs(final_delta))
            
        # Return current_q as well so we can log it in run()
        return target_q, motion_mag, final_delta, current_q

    def run(self):
        rate = rospy.Rate(CONTROL_HZ)
        print(f"Policy Running. Lookahead: {LOOKAHEAD_STEPS} steps. Scale: {ACTION_SCALE}")
        
        counter = 0

        try:
            while not rospy.is_shutdown():
                result = self.get_action()
                
                if result is not None:
                    target_q, motion_mag, used_delta, current_q = result
                    
                    # --- DEBUG LOG (Terminal) ---
                    if counter % 10 == 0:
                        print("-" * 30)
                        print(f"Delta Mag: {motion_mag:.5f}")
                        print(f"Target:    {np.round(target_q, 4)}")
                    
                    # --- CSV LOGGING ---
                    row = [time.time(), self.model_path_str, self.goal_name_str]
                    row.extend(current_q.tolist()) # Add Current Joints
                    row.extend(target_q.tolist())  # Add Command Joints
                    row.extend(used_delta.tolist()) # Add Delta
                    
                    self.csv_writer.writerow(row)

                    # --- ROS PUBLISHING ---
                    msg = JointState()
                    msg.header.stamp = rospy.Time.now()
                    msg.position = target_q.tolist()
                    
                    if self.joint_names:
                        msg.name = self.joint_names
                    else:
                        msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

                    self.cmd_pub.publish(msg)
                    counter += 1
                
                rate.sleep()
        finally:
            print("\nClosing Log File...")
            self.log_file.close()
            print(f"Log Saved: {self.csv_filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="/media/jerry/SSD/checkpoints/best_iris_experiment_combined.pth",
                        help="Path to trained model")
    
    parser.add_argument("--goal", type=str, default="goal1.png", 
                        help="Filename or path")
    
    args = parser.parse_args()

    # --- Goal Path Logic ---
    if os.path.exists(args.goal):
        goal_path = args.goal
    else:
        potential_path = os.path.join(SSD_GOAL_DIR, args.goal)
        if not os.path.exists(potential_path):
            potential_path += ".png"
            
        if os.path.exists(potential_path):
            goal_path = potential_path
        else:
            print(f"Error: Could not find goal image '{args.goal}' in {SSD_GOAL_DIR}")
            return

    try:
        controller = IRISController(args.checkpoint, goal_path)
        controller.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()