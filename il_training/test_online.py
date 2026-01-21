#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import rospy
import message_filters
from sensor_msgs.msg import Image as RosImage, JointState
from std_msgs.msg import Float64MultiArray
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
MAX_DELTA = 0.1 

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
        
        # 4. ROS Setup
        rospy.init_node('iris_neural_policy', anonymous=True)
        self.cmd_pub = rospy.Publisher('/joint_commands_calibrated', Float64MultiArray, queue_size=1)
        
        image_sub = message_filters.Subscriber('/camera/color/image_raw', RosImage)
        joint_sub = message_filters.Subscriber('/joint_states_calibrated', JointState)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_sub], 10, 0.1)
        ts.registerCallback(self.data_callback)

        print("Waiting for ROS messages...")

    def data_callback(self, img_msg, joint_msg):
        with self.lock:
            try:
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
                if len(self.image_buffer) % 2 == 0: 
                    print(f"Buffering... {len(self.image_buffer)}/{SEQ_LEN}")
                return None
            
            # Stack History
            img_seq = torch.stack(list(self.image_buffer)).unsqueeze(0)
            
            joint_seq_np = np.array(list(self.joint_buffer))
            joint_seq = torch.tensor(joint_seq_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            current_q = joint_seq_np[-1]

        # Inference
        with torch.no_grad():
            pred_delta, _ = self.model(img_seq, joint_seq, self.goal_tensor, target_actions=None)
        
        pred_delta = pred_delta.squeeze(0).cpu().numpy()
        
        # Execute first step of prediction (Receding Horizon Control)
        next_delta = pred_delta[0] 
        next_delta = np.clip(next_delta, -MAX_DELTA, MAX_DELTA)
        
        target_q = current_q + next_delta
        return target_q

    def run(self):
        rate = rospy.Rate(CONTROL_HZ)
        print("Policy Running. Publishing commands...")
        
        while not rospy.is_shutdown():
            target_q = self.get_action()
            
            if target_q is not None:
                msg = Float64MultiArray()
                msg.data = target_q.tolist()
                self.cmd_pub.publish(msg)
            
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint", type=str, 
                        default="/media/jerry/SSD/checkpoints/best_iris_experiment_combined.pth",
                        help="Path to trained model")
    
    # Updated Argument: Defaults to "goal1.png" in the SSD folder
    parser.add_argument("--goal", type=str, default="goal1.png", 
                        help="Filename (e.g. goal1.png) or full path to goal image")
    
    args = parser.parse_args()

    # --- Goal Path Logic ---
    # 1. Check if user provided a full path
    if os.path.exists(args.goal):
        goal_path = args.goal
    else:
        # 2. Check in the SSD folder
        potential_path = os.path.join(SSD_GOAL_DIR, args.goal)
        if not os.path.exists(potential_path):
            # 3. Try appending .png if user typed "goal2"
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