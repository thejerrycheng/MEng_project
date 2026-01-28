#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import rospy
import message_filters
import time
import signal
import sys
import csv
import threading
from collections import deque
from PIL import Image as PILImage
import torchvision.transforms as transforms
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import Image as RosImage, JointState

# MuJoCo for Kinematics
import mujoco 

# Import your model
from models.transformer_cvae import ACT_CVAE_Optimized

# --------------------------
# Configuration
# --------------------------
SEQ_LEN = 8
FUTURE_STEPS = 15
CONTROL_HZ = 10 
MAX_DURATION_SEC = 25.0  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DELTA = 0.1 

# Paths
SSD_GOAL_DIR = "/media/jerry/SSD/goal_images"
LOG_DIR = "online_logs" # Folder to save CSVs

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# Kinematics Helper
# --------------------------
class IRISKinematics:
    """Analytical FK for calculating EE position (XYZ)"""
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
        return T[:3, 3] # XYZ

# --------------------------
# CSV Logger & Metrics
# --------------------------
class RealTimeLogger:
    def __init__(self, goal_name):
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filename = os.path.join(LOG_DIR, f"log_{goal_name}_{timestamp}.csv")
        
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Header: Time, Joint 0-5, EE X, EE Y, EE Z
        self.writer.writerow(["time", "j0", "j1", "j2", "j3", "j4", "j5", "x", "y", "z"])
        
        self.positions = [] # Store for summary calculation
        self.start_time = None
        print(f"[LOG] Saving real-time data to: {self.filename}")

    def log(self, q, xyz):
        if self.start_time is None: self.start_time = time.time()
        t = time.time() - self.start_time
        
        # Write Row
        row = [f"{t:.4f}"] + [f"{x:.5f}" for x in q] + [f"{x:.5f}" for x in xyz]
        self.writer.writerow(row)
        self.file.flush() # Force write to disk immediately
        
        self.positions.append(xyz)

    def close(self):
        self.file.close()
        print(f"[LOG] File closed safely: {self.filename}")

    def report_summary(self):
        if len(self.positions) < 4: return "Not enough data for summary."
        
        traj = np.array(self.positions)
        dt = 1.0 / CONTROL_HZ
        
        # Smoothness (Jerk)
        vel = np.diff(traj, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        smoothness = np.mean(np.sum(jerk**2, axis=1))
        
        # Path Length
        dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        
        return f"Total Smoothness (Jerk): {smoothness:.4f} | Total Path Length: {dist:.4f} m"

# --------------------------
# Controller
# --------------------------
class IRISController:
    def __init__(self, model_path, goal_image_path):
        self.device = DEVICE
        self.bridge = CvBridge()
        self.fk = IRISKinematics()
        
        # Initialize Logger with goal name
        goal_name = os.path.basename(goal_image_path).split('.')[0]
        self.logger = RealTimeLogger(goal_name)
        
        # 1. Load Model
        print(f"Loading Model: {model_path}")
        self.model = ACT_CVAE_Optimized(
            seq_len=SEQ_LEN, future_steps=FUTURE_STEPS,
            d_model=256, nhead=8, latent_dim=32
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 2. Load Goal
        print(f"Loading Goal: {goal_image_path}")
        raw_goal = PILImage.open(goal_image_path).convert("RGB")
        self.goal_tensor = transform(raw_goal).unsqueeze(0).to(self.device)

        # 3. State
        self.image_buffer = deque(maxlen=SEQ_LEN)
        self.joint_buffer = deque(maxlen=SEQ_LEN)
        self.lock = threading.Lock()
        self.joint_names = []
        self.last_command = None
        self.start_t = None
        self.is_holding = False
        
        # 4. ROS
        rospy.init_node('iris_neural_policy', anonymous=True)
        self.cmd_pub = rospy.Publisher('/joint_commands_calibrated', JointState, queue_size=1)
        
        image_sub = message_filters.Subscriber('/camera/color/image_raw', RosImage)
        joint_sub = message_filters.Subscriber('/joint_states_calibrated', JointState)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_sub], 10, 0.1)
        ts.registerCallback(self.data_callback)

        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self.stop_handler)

    def data_callback(self, img_msg, joint_msg):
        with self.lock:
            try:
                if not self.joint_names and joint_msg.name:
                    self.joint_names = joint_msg.name
                    
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
                pil_img = PILImage.fromarray(cv_img)
                img_tensor = transform(pil_img).to(self.device)
                
                joints = np.array(joint_msg.position[:6], dtype=np.float32)
                
                self.image_buffer.append(img_tensor)
                self.joint_buffer.append(joints)
            except Exception:
                pass

    def get_action(self):
        # 1. TIMEOUT CHECK
        if self.start_t is not None:
            elapsed = time.time() - self.start_t
            if elapsed > MAX_DURATION_SEC:
                if not self.is_holding:
                    print(f"\n[TIMEOUT] Max duration {MAX_DURATION_SEC}s reached.")
                    print(f"[HOLD] Holding last position.")
                    self.is_holding = True
                return self.last_command 

        # 2. INFERENCE
        with self.lock:
            if len(self.image_buffer) < SEQ_LEN:
                if len(self.image_buffer) % 20 == 0: print(f"Buffering... {len(self.image_buffer)}/{SEQ_LEN}")
                return None
            
            # Start timer
            if self.start_t is None:
                self.start_t = time.time()
                print("[START] Timer started.")

            img_seq = torch.stack(list(self.image_buffer)).unsqueeze(0)
            joint_seq_np = np.array(list(self.joint_buffer))
            joint_seq = torch.tensor(joint_seq_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            current_q = joint_seq_np[-1]

        with torch.no_grad():
            pred_delta, _ = self.model(img_seq, joint_seq, self.goal_tensor, target_actions=None)
        
        pred_delta = pred_delta.squeeze(0).cpu().numpy()
        next_delta = np.clip(pred_delta[0], -MAX_DELTA, MAX_DELTA)
        
        target_q = current_q + next_delta
        self.last_command = target_q
        
        # 3. LOGGING (Real-Time CSV)
        xyz = self.fk.forward(target_q)
        self.logger.log(target_q, xyz)
        
        return target_q

    def stop_handler(self, sig, frame):
        print("\n\n=== SESSION SUMMARY ===")
        print(self.logger.report_summary())
        self.logger.close()
        print("=======================")
        sys.exit(0)

    def run(self):
        rate = rospy.Rate(CONTROL_HZ)
        print("Waiting for data...")
        
        while not rospy.is_shutdown():
            with self.lock:
                if len(self.image_buffer) >= SEQ_LEN: break
            rate.sleep()
            
        print(f"Policy Running (Timeout: {MAX_DURATION_SEC}s)...")
        
        last_time = time.time()
        counter = 0

        while not rospy.is_shutdown():
            target_q = self.get_action()
            
            if target_q is not None:
                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.position = target_q.tolist()
                if self.joint_names: msg.name = self.joint_names
                self.cmd_pub.publish(msg)
                
                counter += 1
                if counter % 10 == 0:
                    curr = time.time()
                    hz = 10.0 / (curr - last_time)
                    status = "HOLDING" if self.is_holding else "ACTIVE"
                    print(f"[{status}] Rate: {hz:.1f} Hz")
                    last_time = curr
            
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="/media/jerry/SSD/checkpoints/best_iris_experiment_combined.pth")
    parser.add_argument("--goal", type=str, default="goal1.png")
    args = parser.parse_args()

    if os.path.exists(args.goal):
        goal_path = args.goal
    else:
        potential_path = os.path.join(SSD_GOAL_DIR, args.goal)
        if not os.path.exists(potential_path): potential_path += ".png"
        if os.path.exists(potential_path):
            goal_path = potential_path
        else:
            print(f"Error: Goal '{args.goal}' not found.")
            return

    try:
        controller = IRISController(args.checkpoint, goal_path)
        controller.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()