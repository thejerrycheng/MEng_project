import rospy
from sensor_msgs.msg import Image as RosImage, JointState
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import time
import logging
import sys
from collections import deque
from ultralytics import YOLO  # For Framing Error & SRR

# ---------------------------------------------------------
# 1. Configuration & Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Thresholds
THRESH_LOCKED = 0.95
THRESH_ALIGNED = 0.90
HISTORY_LEN = 30  # Frames to keep for smoothing metrics

# ---------------------------------------------------------
# 2. Metric Calculators
# ---------------------------------------------------------

class FeatureExtractor(nn.Module):
    """Calculates Visual Semantic Alignment"""
    def __init__(self, device):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.device = device
        self.to(device)
        self.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, img_pil):
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

class SmoothnessCalculator:
    """Calculates Jerk (d3x/dt3) from live joint states"""
    def __init__(self, window_size=10, dt=0.1):
        self.history = deque(maxlen=window_size)
        self.dt = dt  # Assumes roughly 10Hz control loop
    
    def update(self, joint_positions):
        self.history.append(np.array(joint_positions))
        if len(self.history) < 4:
            return 0.0
        
        # Numerical Differentiation
        pos = np.array(self.history)
        vel = np.gradient(pos, self.dt, axis=0)
        acc = np.gradient(vel, self.dt, axis=0)
        jerk = np.gradient(acc, self.dt, axis=0)
        
        # Return magnitude of the latest jerk vector
        return np.linalg.norm(jerk[-1])

class FramingCalculator:
    """Calculates Pixel Error & SRR using YOLO"""
    def __init__(self, object_id=41): # 41 = Cup in COCO
        # Load tiny model (fastest inference)
        self.model = YOLO("yolov8n.pt")
        self.target_cls = object_id
        self.safe_zone_history = deque(maxlen=100) # For calculating % SRR

    def process(self, cv_image):
        H, W = cv_image.shape[:2]
        center_img = np.array([W/2, H/2])
        
        # Safe Zone (Middle 50%)
        safe_x_min, safe_x_max = W*0.25, W*0.75
        safe_y_min, safe_y_max = H*0.25, H*0.75
        
        # Run Inference (suppress logs)
        results = self.model(cv_image, verbose=False, conf=0.3)
        
        framing_error = 0.0
        in_safe_zone = False
        detected = False
        box_center = None

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == self.target_cls:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    box_center = (int(cx), int(cy))
                    
                    # 1. Calc Pixel Error
                    framing_error = np.linalg.norm(np.array([cx, cy]) - center_img)
                    
                    # 2. Calc Safe Zone
                    if (safe_x_min < cx < safe_x_max) and (safe_y_min < cy < safe_y_max):
                        in_safe_zone = True
                    
                    detected = True
                    break # Take first target found
            if detected: break

        # Update History
        self.safe_zone_history.append(1 if in_safe_zone else 0)
        srr_percent = (sum(self.safe_zone_history) / len(self.safe_zone_history)) * 100
        
        return framing_error, srr_percent, box_center

# ---------------------------------------------------------
# 3. ROS Listener
# ---------------------------------------------------------
class RobotListener:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_joints = None
        
        rospy.Subscriber("/camera/color/image_raw", RosImage, self.img_cb)
        rospy.Subscriber("/joint_states", JointState, self.joint_cb) # or /joint_states_calibrated

    def img_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: pass

    def joint_cb(self, msg):
        # Taking first 6 joints
        if len(msg.position) >= 6:
            self.latest_joints = msg.position[:6]

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument("--cls", type=int, default=41, help="COCO ID: 41=Cup, 67=Phone")
    args = parser.parse_args()

    rospy.init_node("iris_live_metrics_paper", anonymous=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Calculators
    logging.info("Initializing Metrics Engines...")
    extractor = FeatureExtractor(device)
    framing_calc = FramingCalculator(object_id=args.cls)
    smoothness_calc = SmoothnessCalculator()
    listener = RobotListener()

    # Load Goal
    if not os.path.exists(args.goal):
        logging.error("Goal not found.")
        return
    
    raw_goal = Image.open(args.goal).convert("RGB")
    goal_emb = extractor.get_embedding(raw_goal)
    vis_goal = cv2.cvtColor(np.array(raw_goal.resize((640, 480))), cv2.COLOR_RGB2BGR)

    logging.info("--- Monitoring Started ---")

    try:
        while not rospy.is_shutdown():
            loop_start = time.time()
            
            # 1. Get Data
            frame = listener.latest_frame
            joints = listener.latest_joints
            
            if frame is None:
                time.sleep(0.1)
                continue

            # 2. Compute Metrics
            # A. Visual Similarity
            live_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            live_emb = extractor.get_embedding(live_pil)
            sim_score = torch.sum(goal_emb * live_emb).item()

            # B. Framing & SRR
            frame_err, srr, box_center = framing_calc.process(frame)
            
            # C. Smoothness (Jerk)
            jerk_val = 0.0
            if joints:
                jerk_val = smoothness_calc.update(joints)

            # 3. Console Output
            # Clear line and print stats
            status = "LOCKED" if sim_score > THRESH_LOCKED else "TRACKING"
            sys.stdout.write(f"\r[Sim: {sim_score:.3f}] [Err: {frame_err:.1f}px] [SRR: {srr:.0f}%] [Jerk: {jerk_val:.2f}]")
            sys.stdout.flush()

            # 4. Visualization Overlay
            h_vis = cv2.resize(frame, (640, 480))
            
            # Draw Safe Zone Box
            H, W = h_vis.shape[:2]
            cv2.rectangle(h_vis, (int(W*0.25), int(H*0.25)), (int(W*0.75), int(H*0.75)), (0, 255, 0), 2)
            
            # Draw Detected Object Center
            if box_center:
                # Scale coordinates if resized
                sx, sy = 640/frame.shape[1], 480/frame.shape[0]
                cx, cy = int(box_center[0]*sx), int(box_center[1]*sy)
                cv2.circle(h_vis, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(h_vis, (int(W/2), int(H/2)), (cx, cy), (0, 255, 255), 1)

            # Combine with Goal
            combined = np.hstack((h_vis, vis_goal))
            
            # Dashboard UI
            cv2.rectangle(combined, (0, 0), (1280, 80), (30, 30, 30), -1)
            
            # Metric 1: Similarity
            color_sim = (0, 255, 0) if sim_score > THRESH_LOCKED else (0, 255, 255)
            cv2.putText(combined, f"VISUAL SIM: {sim_score:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_sim, 2)
            
            # Metric 2: Framing
            cv2.putText(combined, f"FRAMING ERR: {frame_err:.1f} px", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Metric 3: SRR
            cv2.putText(combined, f"RETENTION (SRR): {srr:.1f}%", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Metric 4: Jerk
            cv2.putText(combined, f"SMOOTHNESS: {jerk_val:.2f}", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

            cv2.imshow("IRIS Paper Metrics Monitor", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n")
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()