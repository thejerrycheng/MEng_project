import rospy
from sensor_msgs.msg import Image as RosImage
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
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Configuration & Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

THRESH_LOCKED = 0.95
THRESH_ALIGNED = 0.90

# ---------------------------------------------------------
# 2. Metric Engines (Vision Only)
# ---------------------------------------------------------

class FeatureExtractor(nn.Module):
    """Calculates Visual Semantic Alignment (Cosine Similarity)"""
    def __init__(self, device):
        super().__init__()
        # Load ResNet18 (Standard backbone for perceptual metrics)
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

class FramingCalculator:
    """Calculates Object Centering & Retention using YOLO"""
    def __init__(self, object_id=41): # 41=Cup
        self.model = YOLO("yolov8n.pt") # Tiny model for speed
        self.target_cls = object_id
        self.safe_zone_history = deque(maxlen=100) # For SRR % calculation

    def process(self, cv_image):
        H, W = cv_image.shape[:2]
        center_img = np.array([W/2, H/2])
        
        # Cinematic Safe Zone (Middle 50% of screen)
        safe_x_min, safe_x_max = W*0.25, W*0.75
        safe_y_min, safe_y_max = H*0.25, H*0.75
        
        # Run Detection (Conf=0.3 to avoid flickering)
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
                    
                    # 1. Pixel Error (Distance from center)
                    framing_error = np.linalg.norm(np.array([cx, cy]) - center_img)
                    
                    # 2. Safe Zone Check
                    if (safe_x_min < cx < safe_x_max) and (safe_y_min < cy < safe_y_max):
                        in_safe_zone = True
                    
                    detected = True
                    break 
            if detected: break

        # Update Retention History
        self.safe_zone_history.append(1 if in_safe_zone else 0)
        srr_percent = (sum(self.safe_zone_history) / len(self.safe_zone_history)) * 100
        
        return framing_error, srr_percent, box_center, detected

# ---------------------------------------------------------
# 3. Camera Listener
# ---------------------------------------------------------
class CameraListener:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # Subscribe ONLY to camera
        rospy.Subscriber("/camera/color/image_raw", RosImage, self.img_cb)
        logging.info("Listening to /camera/color/image_raw...")

    def img_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: pass

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="Path to goal image")
    parser.add_argument("--cls", type=int, default=41, help="COCO Class ID (41=Cup)")
    args = parser.parse_args()

    rospy.init_node("iris_visual_metrics", anonymous=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Engines
    extractor = FeatureExtractor(device)
    framing_calc = FramingCalculator(object_id=args.cls)
    listener = CameraListener()

    # Load Goal
    if not os.path.exists(args.goal):
        logging.error("Goal image not found!")
        return
    
    raw_goal = Image.open(args.goal).convert("RGB")
    goal_emb = extractor.get_embedding(raw_goal)
    
    # Prepare Goal Visualization
    vis_goal = cv2.cvtColor(np.array(raw_goal.resize((640, 480))), cv2.COLOR_RGB2BGR)

    logging.info("--- Visual Monitor Started ---")

    try:
        while not rospy.is_shutdown():
            if listener.latest_frame is None:
                time.sleep(0.1)
                continue

            frame = listener.latest_frame.copy()
            
            # ---------------------------
            # A. Compute Metrics
            # ---------------------------
            # 1. Visual Similarity (ResNet)
            live_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            live_emb = extractor.get_embedding(live_pil)
            sim_score = torch.sum(goal_emb * live_emb).item()

            # 2. Framing Error (YOLO)
            frame_err, srr, box_center, detected = framing_calc.process(frame)

            # ---------------------------
            # B. Visualization
            # ---------------------------
            h_vis = cv2.resize(frame, (640, 480))
            H, W = h_vis.shape[:2]
            
            # Draw Cinematic Safe Zone (Green Box)
            cv2.rectangle(h_vis, (int(W*0.25), int(H*0.25)), (int(W*0.75), int(H*0.75)), (0, 255, 0), 2)
            cv2.putText(h_vis, "SAFE ACTION ZONE", (int(W*0.25)+5, int(H*0.25)+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw Object Tracking
            if detected and box_center:
                # Scale coordinates to display size
                sx, sy = 640/frame.shape[1], 480/frame.shape[0]
                cx, cy = int(box_center[0]*sx), int(box_center[1]*sy)
                
                # Draw Center Line (Yellow)
                cv2.line(h_vis, (int(W/2), int(H/2)), (cx, cy), (0, 255, 255), 2)
                # Draw Object Center (Red Dot)
                cv2.circle(h_vis, (cx, cy), 6, (0, 0, 255), -1)
                
                # Display Pixel Error next to object
                cv2.putText(h_vis, f"{frame_err:.0f}px", (cx+10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Combine with Goal Image
            combined = np.hstack((h_vis, vis_goal))
            
            # ---------------------------
            # C. Dashboard UI (Header)
            # ---------------------------
            cv2.rectangle(combined, (0, 0), (1280, 80), (20, 20, 20), -1)
            
            # Status Color
            status_color = (0, 255, 0) if sim_score > THRESH_LOCKED else (0, 165, 255)
            status_text = "LOCKED" if sim_score > THRESH_LOCKED else "TRACKING"

            # Metric 1: Visual Sim
            cv2.putText(combined, f"VISUAL SIM: {sim_score:.3f}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(combined, status_text, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

            # Metric 2: Subject Retention
            srr_color = (0, 255, 0) if srr > 90 else (0, 0, 255)
            cv2.putText(combined, f"RETENTION (SRR): {srr:.1f}%", (350, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, srr_color, 2)
            
            # Metric 3: Framing Error
            cv2.putText(combined, f"CENTER ERROR: {frame_err:.1f} px", (350, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Labels
            cv2.putText(combined, "LIVE CAMERA ANALYSIS", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "TARGET GOAL", (660, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Console Log
            sys.stdout.write(f"\r[Sim: {sim_score:.3f}] [SRR: {srr:.1f}%] [Err: {frame_err:.1f}px]")
            sys.stdout.flush()

            cv2.imshow("IRIS Visual Metrics (No Joint Data)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n")
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()