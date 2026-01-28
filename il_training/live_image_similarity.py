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

# ---------------------------------------------------------
# 1. Configuration & Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Visual Thresholds
THRESH_LOCKED = 0.95
THRESH_ALIGNED = 0.90
THRESH_SEARCHING = 0.80

# ---------------------------------------------------------
# 2. Feature Extractor (ResNet18)
# ---------------------------------------------------------
class FeatureExtractor(nn.Module):
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, img_pil):
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

# ---------------------------------------------------------
# 3. ROS Camera Listener
# ---------------------------------------------------------
class RosCameraListener:
    def __init__(self, topic="/camera/color/image_raw"):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.topic = topic
        self.connected = False
        
        # Initialize Subscriber
        self.sub = rospy.Subscriber(self.topic, RosImage, self.callback)
        logging.info(f"Subscribed to ROS topic: {self.topic}")
        logging.info("Waiting for first image message...")

    def callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_frame = cv_image
            if not self.connected:
                self.connected = True
                logging.info("ROS Image Stream Connected!")
        except CvBridgeError as e:
            logging.error(f"CvBridge Error: {e}")

    def get_frame(self):
        return self.latest_frame

# ---------------------------------------------------------
# 4. Main Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live ROS Similarity Monitor")
    parser.add_argument("--goal", type=str, required=True, help="Path to goal image")
    args = parser.parse_args()

    # Initialize ROS Node
    rospy.init_node("iris_visual_alignment_monitor", anonymous=True)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Starting ROS Monitor on {device} ---")

    # 1. Initialize Model
    extractor = FeatureExtractor(device)

    # 2. Load Goal Image
    if not os.path.exists(args.goal):
        logging.error(f"Goal image not found: {args.goal}")
        return

    logging.info(f"Loading Goal: {os.path.basename(args.goal)}")
    try:
        raw_goal = Image.open(args.goal).convert("RGB")
        goal_emb = extractor.get_embedding(raw_goal)
        # Prepare goal visualization (Resize to 640x480 for consistency)
        vis_goal = cv2.cvtColor(np.array(raw_goal.resize((640, 480))), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Failed to process goal image: {e}")
        return

    # 3. Start ROS Listener
    cam = RosCameraListener(topic="/camera/color/image_raw")

    logging.info("--- Press 'q' to Quit ---")

    try:
        while not rospy.is_shutdown():
            start_time = time.time()

            # A. Get Frame
            frame = cam.get_frame()
            if frame is None:
                # Wait briefly if no frame yet
                time.sleep(0.1)
                continue

            # B. Inference
            live_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            live_emb = extractor.get_embedding(live_pil)

            # C. Similarity Calculation
            similarity = torch.sum(goal_emb * live_emb).item()
            
            # D. Metrics
            inference_time = (time.time() - start_time) * 1000 # ms

            # E. Status Logic
            if similarity > THRESH_LOCKED:
                status, color = "LOCKED (Perfect)", (0, 255, 0)
            elif similarity > THRESH_ALIGNED:
                status, color = "ALIGNED (Good)", (0, 255, 255)
            elif similarity > THRESH_SEARCHING:
                status, color = "SEARCHING...", (0, 165, 255)
            else:
                status, color = "LOST / FAR", (0, 0, 255)

            # F. Console Log
            sys.stdout.write(f"\r[Score: {similarity:.4f}] | Status: {status} | Time: {inference_time:.1f}ms")
            sys.stdout.flush()

            # G. Visualization
            # Resize frame to match goal if necessary
            frame_resized = cv2.resize(frame, (640, 480))
            h_vis = np.hstack((frame_resized, vis_goal))
            
            # UI Overlay
            cv2.rectangle(h_vis, (0, 0), (1280, 60), (30, 30, 30), -1)
            cv2.putText(h_vis, f"Sim: {similarity:.4f}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(h_vis, f"Status: {status}", (350, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.putText(h_vis, f"Latency: {inference_time:.1f}ms", (1000, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            
            # Labels
            cv2.putText(h_vis, "ROS TOPIC INPUT", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(h_vis, "GOAL TARGET", (660, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if similarity > THRESH_LOCKED:
                cv2.rectangle(h_vis, (0, 0), (1279, 479), (0, 255, 0), 5)

            cv2.imshow("IRIS ROS Alignment Monitor", h_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n")
                break
            
            # Throttle to ~30Hz to match camera
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    main()