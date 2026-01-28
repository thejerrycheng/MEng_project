import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import time
import logging

# ---------------------------------------------------------
# 0. Configure Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%H:%M:%S",
    level=logging.INFO
)

# ---------------------------------------------------------
# 1. Feature Extractor Setup (ResNet18)
# ---------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Use ResNet18 for feature extraction (consistent with policy backbone)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        
        # Remove the final classification layer (fc) to get raw features
        # Output shape will be (batch, 512, 1, 1) -> Flatten -> (batch, 512)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        
        self.device = device
        self.to(device)
        self.eval()

    def get_embedding(self, img_path):
        # Load and Preprocess
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to load image: {img_path} | Error: {e}")
            return None, 0.0

        # Standard ImageNet normalization
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features & Measure Time
            start_feat = time.time()
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            # Normalize to unit length for Cosine Similarity
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            end_feat = time.time()
            
        return emb, (end_feat - start_feat) * 1000 # Return ms

def main():
    parser = argparse.ArgumentParser(description="Calculate Visual Similarity between two images.")
    parser.add_argument("--goal", type=str, required=True, help="Path to the Goal Image")
    parser.add_argument("--test", type=str, required=True, help="Path to the Test/Result Image")
    args = parser.parse_args()

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using Device: {device}")
    
    # Initialize Model
    logging.info("Loading ResNet18 Feature Extractor...")
    model_start = time.time()
    extractor = FeatureExtractor(device)
    model_load_time = (time.time() - model_start) * 1000
    logging.info(f"Model Loaded in {model_load_time:.2f} ms")

    # Get Embeddings
    logging.info(f"Processing Goal Image: {os.path.basename(args.goal)}")
    goal_emb, goal_time = extractor.get_embedding(args.goal)
    
    logging.info(f"Processing Test Image: {os.path.basename(args.test)}")
    test_emb, test_time = extractor.get_embedding(args.test)

    if goal_emb is not None and test_emb is not None:
        # Calculate Cosine Similarity
        start_sim = time.time()
        similarity = torch.sum(goal_emb * test_emb).item()
        sim_time = (time.time() - start_sim) * 1000
        
        # Summary Log
        logging.info("-" * 40)
        logging.info(f"Similarity Score: {similarity:.4f}")
        logging.info("-" * 40)
        logging.info(f"Timing Breakdown:")
        logging.info(f"  Goal Feature Extraction: {goal_time:.2f} ms")
        logging.info(f"  Test Feature Extraction: {test_time:.2f} ms")
        logging.info(f"  Similarity Calculation:  {sim_time:.4f} ms")
        logging.info(f"  Total Inference Time:    {goal_time + test_time + sim_time:.2f} ms")
        
        # Print raw number at the very end (useful for bash scripts)
        print(f"{similarity:.4f}")

if __name__ == "__main__":
    main()