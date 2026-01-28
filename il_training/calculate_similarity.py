import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os

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
            print(f"Error loading {img_path}: {e}")
            return None

        # Standard ImageNet normalization
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features
            emb = self.encoder(img_tensor).flatten(start_dim=1)
            # Normalize to unit length for Cosine Similarity
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        
        return emb

def main():
    parser = argparse.ArgumentParser(description="Calculate Visual Similarity between two images.")
    parser.add_argument("--goal", type=str, required=True, help="Path to the Goal Image")
    parser.add_argument("--test", type=str, required=True, help="Path to the Test/Result Image")
    args = parser.parse_args()

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    extractor = FeatureExtractor(device)

    # Get Embeddings
    goal_emb = extractor.get_embedding(args.goal)
    test_emb = extractor.get_embedding(args.test)

    if goal_emb is not None and test_emb is not None:
        # Calculate Cosine Similarity
        # Since vectors are normalized, Dot Product == Cosine Similarity
        similarity = torch.sum(goal_emb * test_emb).item()
        
        # Output strictly the number (useful for piping to other scripts)
        print(f"{similarity:.4f}")

if __name__ == "__main__":
    main()