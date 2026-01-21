import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your model class explicitly or dynamically
# Assuming your model file is models/transformer_cvae.py
from models.transformer_cvae import ACT_CVAE_Optimized

class IRISPolicy:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        
        # ----------------------------------------------------
        # 1. Initialize Model Architecture
        # ----------------------------------------------------
        # MUST match the arguments you used for training!
        self.model = ACT_CVAE_Optimized(
            seq_len=8,
            future_steps=15,
            d_model=256,
            nhead=8,
            latent_dim=32
        ).to(self.device)
        
        # ----------------------------------------------------
        # 2. Load Weights
        # ----------------------------------------------------
        print(f"Loading policy from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval() # Vital: Switches off Dropout and uses Eval statistics
        
        # ----------------------------------------------------
        # 3. Pre-processing (Must match Training!)
        # ----------------------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_data):
        """
        Args:
            image_data: Can be a file path (str), a PIL Image, or a numpy array.
        Returns:
            Tensor (1, 3, 224, 224) on device
        """
        img = image_data
        if isinstance(image_data, str):
            img = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data).convert("RGB")
            
        tensor = self.transform(img)
        return tensor.unsqueeze(0).to(self.device) # Add batch dim

    def get_action(self, rgb_seq_images, joint_seq_values, goal_image):
        """
        Args:
            rgb_seq_images: List of 8 PIL Images or paths (History)
            joint_seq_values: Numpy array of shape (8, 6) (History)
            goal_image: PIL Image or path
        
        Returns:
            pred_action_chunk: Numpy array (15, 6) - The predicted future deltas
        """
        with torch.no_grad():
            # 1. Process Images
            # Stack the 8 images into (1, 8, 3, 224, 224)
            rgb_tensors = [self.transform(img).to(self.device) for img in rgb_seq_images]
            rgb_seq = torch.stack(rgb_tensors).unsqueeze(0) 
            
            # 2. Process Goal
            goal_tensor = self.transform(goal_image).unsqueeze(0).to(self.device)
            
            # 3. Process Joints
            # (8, 6) -> (1, 8, 6)
            joint_seq = torch.tensor(joint_seq_values, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 4. Inference
            # For Inference, we pass target_actions=None.
            # The CVAE will automatically sample z=0 (Deterministic/Mean mode)
            pred_delta, stats = self.model(rgb_seq, joint_seq, goal_tensor, target_actions=None)
            
            # 5. Detach to Numpy
            return pred_delta.squeeze(0).cpu().numpy()