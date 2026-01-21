import torch
import torch.nn as nn
from torchvision import models

class VanillaBC(nn.Module):
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
        # We share this backbone across all timesteps in the sequence
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        self.backbone_out_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Feature Fusion & MLP Encoder
        # We flatten (Seq_Len * 512) + (Seq_Len * Joints) + (Goal)
        input_dim = (self.seq_len * self.backbone_out_dim) + \
                    (self.seq_len * self.num_joints) + \
                    3 # Goal delta (x,y,z)

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
        # Output: Flattened vector of (Future_Steps * Joints)
        self.head = nn.Linear(hidden_dim // 2, self.future_steps * self.num_joints)

    def forward(self, rgb_seq, joint_seq, goal_delta):
        """
        Args:
            rgb_seq: (B, Seq_Len, 3, H, W)
            joint_seq: (B, Seq_Len, 6)
            goal_delta: (B, 3)
        Returns:
            pred_action_delta: (B, Future_Steps, 6)
        """
        B, S, C, H, W = rgb_seq.shape
        
        # --- 1. Vision Encoding ---
        # Flatten sequence to batch dimension for efficient ResNet pass
        # (B*S, 3, H, W)
        x_img = rgb_seq.view(B * S, C, H, W)
        
        img_feat = self.backbone(x_img)         # (B*S, 512, 1, 1)
        img_feat = torch.flatten(img_feat, 1)   # (B*S, 512)
        
        # Reshape back to separate sequence: (B, S*512)
        img_feat = img_feat.view(B, -1)

        # --- 2. Joint Encoding ---
        # Flatten joints: (B, S*6)
        joint_feat = joint_seq.view(B, -1)

        # --- 3. Fusion ---
        # Concatenate everything: Image Hist + Joint Hist + Goal
        # Shape: (B, [S*512 + S*6 + 3])
        combined_feat = torch.cat([img_feat, joint_feat, goal_delta], dim=1)

        # --- 4. MLP Pass ---
        x = self.mlp(combined_feat)
        
        # --- 5. Output Reshaping ---
        # Prediction: (B, Future_Steps * 6) -> (B, Future_Steps, 6)
        pred_flat = self.head(x)
        pred_action_delta = pred_flat.view(B, self.future_steps, self.num_joints)
        
        return pred_action_delta