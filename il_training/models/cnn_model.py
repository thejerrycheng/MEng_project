import torch
import torch.nn as nn
from torchvision import models

class VanillaBC_Visual_Absolute(nn.Module):
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
        # Shared for both Input Frames and Goal Image
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        self.backbone_out_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Feature Fusion & MLP Encoder
        # Inputs: 
        #   - RGB Sequence: (Seq_Len * 512)
        #   - Joint History: (Seq_Len * 6)
        #   - Goal Image:   (512)  <-- Changed from 3 (xyz) to 512 (visual feature)
        input_dim = (self.seq_len * self.backbone_out_dim) + \
                    (self.seq_len * self.num_joints) + \
                    self.backbone_out_dim # Goal Feature

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
        # Output: Absolute positions for future steps
        self.head = nn.Linear(hidden_dim // 2, self.future_steps * self.num_joints)

    def forward(self, rgb_seq, joint_seq, goal_image):
        """
        Args:
            rgb_seq: (B, Seq_Len, 3, H, W)
            joint_seq: (B, Seq_Len, 6)
            goal_image: (B, 3, H, W)   <-- Expecting Image Tensor now
        Returns:
            pred_action_abs: (B, Future_Steps, 6)
        """
        B, S, C, H, W = rgb_seq.shape
        
        # --- 1. Vision Encoding (Input Sequence) ---
        # Reshape to (B*S, C, H, W) to pass all frames through ResNet at once
        x_img = rgb_seq.view(B * S, C, H, W)
        
        img_feat = self.backbone(x_img)         # (B*S, 512, 1, 1)
        img_feat = torch.flatten(img_feat, 1)   # (B*S, 512)
        img_feat = img_feat.view(B, -1)         # (B, S*512)

        # --- 2. Vision Encoding (Goal Image) ---
        # Pass goal image through the SAME backbone
        goal_feat = self.backbone(goal_image)   # (B, 512, 1, 1)
        goal_feat = torch.flatten(goal_feat, 1) # (B, 512)

        # --- 3. Joint Encoding ---
        # Flatten joint history: (B, S*6)
        joint_feat = joint_seq.view(B, -1)

        # --- 4. Fusion ---
        # Concatenate: [Input_Img_Feats, Joint_Feats, Goal_Img_Feats]
        combined_feat = torch.cat([img_feat, joint_feat, goal_feat], dim=1)

        # --- 5. MLP Pass ---
        x = self.mlp(combined_feat)
        
        # --- 6. Output Reshaping ---
        # Prediction: (B, Future_Steps * 6) -> (B, Future_Steps, 6)
        pred_flat = self.head(x)
        pred_action_abs = pred_flat.view(B, self.future_steps, self.num_joints)
        
        return pred_action_abs