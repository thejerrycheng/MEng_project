import torch
import torch.nn as nn
from torchvision import models

class VanillaBC_GoalImage(nn.Module):
    def __init__(
        self,
        seq_len: int,
        future_steps: int,
        num_joints: int = 6,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.num_joints = num_joints

        # 1. Shared Vision Backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        self.backbone_out_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. MLP Encoder
        # Input: (Seq_Len * 512) + (Seq_Len * Joints) + (Goal Image Features = 512)
        input_dim = (self.seq_len * self.backbone_out_dim) + \
                    (self.seq_len * self.num_joints) + \
                    self.backbone_out_dim

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

        # 3. Head
        self.head = nn.Linear(hidden_dim // 2, self.future_steps * self.num_joints)

    def forward(self, rgb_seq, joint_seq, goal_image):
        """
        Args:
            rgb_seq: (B, Seq, 3, H, W)
            joint_seq: (B, Seq, 6)
            goal_image: (B, 3, H, W)
        """
        B, S, C, H, W = rgb_seq.shape
        
        # --- 1. Vision Encoding (Sequence) ---
        x_img = rgb_seq.view(B * S, C, H, W)
        img_feat = self.backbone(x_img)         # (B*S, 512, 1, 1)
        img_feat = torch.flatten(img_feat, 1)   # (B*S, 512)
        img_feat = img_feat.view(B, -1)         # (B, S*512)

        # --- 2. Vision Encoding (Goal) ---
        # Pass goal through SAME backbone
        g_feat = self.backbone(goal_image)      # (B, 512, 1, 1)
        g_feat = torch.flatten(g_feat, 1)       # (B, 512)

        # --- 3. Joint Encoding ---
        joint_feat = joint_seq.view(B, -1)      # (B, S*6)

        # --- 4. Fusion ---
        # Concat: [Image History, Joint History, Goal Image Features]
        combined_feat = torch.cat([img_feat, joint_feat, g_feat], dim=1)

        # --- 5. Prediction ---
        x = self.mlp(combined_feat)
        pred_flat = self.head(x)
        return pred_flat.view(B, self.future_steps, self.num_joints)