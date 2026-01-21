import torch
import torch.nn as nn
from torchvision import models

NUM_JOINTS = 6

class ACT_RGB(nn.Module):
    def __init__(
        self,
        seq_len: int,
        future_steps: int,
        d_model: int = 256,
        nhead: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_decoder_pos: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.d_model = d_model

        # 1. Shared ResNet34 Backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.backbone_out_dim = 512
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Projections
        # Shared Vision Projection (Used for both Sequence and Goal Image)
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        
        # Proprioception Projection
        self.joint_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
        )

        # 3. Positional Embeddings
        # +1 because we append the Goal Token to the sequence
        self.enc_pos = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.02)
        
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        
        self.use_decoder_pos = use_decoder_pos
        if use_decoder_pos:
            self.dec_pos = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        # 4. Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # 5. Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS),
        )

    def forward(self, rgb_seq, joint_seq, goal_image):
        """
        Args:
            rgb_seq: (B, Seq_Len, 3, H, W)
            joint_seq: (B, Seq_Len, 6)
            goal_image: (B, 3, H, W) -> The visual goal
        """
        B, S, C, H, W = rgb_seq.shape
        
        # --- 1. Process Sequence Images ---
        # Flatten sequence: (B*S, C, H, W)
        x_seq = rgb_seq.view(B * S, C, H, W)
        feat_seq = self.backbone(x_seq)                     # (B*S, 512, h, w)
        feat_seq = self.spatial_pool(feat_seq).flatten(1)   # (B*S, 512)
        feat_seq = feat_seq.view(B, S, self.backbone_out_dim)
        
        # Project Sequence
        rgb_tok = self.rgb_proj(feat_seq) # (B, S, D)

        # --- 2. Process Goal Image ---
        # Pass goal through SAME backbone and projection
        feat_goal = self.backbone(goal_image)               # (B, 512, h, w)
        feat_goal = self.spatial_pool(feat_goal).flatten(1) # (B, 512)
        
        # Project Goal
        goal_tok = self.rgb_proj(feat_goal).unsqueeze(1)    # (B, 1, D)

        # --- 3. Process Joints ---
        joint_tok = self.joint_proj(joint_seq) # (B, S, D)
        
        # --- 4. Fusion ---
        # Add visual history + joint history
        enc_tok = rgb_tok + joint_tok 
        
        # Append Goal Token to the end of the sequence
        # Shape becomes: (B, S+1, D)
        enc_tok = torch.cat([enc_tok, goal_tok], dim=1)

        # Add Positional Embedding (Learned)
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        # --- 5. Transformer ---
        # Decoder Queries
        q = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        if self.use_decoder_pos:
            q = q + self.dec_pos.unsqueeze(0)

        dec_out = self.transformer(enc_tok, q)
        
        # Output Delta Q
        pred_action_delta = self.head(dec_out) 
        
        return pred_action_delta