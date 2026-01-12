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

        # ResNet34 Backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output (B, 512, 1, 1)
        self.backbone_out_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Projections
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.joint_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional Embeddings
        self.enc_pos = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        
        self.use_decoder_pos = use_decoder_pos
        if use_decoder_pos:
            self.dec_pos = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        # Transformer
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

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS),
        )

    def forward(self, rgb_seq, joint_seq, goal_xyz):
        B, S, C, H, W = rgb_seq.shape
        
        # 1. Vision Features
        x = rgb_seq.view(B * S, C, H, W)
        feat = self.backbone(x).view(B, S, self.backbone_out_dim)
        rgb_tok = self.rgb_proj(feat)

        # 2. Proprio Features
        joint_tok = self.joint_proj(joint_seq)
        
        # 3. Fuse & Add Goal
        enc_tok = rgb_tok + joint_tok
        goal_tok = self.goal_proj(goal_xyz).unsqueeze(1)
        enc_tok = torch.cat([enc_tok, goal_tok], dim=1) # Shape: (B, S+1, D)

        # 4. Positional Embedding
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        # 5. Decoder Queries
        q = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        if self.use_decoder_pos:
            q = q + self.dec_pos.unsqueeze(0)

        # 6. Transformer & Head
        dec_out = self.transformer(enc_tok, q)
        return self.head(dec_out) # (B, F, 6)