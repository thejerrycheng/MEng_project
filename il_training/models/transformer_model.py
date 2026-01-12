# models/transformer_model.py
# Revised ACT-style architecture (SOTA-aligned) for IRIS
#
# Key upgrades vs your current version:
# - ResNet34 backbone (stronger features than ResNet18)
# - ImageNet-style normalization expected upstream (documented below)
# - Light projection + dropout (helps small/medium datasets)
# - LayerNorm on tokens (stability)
# - Proper encoder/decoder positional embeddings (separate, cleaner)
# - Learnable action queries + optional decoder positional embedding
#
# NOTE:
# - Dataset should output rgb_seq normalized with ImageNet mean/std.
#   mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
# - rgb_seq shape: (B, S, 3, H, W) where H=W=128 is fine.
# - joint_seq shape: (B, S, 6) in radians
# - goal_xyz shape: (B, 3) in meters (FK end-effector position)
# - Output: (B, F, 6) predicted Δq (radians) for each future step.

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

        # -------------------------
        # Vision backbone (ResNet34)
        # -------------------------
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B,512,1,1)
        self.backbone_out_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # -------------------------
        # Token projections
        # -------------------------
        # Visual embedding -> d_model
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Proprio embedding -> d_model
        self.joint_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
        )

        # Goal token (Cartesian xyz) -> d_model
        self.goal_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
        )

        # -------------------------
        # Positional embeddings
        # -------------------------
        # Encoder has S observation tokens + 1 goal token
        self.enc_pos = nn.Parameter(torch.randn(seq_len + 1, d_model) * 0.02)

        # Decoder has F action query tokens
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        self.use_decoder_pos = use_decoder_pos
        if use_decoder_pos:
            self.dec_pos = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        # -------------------------
        # Transformer
        # -------------------------
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # more stable transformer (pre-norm)
        )

        # -------------------------
        # Output head: token -> Δq
        # -------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS),
        )

    def forward(self, rgb_seq: torch.Tensor, joint_seq: torch.Tensor, goal_xyz: torch.Tensor) -> torch.Tensor:
        """
        rgb_seq:  (B, S, 3, H, W) float, ImageNet-normalized recommended
        joint_seq:(B, S, 6)
        goal_xyz: (B, 3)
        returns:  (B, F, 6) predicted Δq
        """
        B, S, C, H, W = rgb_seq.shape
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got S={S}")

        # ---- backbone over frames ----
        x = rgb_seq.view(B * S, C, H, W)
        feat = self.backbone(x).view(B, S, self.backbone_out_dim)  # (B,S,512)

        # ---- build encoder tokens ----
        rgb_tok = self.rgb_proj(feat)              # (B,S,D)
        joint_tok = self.joint_proj(joint_seq)     # (B,S,D)
        enc_tok = rgb_tok + joint_tok              # fuse by sum (stable)

        goal_tok = self.goal_proj(goal_xyz).unsqueeze(1)  # (B,1,D)
        enc_tok = torch.cat([enc_tok, goal_tok], dim=1)   # (B,S+1,D)

        # add encoder positional embedding
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        # ---- decoder queries ----
        q = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # (B,F,D)
        if self.use_decoder_pos:
            q = q + self.dec_pos.unsqueeze(0)

        # ---- transformer forward ----
        dec_out = self.transformer(enc_tok, q)  # (B,F,D)

        # ---- predict Δq ----
        return self.head(dec_out)
