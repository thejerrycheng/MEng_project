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

        # 1. ResNet34 Backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Remove FC and avgpool to keep spatial features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.backbone_out_dim = 512
        
        # Spatial pooling to flatten 512xHxW -> 512
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Projections
        # Vision Projection
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        # Proprioception Projection (Current Joint State)
        self.joint_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
        )
        # Goal Projection (Delta: Start -> End)
        # Assuming Goal is 3D (XYZ delta). If 6D, change input to 6.
        self.goal_proj = nn.Sequential(
            nn.Linear(3, d_model), 
            nn.LayerNorm(d_model),
        )

        # 3. Positional Embeddings
        # +1 for the goal token
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
        # Output: Delta q (change in joint angles)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS),
        )

    def forward(self, rgb_seq, joint_seq, goal_delta):
        """
        Args:
            rgb_seq: (B, Seq_Len, C, H, W)
            joint_seq: (B, Seq_Len, 6)
            goal_delta: (B, 3) -> Vector from seq_start to seq_end
        Returns:
            pred_action_delta: (B, Future_Steps, 6)
        """
        B, S, C, H, W = rgb_seq.shape
        
        # --- Encoder Preparation ---
        # 1. Vision Features
        x = rgb_seq.view(B * S, C, H, W)
        feat_map = self.backbone(x)                 # (B*S, 512, h, w)
        feat = self.spatial_pool(feat_map).flatten(1) # (B*S, 512)
        feat = feat.view(B, S, self.backbone_out_dim)
        rgb_tok = self.rgb_proj(feat)

        # 2. Proprio Features
        joint_tok = self.joint_proj(joint_seq)
        
        # 3. Fuse & Add Goal
        # Simple addition fusion (can also use concatenation)
        enc_tok = rgb_tok + joint_tok
        
        # Process Goal (Delta)
        goal_tok = self.goal_proj(goal_delta).unsqueeze(1) # (B, 1, D)
        
        # Concatenate: [Sequence Tokens, Goal Token]
        enc_tok = torch.cat([enc_tok, goal_tok], dim=1) # Shape: (B, S+1, D)

        # 4. Add Positional Embedding (Learned)
        # enc_pos shape is (S+1, D), broadcasted to B
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        # --- Decoder Preparation ---
        # 5. Decoder Queries (Action Queries)
        q = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        if self.use_decoder_pos:
            q = q + self.dec_pos.unsqueeze(0)

        # --- Transformer Pass ---
        # memory=enc_tok, tgt=q
        dec_out = self.transformer(enc_tok, q)
        
        # --- Prediction ---
        # Predict Delta Q
        pred_action_delta = self.head(dec_out) # (B, Future_Steps, 6)
        
        return pred_action_delta