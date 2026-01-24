import torch
import torch.nn as nn
from torchvision import models

# Constants
NUM_JOINTS = 6

# --- 1. Helper: Spatial Softmax ---
class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int, temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        flat = feature_map.view(B, C, -1)
        attention = torch.nn.functional.softmax(flat / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=2)
        expected_y = torch.sum(self.pos_y * attention, dim=2)
        return torch.cat([expected_x, expected_y], dim=1)

# --- 2. Main Model: RGB History -> Absolute Position ---
class Transformer_RGB_Only_Absolute(nn.Module):
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
    ):
        super().__init__()
        self.d_model = d_model

        # --- Vision Backbone (Frozen ResNet18) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        
        # Freeze Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.spatial_softmax = SpatialSoftmax(height=7, width=7)
        self.vision_feature_dim = 512 * 2

        # --- Projection ---
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.vision_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # --- Transformer ---
        # Positional Encoding: Just the sequence length (No Goal Token)
        self.enc_pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        
        # Decoder Queries (Future Steps)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

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

        # Output Head: Predicts ABSOLUTE joint positions
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS),
        )

    def forward(self, rgb_seq):
        """
        Args:
            rgb_seq: (B, S, 3, H, W) - Sequence of observation images
        Returns:
            pred_absolute_joints: (B, Future_Steps, 6)
        """
        B, S, C, H, W = rgb_seq.shape
        
        # --- 1. Vision History Encoding ---
        x_seq = rgb_seq.view(B * S, C, H, W)
        
        with torch.no_grad(): 
            feat_map = self.backbone(x_seq)
        
        feat_coords_seq = self.spatial_softmax(feat_map).view(B, S, -1)
        enc_tok = self.rgb_proj(feat_coords_seq) # (B, S, D)

        # --- 2. Add Positional Embeddings ---
        # Only adding to the visual tokens, no goal token
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        # --- 3. Transformer Decoder ---
        q = self.action_queries.unsqueeze(0).expand(B, -1, -1) + self.dec_pos.unsqueeze(0)
        
        # The decoder queries (q) attend to the visual history (enc_tok)
        dec_out = self.transformer(enc_tok, q)
        
        # Prediction
        pred_absolute_joints = self.head(dec_out) 
        
        return pred_absolute_joints