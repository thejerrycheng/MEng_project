import torch
import torch.nn as nn
from torchvision import models

# Constants
NUM_JOINTS = 6

# --- Helper: Spatial Softmax ---
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

# --- Helper: CVAE Encoder ---
class CVAEEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, actions, current_joints):
        B = actions.shape[0]
        flat_actions = actions.view(B, -1)
        inputs = torch.cat([flat_actions, current_joints], dim=1)
        x = self.encoder(inputs)
        mu = self.mean_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# --- Main Model: Fully Frozen Backbone ---
class ACT_CVAE_Frozen(nn.Module):
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
        latent_dim: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.d_model = d_model
        self.latent_dim = latent_dim

        # --- Vision Backbone (Frozen ResNet18) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        
        # FREEZE ENTIRE BACKBONE
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.spatial_softmax = SpatialSoftmax(height=7, width=7)
        self.vision_feature_dim = 512 * 2

        # --- Projections ---
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.vision_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        
        self.joint_proj = nn.Sequential(
            nn.Linear(NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # --- CVAE & Transformer ---
        cvae_input_dim = (future_steps * NUM_JOINTS) + NUM_JOINTS
        self.cvae_encoder = CVAEEncoder(cvae_input_dim, hidden_dim=256, latent_dim=latent_dim)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        self.enc_pos = nn.Parameter(torch.randn(seq_len + 2, d_model) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS),
        )

    def forward(self, rgb_seq, joint_seq, goal_image, target_actions=None):
        B, S, C, H, W = rgb_seq.shape
        is_training = target_actions is not None
        
        mu, logvar = None, None
        
        if is_training:
            current_joint = joint_seq[:, -1, :] 
            mu, logvar = self.cvae_encoder(target_actions, current_joint)
            z = self.cvae_encoder.reparameterize(mu, logvar)
        else:
            z = torch.zeros((B, self.latent_dim), device=rgb_seq.device)
            
        latent_tok = self.latent_proj(z).unsqueeze(1)

        x_seq = rgb_seq.view(B * S, C, H, W)
        with torch.no_grad(): # Explicitly disable grad for backbone logic
            feat_map = self.backbone(x_seq)
            
        feat_coords_seq = self.spatial_softmax(feat_map).view(B, S, -1)
        rgb_tok = self.rgb_proj(feat_coords_seq)
        
        with torch.no_grad():
            goal_feat = self.backbone(goal_image)
            
        feat_coords_goal = self.spatial_softmax(goal_feat)
        goal_tok = self.rgb_proj(feat_coords_goal).unsqueeze(1)
        
        joint_tok = self.joint_proj(joint_seq)
        enc_tok = self.fusion_proj(torch.cat([rgb_tok, joint_tok], dim=-1))
        enc_tok = torch.cat([latent_tok, enc_tok, goal_tok], dim=1)
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        q = self.action_queries.unsqueeze(0).expand(B, -1, -1) + self.dec_pos.unsqueeze(0)
        dec_out = self.transformer(enc_tok, q)
        pred_action_delta = self.head(dec_out) 
        
        return pred_action_delta, (mu, logvar)