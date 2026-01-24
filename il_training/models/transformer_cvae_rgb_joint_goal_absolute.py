import torch
import torch.nn as nn
from torchvision import models

# Constants
NUM_JOINTS = 6

# --- Helper Classes ---
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
    
    def forward(self, actions):
        B = actions.shape[0]
        flat_actions = actions.view(B, -1)
        x = self.encoder(flat_actions)
        return self.mean_proj(x), self.logvar_proj(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# --- Main Model ---
class CVAE_RGB_Joints_Goal_Absolute(nn.Module):
    def __init__(self, seq_len=8, future_steps=15, d_model=256, nhead=8, 
                 enc_layers=4, dec_layers=4, ff_dim=1024, dropout=0.1, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Frozen Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        for param in self.backbone.parameters(): param.requires_grad = False
        self.spatial_softmax = SpatialSoftmax(height=7, width=7)

        # Encoders
        self.rgb_proj = nn.Linear(1024, d_model)
        self.joint_proj = nn.Linear(NUM_JOINTS, d_model)
        self.fusion_proj = nn.Sequential(nn.Linear(d_model*2, d_model), nn.GELU())
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # CVAE Encoder (Input = Future Trajectory)
        self.cvae = CVAEEncoder(future_steps * NUM_JOINTS, 256, latent_dim)

        # Transformer
        # +2 Tokens: Latent, Goal
        self.enc_pos = nn.Parameter(torch.randn(seq_len + 2, d_model) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(future_steps, d_model) * 0.02)
        
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff_dim, dropout, batch_first=True)
        self.head = nn.Linear(d_model, NUM_JOINTS)

    def forward(self, rgb_seq, joint_seq, goal_image, target_actions=None):
        B, S, C, H, W = rgb_seq.shape
        
        # 1. CVAE Handling
        mu, logvar = None, None
        if target_actions is not None:
            mu, logvar = self.cvae(target_actions)
            z = self.cvae.reparameterize(mu, logvar)
        else:
            z = torch.zeros((B, self.latent_dim), device=rgb_seq.device)
        latent_tok = self.latent_proj(z).unsqueeze(1)

        # 2. Vision & Joint Encoding
        x_seq = rgb_seq.view(B*S, C, H, W)
        with torch.no_grad(): feat = self.backbone(x_seq)
        rgb_tok = self.rgb_proj(self.spatial_softmax(feat).view(B, S, -1))
        
        with torch.no_grad(): goal_feat = self.backbone(goal_image)
        goal_tok = self.rgb_proj(self.spatial_softmax(goal_feat)).unsqueeze(1)
        
        joint_tok = self.joint_proj(joint_seq)
        
        # 3. Fuse & Assemble
        enc_tok = self.fusion_proj(torch.cat([rgb_tok, joint_tok], dim=-1))
        enc_tok = torch.cat([latent_tok, enc_tok, goal_tok], dim=1) # [Latent, Obs..., Goal]
        enc_tok = enc_tok + self.enc_pos.unsqueeze(0)

        # 4. Decode
        q = self.action_queries.unsqueeze(0).expand(B, -1, -1) + self.dec_pos.unsqueeze(0)
        out = self.transformer(enc_tok, q)
        return self.head(out), (mu, logvar)