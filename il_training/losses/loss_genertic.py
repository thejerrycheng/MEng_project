import torch
import torch.nn as nn

# --- 1. Deterministic Loss (For Absolute Position Models) ---
class AbsoluteMotionLoss(nn.Module):
    def __init__(self, smoothness_weight: float = 0.05):
        """
        Optimized for: Transformer_Absolute, Transformer_Visual_Absolute, RGB_Only
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() 

    def forward(self, pred_pos, target_pos):
        # 1. Position Accuracy
        loss_mse = self.mse(pred_pos, target_pos)

        # 2. Smoothness Regularization
        loss_smooth = 0.0
        if self.smoothness_weight > 0:
            # First Derivative (Velocity approx)
            velocity = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
            # Second Derivative (Acceleration approx)
            accel = velocity[:, 1:, :] - velocity[:, :-1, :]
            
            loss_smooth = self.l1(accel, torch.zeros_like(accel))

        # Total Loss
        total_loss = loss_mse + (self.smoothness_weight * loss_smooth)

        loss_dict = {
            "loss": total_loss.item(),
            "mse": loss_mse.item(),
            "smooth": loss_smooth.item() if self.smoothness_weight > 0 else 0.0
        }
        
        return total_loss, loss_dict


# --- 2. Probabilistic Loss (For CVAE Models) ---
class ACTCVAELoss(nn.Module):
    def __init__(self, beta: float = 0.01, smoothness_weight: float = 0.01):
        """
        Optimized for: CVAE_Absolute models
        """
        super().__init__()
        self.beta = beta
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, mu, logvar):
        """
        Args:
            pred:   Predicted Actions (Absolute)
            target: Ground Truth Actions
            mu, logvar: Latent distribution params
        """
        # 1. Accuracy
        loss_mse = self.mse(pred, target)

        # 2. KL Divergence (Latent Regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()

        # 3. Smoothness (Minimizing jerk/accel)
        loss_smooth = 0.0
        if self.smoothness_weight > 0:
            diff1 = pred[:, 1:, :] - pred[:, :-1, :]
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
            loss_smooth = self.l1(diff2, torch.zeros_like(diff2))

        total_loss = loss_mse + (self.beta * kl_loss) + (self.smoothness_weight * loss_smooth)

        loss_dict = {
            "loss": total_loss.item(),
            "mse": loss_mse.item(),
            "kl": kl_loss.item(),
            "smooth": loss_smooth.item()
        }
        
        return total_loss, loss_dict