import torch
import torch.nn as nn

class ACTCVAELoss(nn.Module):
    def __init__(self, beta: float = 0.01, smoothness_weight: float = 0.0):
        """
        Args:
            beta: Weight for KL Divergence.
                  Low (e.g., 0.01) = Prioritizes Reconstruction accuracy (better tracking).
                  High (e.g., 1.0) = Prioritizes Latent structure (better generation diversity).
            smoothness_weight: Weight for penalizing jerky movements. 
                               Recommended ~0.01 for cinema robots.
        """
        super().__init__()
        self.beta = beta
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() # L1 is often better for smoothness constraints

    def forward(self, pred_delta, future_delta, mu, logvar):
        """
        Args:
            pred_delta:   (B, Future_Steps, 6) Predicted change in joint angles
            future_delta: (B, Future_Steps, 6) Ground Truth change in joint angles
            mu:           (B, Latent_Dim) Mean from CVAE Encoder
            logvar:       (B, Latent_Dim) Log Variance from CVAE Encoder
        """
        
        # --- 1. Reconstruction Loss (Accuracy) ---
        # "Did the robot move to the right place?"
        loss_mse = self.mse(pred_delta, future_delta)

        # --- 2. KL Divergence Loss (Regularization) ---
        # "Is the latent space normally distributed?"
        # This prevents the model from "cheating" by encoding the exact answer in z.
        # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean() # Average over batch

        # --- 3. Smoothness Loss (Cinema Quality) ---
        # "Is the movement fluid?"
        # Since 'pred_delta' represents velocity (change per step), 
        # the difference between consecutive deltas represents ACCELERATION.
        # We want to minimize unnecessary acceleration spikes.
        loss_smooth = 0.0
        if self.smoothness_weight > 0:
            # Calculate acceleration: (Vel_t+1 - Vel_t)
            accel = pred_delta[:, 1:, :] - pred_delta[:, :-1, :]
            # Force acceleration towards 0 (constant velocity motion preference)
            loss_smooth = self.l1(accel, torch.zeros_like(accel))

        # --- Total Loss ---
        total_loss = loss_mse + (self.beta * kl_loss) + (self.smoothness_weight * loss_smooth)

        loss_dict = {
            "loss": total_loss.item(),
            "mse": loss_mse.item(),
            "kl": kl_loss.item(),
            "smooth": loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict