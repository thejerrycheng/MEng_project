import torch
import torch.nn as nn

# -------------------------
# ACT / Vanilla BC Loss
# -------------------------

class ACTLoss(nn.Module):
    def __init__(self, lambda_cont=0.0):
        """
        Args:
            lambda_cont: Weight for continuity loss (punishing the first action for not being 0).
                         Usually 0.0 or very small (e.g. 0.01) for pure BC.
        """
        super().__init__()
        self.lambda_cont = lambda_cont
        self.mse = nn.MSELoss()

    def forward(self, pred_delta, future_delta):
        """
        Args:
            pred_delta:   (B, Future_Steps, 6) Predicted change in joint angles
            future_delta: (B, Future_Steps, 6) Ground Truth change in joint angles
            
            Note: We do NOT pass the goal_image here. The goal_image is an INPUT 
            to the model, not a target for the loss function.
        """
        
        # 1. Imitation Loss (MSE on Action Chunk)
        # "Make the predicted trajectory match the expert's trajectory"
        loss_action = self.mse(pred_delta, future_delta)

        # 2. Continuity Loss (Optional)
        # Forces the first predicted delta to be close to 0 to ensure smooth start.
        # This is useful if your controller adds these deltas to the CURRENT state immediately.
        loss_cont = 0.0
        if self.lambda_cont > 0:
            loss_cont = self.mse(pred_delta[:, 0, :], torch.zeros_like(pred_delta[:, 0, :]))

        total_loss = loss_action + (self.lambda_cont * loss_cont)

        return total_loss, {"mse": loss_action.item(), "cont": loss_cont}