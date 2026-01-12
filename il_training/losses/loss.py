import torch
import torch.nn as nn

class ACTLoss(nn.Module):
    def __init__(self, lambda_cont=0.05, lambda_goal=0.2):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_goal = lambda_goal
        self.mse = nn.MSELoss()

    def forward(self, pred_deltas, target_deltas, current_joints, target_goal_joints):
        """
        pred_deltas: (B, F, 6) - Model Output
        target_deltas: (B, F, 6) - Ground Truth
        current_joints: (B, 6) - Joint state at t=0 of prediction
        target_goal_joints: (B, 6) - The final goal joint configuration
        """
        
        # 1. Reconstruction Loss (Trajectory matching)
        loss_mse = self.mse(pred_deltas, target_deltas)

        # 2. Continuity Loss (Penalize jumping at step 0)
        # We want the first action to be small or consistent
        loss_cont = self.mse(pred_deltas[:, 0, :], torch.zeros_like(pred_deltas[:, 0, :]))

        # 3. Goal Loss (Did we end up at the right place?)
        # Predict where the robot is at the final future step
        # Note: We assume deltas are summed over the horizon or relative to current state.
        # For simplicity in this implementation, we check the final step delta.
        pred_final_pose = current_joints + pred_deltas[:, -1, :]
        loss_goal = self.mse(pred_final_pose, target_goal_joints)

        total_loss = loss_mse + (self.lambda_cont * loss_cont) + (self.lambda_goal * loss_goal)
        
        return total_loss, {"mse": loss_mse, "cont": loss_cont, "goal": loss_goal}