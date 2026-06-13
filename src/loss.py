import torch
import torch.nn as nn

class DirectionalLoss(nn.Module):
    def __init__(self, penalty_weight: float = 5.0):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error_sq = torch.square(y_true - y_pred)
        interaction = y_true * y_pred
        penalty = torch.where(interaction < 0, self.penalty_weight, 1.0)
        return torch.mean(error_sq * penalty)
