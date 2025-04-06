import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConvergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        y_true_flat = y_true.view(batch_size, -1)
        y_pred_flat = y_pred.view(batch_size, -1)
        
        numerator = torch.norm(y_true_flat - y_pred_flat, p=2, dim=1)
        denominator = torch.norm(y_true_flat, p=2, dim=1)
        
        loss = numerator / (denominator + 1e-8)
        
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.spectral_loss = SpectralConvergenceLoss()
    
    def forward(self, y_pred, y_true):
        l1 = self.l1_loss(y_pred, y_true)
        sc = self.spectral_loss(y_pred, y_true)
        
        return self.alpha * l1 + self.beta * sc