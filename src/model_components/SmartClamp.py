import torch
import torch.nn as nn

class SmartClamp(nn.Module):
    def __init__(self, dim:int, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.t_logits = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.clamp_value = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
    
    def forward(self, x):
        return 2 * (x + torch.sigmoid(self.t_logits) * (self.clamp_value - x))