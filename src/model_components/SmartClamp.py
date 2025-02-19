import torch
import torch.nn as nn

class SmartClamp(nn.Module):
    def __init__(self, dim:int, device:torch.device=None, dtype:torch.dtype=None):
        """
        SmartClamp is a differentiable (and therefore optimizable) clamping function. 

        params:
            dim:int the size of the vector that will be passed into it
            device: where to put the parameters (pytorch)
            dtype: what data type to use (pytorch)

        Wow, I feel smart that I realized I can multiply by 2 to have my initial identity funciton and my nonzero gradients. -MK
        """
        super().__init__()
        self.dim = dim
        self.t_logits = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.clamp_value = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
    
    def forward(self, x):
        return 2 * (x + torch.sigmoid(self.t_logits) * (self.clamp_value - x))