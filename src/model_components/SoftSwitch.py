import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SoftSwitchInitializationHighA():
    """Class used for initializing the SoftSwitch module with a high a value, which makes the softswitch mostly be an identity function to start."""
    a:int = 200
    
class SoftSwitch(nn.Module):
    def __init__(self, dim:int, initialization, device=None, dtype=None):
        """
        Initializes the smooth transition module.
        
        Args:
            dim:int The size of the vector to process.
            initialization: Initialization configuration object
            device: Device to place tensors on. Default: None
            dtype: Data type for tensors. Default: None
        """
        super(SoftSwitch, self).__init__()
        self.dim = dim
        if isinstance(initialization, SoftSwitchInitializationHighA):
            self.a = nn.Parameter(torch.full((dim,), initialization.a, dtype=dtype, device=device))
            self.b = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))
            self.k = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))
        else:
            raise NotImplementedError("No initialization types other than SoftSwitchInitializationHighA are implemented")
    
    def forward(self, x):
        """
        Forward pass of the smooth transition function.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after applying the smooth transition.
        """
        # Compute smooth step using the sigmoid function.
        smooth_step = torch.sigmoid(self.k * (x - self.a))
        # Return the smoothly transitioned value.
        return x + (self.b - x) * smooth_step

