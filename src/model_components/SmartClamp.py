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
        # Initialize t_logits to zeros so that the sigmoid has its largest derivative, 0.25
        self.t_logits = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        # Initialize clamp values to zero because I see no reason to choose differently
        self.clamp_value = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
    
    def forward(self, x):
        # sigmoid of t_logits so that the values are between 0 and 1 (lerping a negative amount or more than 1 would be weird and harder to interpret. This is plenty expressive)
        t = torch.sigmoid(self.t_logits)
        return 2 * (x + t * (self.clamp_value - x))

def test_smart_clamp():
    # Create SmartClamp instance
    dim = 10
    clamp = SmartClamp(dim)
    
    # Create random input vector
    x = torch.randn(dim, requires_grad=True)
    
    # Pass through SmartClamp
    y = clamp(x)
    print("Checking identity property")
    # Check it's initially identity function
    assert torch.allclose(x, y), "SmartClamp should initially be identity function"
    
    # Take backward pass
    loss = y.sum()
    loss.backward()
    
    print("Checking gradients exist")
    # Check gradients exist
    assert clamp.t_logits.grad is not None, "t_logits should have gradients"
    assert clamp.clamp_value.grad is not None, "clamp_value should have gradients"
    # Check gradients aren't all zeros
    print("Checking gradients aren't zero")
    assert not torch.allclose(clamp.t_logits.grad, torch.zeros_like(clamp.t_logits.grad)), "t_logits gradients shouldn't be all zeros"
    assert not torch.allclose(clamp.clamp_value.grad, torch.zeros_like(clamp.clamp_value.grad)), "clamp_value gradients shouldn't be all zeros"
    
if __name__ == "__main__":
    test_smart_clamp()
