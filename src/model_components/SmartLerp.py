import torch
import torch.nn as nn

class SmartLerp(nn.Module):
    def __init__(self, dim:int, device:torch.device=None, dtype:torch.dtype=None):
        """
        SmartLerp is a differentiable (and therefore optimizable) interpolation function to a learned point. 

        params:
            dim:int the size of the vector that will be passed into it
            device: where to put the parameters (pytorch)
            dtype: what data type to use (pytorch)

        Wow, I feel smart that I realized I can multiply by 2 to have my initial identity funciton and my nonzero gradients. -MK

        Notes: It might be interesting to break target_value into a mean_target_value and std_target_value, so that occasionally a good target would happen and the training would learn to target higher values (at the moment, there's a local minima where components that are optimized by targeting at or near zero are targeted there but values that have an optimal target value at a higher value (eg 100) are never activated. The loss landscape here is interesting.
        """
        super().__init__()
        self.dim = dim
        # Initialize t_logits to zeros so that the sigmoid has its largest derivative, 0.25
        self.t_logits = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        # Initialize target values to zero because I see no reason to choose differently
        self.target_value = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
    
    def forward(self, x):
        # sigmoid of t_logits so that the values are between 0 and 1 (lerping a negative amount or more than 1 would be weird and harder to interpret. This is plenty expressive)
        t = torch.sigmoid(self.t_logits)

        # multiply target value by .5 to cancel out the times 2.
        # return 2 * (x + t * (.5 * self.target_value - x))
        # I did some experiments with a plot and I think that this has some better properties than the original (it actually goes from the start (when x = .5) to the target (x = 1)), but lacks the gradients through target_value :( I still think this is an improvement, since there are still gradients through t.
        return (x + 2*(t - .5) * (self.target_value - x))

def test_smart_lerp():
    # Create SmartLerp instance
    dim = 10
    lerp = SmartLerp(dim)
    
    # Create random input vector
    x = torch.randn(dim, requires_grad=True)
    
    # Pass through SmartLerp
    y = lerp(x)
    print("Checking identity property")
    # Check it's initially identity function
    assert torch.allclose(x, y), "SmartLerp should initially be identity function"
    
    # Take backward pass
    loss = y.sum()
    loss.backward()
    
    print("Checking gradients exist")
    # Check gradients exist
    assert lerp.t_logits.grad is not None, "t_logits should have gradients"
    assert lerp.target_value.grad is not None, "target_value should have gradients"
    # Check gradients aren't all zeros
    print("Checking gradients aren't zero")
    assert not torch.allclose(lerp.t_logits.grad, torch.zeros_like(lerp.t_logits.grad)), "t_logits gradients shouldn't be all zeros"
    # Take an optimizer step
    optimizer = torch.optim.Adam(lerp.parameters(), lr=1)
    optimizer.step()
    assert not torch.allclose(lerp.t_logits, torch.zeros_like(lerp.t_logits)), "Why is t_logits still zero when there were gradients through them?"
    y = lerp(x)
    loss = y.sum()
    loss.backward()
    assert not torch.allclose(lerp.target_value.grad, torch.zeros_like(lerp.target_value.grad)), "target_value gradients shouldn't be all zeros"
    
if __name__ == "__main__":
    test_smart_lerp()
