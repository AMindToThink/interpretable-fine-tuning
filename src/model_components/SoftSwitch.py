import torch
import torch.nn as nn

class SoftSwitch(nn.Module):
    def __init__(self, a=0.0, b=1.0, k=1.0, trainable=True):
        """
        Initializes the smooth transition module.
        
        Args:
            a (float): Threshold parameter. Default is 0.0.
            b (float): Value to output when x is above the threshold. Default is 1.0.
            k (float): Steepness of the transition. Default is 1.0.
            trainable (bool): Whether to make a, b, and k trainable. Default is True.
        """
        super(SoftSwitch, self).__init__()
        
        if trainable:
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
            self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        else:
            # Use buffers if parameters should remain fixed
            self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
            self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
            self.register_buffer('k', torch.tensor(k, dtype=torch.float32))
    
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

# Example usage:
if __name__ == "__main__":
    # Create a sample input tensor
    x = torch.linspace(-5, 5, steps=100)
    # Initialize the module
    module = SoftSwitch(a=0.0, b=2.0, k=3.0)
    # Compute the output
    y = module(x)

    # Create and save plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y.detach().numpy())
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SoftSwitch Function')
    plt.savefig('soft_switch_plot.png')
    plt.close()