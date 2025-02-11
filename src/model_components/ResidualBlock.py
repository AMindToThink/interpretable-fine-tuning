import torch
from .BiasOnly import BiasOnly
from .LoRALinear import LoRALinear

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_layers:int, hidden_dim:int|None=None, activation=torch.nn.ReLU(), name:str=""):
        """A flexible residual neural network block that maintains input/output dimension compatibility.
    
        This block implements a residual connection of the form output = F(x) + x, where F is a configurable
        neural network. The architecture supports various depths and can degenerate to a simple bias-only layer.
        
        Args:
            input_dim (int): Dimension of input features. Must be positive. Output will have same dimension.
            hidden_layers (int): Number of hidden layers in the network.
                * -1: Creates a bias-only layer
                * 0: Single linear transformation
                * >0: Creates that many hidden layers with activation functions between them
            hidden_dim (int, optional): Dimension of hidden layers. If None, uses input_dim.
            lora_r (int, optional): The rank of the matrices for the FFNN. Depth must be greater than -1.
            activation (torch.nn.Module): Activation function to use between layers. Defaults to ReLU.
            name (str): Optional name for the block. Defaults to empty string.
        
        Example:
            >>> block = ResidualBlock(input_dim=512, hidden_layers=2, hidden_dim=1024)
            >>> x = torch.randn(32, 512)  # batch_size=32, features=512
            >>> output = block(x)  # Shape: (32, 512)
        """
        super().__init__()
        self.name = name
        assert input_dim > 0
        assert hidden_layers >= -1
        assert hidden_dim is None or hidden_dim > 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else input_dim
        self.activation = activation
        sequential = []
        if hidden_layers == -1:
            sequential.append(BiasOnly(input_dim))
        else:
            input_dims = [self.input_dim] + [self.hidden_dim] * hidden_layers
            output_dims = [self.hidden_dim] * hidden_layers + [self.input_dim]
            for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)): # plus one because zero hidden layers is just a forward from input to output
                linear = torch.nn.Linear(in_dim, out_dim)
                if i == len(input_dims) - 1:  # Final layer
                    # Initialize final layer to zero for identity function behavior
                    torch.nn.init.zeros_(linear.weight)
                    torch.nn.init.zeros_(linear.bias)
                else:
                    # Xavier initialization for hidden layers
                    torch.nn.init.xavier_uniform_(linear.weight)
                    torch.nn.init.zeros_(linear.bias)
                sequential.append(linear)
                if i < hidden_layers - 1:
                    sequential.append(activation)
        self.sequential = torch.nn.Sequential(*sequential)
    
    def forward(self, x):
        return self.sequential(x) + x

if __name__ == '__main__':
    # Test basic functionality
    input_dim = 5
    batch_size = 3
    
    # Create test input
    x = torch.randn(batch_size, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Test bias-only case (hidden_layers = -1)
    bias_only = ResidualBlock(input_dim=input_dim, hidden_layers=-1)
    output_bias = bias_only(x)
    print(f"\nBias-only case:")
    print(f"Output shape: {output_bias.shape}")
    print(f"Residual difference: {torch.norm(output_bias - x)}")  # Should be small initially due to zero init
    
    # Test single linear layer case (hidden_layers = 0)
    single_layer = ResidualBlock(input_dim=input_dim, hidden_layers=0)
    output_single = single_layer(x)
    print(f"\nSingle layer case:")
    print(f"Output shape: {output_single.shape}")
    print(f"Residual difference: {torch.norm(output_single - x)}")  # Should be small initially due to zero init
    
    # Test multi-layer case with different hidden dimension
    hidden_dim = 10
    multi_layer = ResidualBlock(input_dim=input_dim, hidden_layers=2, hidden_dim=hidden_dim)
    output_multi = multi_layer(x)
    print(f"\nMulti-layer case:")
    print(f"Output shape: {output_multi.shape}")
    print(f"Residual difference: {torch.norm(output_multi - x)}")  # Should be small initially due to zero init
    import pdb;pdb.set_trace()

