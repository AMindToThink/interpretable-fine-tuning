import torch
import BiasOnly

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_layers:int, hidden_dim:int|None=None, activation=torch.nn.ReLU()):
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
            activation (torch.nn.Module): Activation function to use between layers. Defaults to ReLU.
        
        Example:
            >>> block = ResidualBlock(input_dim=512, hidden_layers=2, hidden_dim=1024)
            >>> x = torch.randn(32, 512)  # batch_size=32, features=512
            >>> output = block(x)  # Shape: (32, 512)
        """
        super().__init__()
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