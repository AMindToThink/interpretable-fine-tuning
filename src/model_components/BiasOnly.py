import torch
from torch import nn
class BiasOnly(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        return x + self.bias

if __name__ == '__main__':
    example_input = torch.randn(10)
    print(example_input)
    print(example_input.shape)
    # Example usage:
    bias_layer = BiasOnly(features=example_input.shape[0])
    output = bias_layer(example_input)  # Will add a learnable bias to each feature
    print(output)