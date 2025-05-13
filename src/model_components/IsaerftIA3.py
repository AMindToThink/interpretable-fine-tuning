import torch
from typing import Callable

def grad_1_abs(x):
    """A version of absolute value which has a gradient of 1 when x_i = 0 instead of 0. Necessary so that we can initialize our parameters to 0 without losing all our gradients."""
    return torch.where(x < 0, -x, x)

class IsaerftIA3(torch.nn.Module):
    def __init__(self, num_features:int, name:str="", scale_processor:Callable[[torch.Tensor], torch.Tensor]=lambda x:x):
        """Scales the features given to it by multiplying elementwise by a learned vector, scaling_factors. Naturally, scaling_factors is initialized to 1s so that this component acts as the identity during the start of training.
        
        Args:
            num_features (int): Dimension of the SAE, which is the dimension of the input features. Must be positive integer. Output will have same dimension.
            scale_processor: Function which processes the scales and returns new scales. Useful for ensuring that only some values are possible (for example, only positive or only negative). For example, making it torch.abs would break because its derivative is 0 at x=0. 
        
        Example:
            >>> layer = IsaerftIA3(num_features=512)  # Create layer with 512 features
            >>> x = torch.randn(32, 512)  # batch_size=32, features=512
            >>> output = layer(x)  # Shape: (32, 512), initially equal to x since scaling_factors starts as ones

        """
        super().__init__()
        self.name = name
        self.scale_processor = scale_processor
        assert num_features > 0 and int(num_features) == num_features, "SAEs have a positive integer number of features. Probably a number like 16k or 32k."
        self.num_features = num_features
        self.scaling_factors = torch.nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        return (self.scale_processor(self.scaling_factors) + 1.0) * x

if __name__ == '__main__':
    # Test basic functionality
    input_dim = 5
    batch_size = 3
    
    # Create test input
    x = torch.randn(batch_size, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Test bias-only case (hidden_layers = -1)
    # Test IA3 layer
    ia3_layer = IsaerftIA3(num_features=input_dim)
    output_ia3 = ia3_layer(x)
    print(f"\nIA3 case:")
    print(f"Output shape: {output_ia3.shape}")
    print(f"Initial output equals input (since scaling starts at 1):", torch.allclose(output_ia3, x))
    
    # Test that scaling works as expected
    ia3_layer.scaling_factors.data.fill_(2.0)  # Set all scaling factors to 2
    output_scaled = ia3_layer(x)
    print(f"After scaling by 2, output equals 2*input:", torch.allclose(output_scaled, 2*x))

