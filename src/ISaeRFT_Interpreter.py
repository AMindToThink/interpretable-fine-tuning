import torch
from torch import Tensor
from torch.nn import Linear
from model_components.LoRALinear import LoRALinear
from model_components.ResidualBlock import ResidualBlock

class ISaeRFT_Interpreter():
    """
        Class that turns the learned parameters into human-readable algorithms and outputs.
        
        Predicted help:
        If your outputs don't make sense, try checking that the release_id and sae_id are correct and applying L1 loss to the training to encourage sparsity.
    """
    def __init__(self, release_id:str, sae_id:str):
        self.release_id = release_id
        self.sae_id = sae_id

    def interpret_bias(self, bias:Tensor):
        """
        Interpretation for BiasOnly biases or Residual blocks where hidden_layers = -1.
        
        Args:
            bias (Tensor): The bias vector of the linear transformation. Shape should be rank 1 (latent_dim).
        """
        assert len(bias.shape) == 1, "Can only interpret bias vectors with interpret_bias. That means the shape input must be rank 1."
        raise NotImplementedError()

    def interpret_linear(self, weights:Tensor):
        """
        Interpretation for ResidualBlock with hidden_layers = 0.

        Args:
            weights (Tensor): The weight matrix of the linear transformation. Shape should be square (latent_dim, latent_dim).
        """
        raise NotImplementedError()
    
    def interpret_1_hidden_layerFFNN(self, linear1:Linear | LoRALinear, linear2:Linear | LoRALinear):
        """
        Interpretation for ResidualBlock with hidden_layers = 1. 

        Args:
            linear1: (torch.nn.Linear | LoRALinear)
            linear2: (torch.nn.Linear | LoRALinear)
        """

        bias_interpretation = self.interpret_bias(linear2.bias())
        raise NotImplementedError()
    
    def interpret_ResidualBlock(self, rb:ResidualBlock):
        raise NotImplementedError()
