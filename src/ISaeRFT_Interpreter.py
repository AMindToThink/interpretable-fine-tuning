import torch
from torch import Tensor
from torch.nn import Linear
from model_components.LoRALinear import LoRALinear
from model_components.ResidualBlock import ResidualBlock
from model_components.BiasOnly import BiasOnly
interpretation_types = {'expectation', 'absolute'}
class ISaeRFT_Interpreter():
    """
        Class that turns the learned parameters into human-readable algorithms and outputs.
        
        Predicted help:
        If your outputs don't make sense, try checking that the release_id and sae_id are correct and applying L1 loss to the training to encourage sparsity.
    """
    def __init__(self, release_id:str, sae_id:str):
        self.release_id = release_id
        self.sae_id = sae_id

    def interpret_bias(self, bias:BiasOnly, interpretation_type='expectation', top_k:int | None=20):
        """
        Interpretation for BiasOnly biases or Residual blocks where hidden_layers = -1.
        
        Args:
            bias (Tensor): The bias vector of the linear transformation. Shape should be rank 1 (latent_dim).
        """
        # Decided to go with taking the components instead of taking the vectors.
        # assert len(bias.shape) == 1, "Can only interpret bias vectors with interpret_bias. That means the shape input must be rank 1."
        assert interpretation_type in interpretation_types, f"No such interpretation type '{interpretation_type}'. Must be in {interpretation_types}"
        vector = bias.bias.data

        if interpretation_type == 'absolute':
            # Sort indices by absolute values in descending order
            sorted_indices = torch.argsort(torch.abs(vector), descending=True)
        
        if interpretation_type == 'expectation':
            raise NotImplementedError("Need to make a request to Neuronpedia's get feature and look at hist_data")
        
        for feature in range(top_k):
            pass

        raise NotImplementedError()

    def interpret_linear(self,  linear:Linear | LoRALinear, interpretation_type='expectation'):
        """
        Interpretation for ResidualBlock with hidden_layers = 0.

        Args:
            linear (Linear | LoRALinear): Interpret a matrix multiplication. Shape should be square (latent_dim, latent_dim).
        """
        assert interpretation_type in interpretation_types, f"No such interpretation type '{interpretation_type}'. Must be in {interpretation_types}"
        raise NotImplementedError()
    
    def interpret_1_hidden_layerFFNN(self, linear1:Linear | LoRALinear, linear2:Linear | LoRALinear, interpretation_type='expectation'):
        """
        Interpretation for ResidualBlock with hidden_layers = 1. 

        Args:
            linear1: (torch.nn.Linear | LoRALinear)
            linear2: (torch.nn.Linear | LoRALinear)
        """
        assert interpretation_type in interpretation_types, f"No such interpretation type '{interpretation_type}'. Must be in {interpretation_types}"

        bias_interpretation = self.interpret_bias(linear2.bias())
        raise NotImplementedError()
    
    def interpret_ResidualBlock(self, rb:ResidualBlock, interpretation_type='expectation'):
        assert interpretation_type in interpretation_types, f"No such interpretation type '{interpretation_type}'. Must be in {interpretation_types}"

        if rb.hidden_layers == -1:
            return self.interpret_bias(rb.sequential[0].bias.data, interpretation_type=interpretation_type)
        if rb.hidden_layers == 0:
            return self.interpret_linear(rb.sequential[0], interpretation=interpretation_type)
        if rb.hidden_layers == 1:
            return self.interpret_1_hidden_layerFFNN(rb.sequential[0], rb.sequential[1], interpretation_type=interpretation_type)
        raise ValueError("This ResidualBlock is too deep! Interpretations are only available for rb with hidden layers in [-1,0,1]. If you have a way to interpret deeper networks, feel free to implement!")

