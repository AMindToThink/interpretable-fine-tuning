# Standard imports
import os
import torch
from typing import Callable
from torch import Tensor
from torch.nn import Linear
# Odd imports
from NeuronpediaClient import NeuronpediaClient
from sae_lens import SAE
# Own code
from model_components.LoRALinear import LoRALinear
from model_components.ResidualBlock import ResidualBlock
from model_components.BiasOnly import BiasOnly
class ISaeRFT_Interpreter():
    """
        Class that turns the learned parameters into human-readable algorithms and outputs.

        Args:
            # sae_id (str): The layer/SAE ID (e.g., '0-res-jb')
            neuronpedia_api_key (str | None, optional): API key for Neuronpedia. If None, uses environment variable. Defaults to None.

        Predicted help:
        If your outputs don't make sense, try checking that the release_id and sae_id are correct and applying L1 loss to the training to encourage sparsity.
    """
    def __init__(self, model_id:str, layer:str, sae:SAE|None=None, release:str='', sae_id:str='', neuronpedia_api_key:str|None=None):
        assert (sae is None) != (release and sae_id), "Must either give an sae or a release and sae_id and not both."
        # assert not (release and sae_id) or sae_id.split('/')[0] == layer, "Clearly one of us doesn't understand the relationship between sae_id and layer. Please check!"
        self.model_id = model_id
        # self.release_id = release_id
        self.layer = layer
        self.neuronpedia_client = NeuronpediaClient(api_key=os.environ['NEURONPEDIA_API_KEY'] if neuronpedia_api_key is None else neuronpedia_api_key)
        self.sae = sae if sae is not None else SAE.from_pretrained(release=release, sae_id=sae_id, device='cpu') # If vRAM becomes an issue, could try putting this on cpu instead, since it isn't involved in much .

    @property
    def bias_interpretation_types(self):
        return frozenset({'absolute', 'L2'})
    
    @property
    def name_importance_map(self):
        return {'absolute': torch.abs, 'L2': self.L2Importance}
    
    def L2Importance(self, vector: Tensor) -> Tensor:
        broadcasted = self.sae.W_dec * vector # broadcast together
        importance = broadcasted.norm(dim=1)
        return importance

    def interpret_bias(self, vector:Tensor, interpretation_type='expectation', top_k:int | None=20):
        """
        Interpretation for BiasOnly biases or Residual blocks where hidden_layers = -1.
        
        Args:
            vector (Tensor): The bias vector of the linear transformation. Shape should be rank 1 (latent_dim).

        protip for testing: gemmascope's res-16k means 16,384 features 
        """
        # Decided to go with the vectors instead of the components.
        # assert len(bias.shape) == 1, "Can only interpret bias vectors with interpret_bias. That means the shape input must be rank 1."
        assert interpretation_type in self.bias_interpretation_types, f"No such interpretation type '{interpretation_type}'. Must be in {self.bias_interpretation_types}"
        assert len(vector.shape) == 1, "Can only interpret bias vectors with interpret_bias. That means the shape input must be rank 1." 
        assert self.sae.cfg.d_sae == len(vector), "The dimension of the sae and the length of the bias vector do not match."

        importance = self.name_importance_map[interpretation_type](vector)
        sorted_indices = torch.argsort(importance, descending=True)
        # if interpretation_type == 'absolute':
        #     # Sort indices by absolute values in descending order
        #     sorted_indices = torch.argsort(torch.abs(vector), descending=True)
        
        # if interpretation_type == 'L2':
        #     broadcasted = self.sae.W_dec * vector # broadcast together
        #     importance = broadcasted.norm(dim=1)
        #     sorted_indices = torch.argsort(importance, descending=True)
        # if interpretation_type == 'expectation':
        #     for i, value in enumerate(vector):
        #         client_json = self.neuronpedia_client.get_feature(model_id=self.model_id, layer=self.layer, index=i)
        #         freq_hist_data_bar_values = client_json['freq_hist_data_bar_values']
        #         freq_hist_data_bar_heights = client_json['freq_hist_data_bar_heights']
                

        #     raise NotImplementedError("Need to figure out how to manage this long-tail-looking distribution")
        
        indices_to_check = sorted_indices[:top_k]
        feature_results = []
        for rank, index in enumerate(indices_to_check):
            client_json = self.neuronpedia_client.get_feature(model_id=self.model_id, layer=self.layer, index=sorted_indices[rank])
            feature_results.append({'rank':rank, 'index':index, 'value':vector[index], 'explanation': client_json['explanations'][0]['description']})

        return feature_results

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

