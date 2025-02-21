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
    def __init__(self, sae:SAE, neuronpedia_api_key:str|None=None):
        """
        params:
            sae: here's an example: SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_25/width_16k/canonical", device='cpu')[0] # It is just one matmul, cpu is fine here. Takes .08 seconds
        """
        self.model_id = sae.cfg.model_name
        self.neuronpedia_client = NeuronpediaClient(api_key=os.environ['NEURONPEDIA_API_KEY'] if neuronpedia_api_key is None else neuronpedia_api_key)
        self.sae = sae # If vRAM becomes an issue, could try putting this on cpu instead, since it isn't involved in much .

    @property
    def bias_interpretation_types(self):
        return frozenset({'absolute', 'L2'})
    
    @property
    def name_importance_map(self):
        return {'absolute': torch.abs, 'L2': self.L2Importance}
    
    def L2Importance(self, vector: Tensor) -> Tensor:
        # import pdb;pdb.set_trace()
        broadcasted = self.sae.W_dec.T * vector # broadcast together
        importance = broadcasted.norm(dim=0)
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
        if 'gemma-2' in self.sae.cfg.neuronpedia_id.split('/')[0] and interpretation_type == "L2":
            print("Warning: the following sae has decryption matrices that have rows norming to 1, so using L2 and using absolute are the same. Maybe something you should check? gemma-scope-2b-pt-res-canonical")
        import time
        start = time.time()
        print("starting importance calculation")
        importance = self.name_importance_map[interpretation_type](vector)
        print(f"end calculation {time.time() - start}")
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
            client_json = self.neuronpedia_client.get_feature(neuronpedia_id=self.sae.cfg.neuronpedia_id, index=sorted_indices[rank])
            feature_results.append({'interpretation_type':interpretation_type, 'rank':rank, 'index':index.item(), 'value':vector[index].item(), 'importance':importance[index].item(), 'explanation': client_json['explanations'][0]['description']})

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

if __name__ == "__main__":
    mysae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_25/width_16k/canonical", device='cpu')[0]
    myinter = ISaeRFT_Interpreter(mysae)
    d_sae = mysae.cfg.d_sae
    for i in range(4):
        v = torch.randn(d_sae)
        interpretations = myinter.interpret_bias(v, 'L2', 3)
        print(interpretations)
        interpretations = myinter.interpret_bias(v, 'absolute', 3)
        print(interpretations)
    # import pdb;pdb.set_trace()