import torch
from torch import nn
from sae_lens import HookedSAETransformer, SAE
from functools import partial
from transformer_lens.hook_points import HookPoint

class IsaerftIA3Model(nn.Module):
    def __init__(self, model: HookedSAETransformer, sae: SAE):
        super().__init__()
        self.model = model
        self.sae = sae
        
        # Freeze all parameters in the base model and SAE
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.sae.parameters():
            param.requires_grad = False
            
        # Ensure SAE has error term enabled
        assert hasattr(self.sae, 'use_error_term'), "Where's the error term?"
        self.sae.use_error_term = True
        
        # Add SAE to model using built-in method
        self.model.add_sae(self.sae)
    
    def forward(self, *args, **kwargs):
        """Forward pass that handles input format and returns proper output format.
        
        This method follows the same pattern as IsaerftPeft.forward to ensure
        consistent input/output handling.
        """
        # Pop use_cache if present since model doesn't support it
        assert not kwargs.pop('use_cache', None), "Sorry, HookedSAETransformer doesn't have use_cache. Please make it either false or don't give it."
        
        # Handle input_ids
        if "input_ids" in kwargs:
            input_tensor = kwargs.pop("input_ids")
        elif args:
            input_tensor = args[0]
            args = args[1:]
        else:
            input_tensor = None
            
        # Run the model's forward pass
        output = self.model(input_tensor, *args, **kwargs, return_type='both')
        
        return output
