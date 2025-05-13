import torch
from torch import nn
from sae_lens import HookedSAETransformer, SAE
from functools import partial
from transformer_lens.hook_points import HookPoint
from model_components.IsaerftIA3 import IsaerftIA3

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
        
        # Create trainable IA3 module
        self.trainable_ia3 = IsaerftIA3(self.sae.cfg.d_sae, "sae_IA3_2025-04-29")
        self.sae.trainable_ia3 = self.trainable_ia3
        
        # Add SAE to model using built-in method
        self.model.add_sae(self.sae)
        
        # Add IA3 hook to the SAE
        self._add_ia3_hook()
        
        # Verify only trainable_ia3 is trainable
        self._verify_trainable_params()
    
    def _add_ia3_hook(self):
        """Add IA3 hook to the SAE"""
        def ia3_hook(tensor: torch.Tensor, *, hook: HookPoint) -> torch.Tensor:
            """Runs the input through a trainable isaerft ia3.

            Args:
                tensor (Tensor): The SAE activations tensor, shape [batch, pos, features]
                hook (HookPoint): The transformer-lens hook point

            Returns:
                Tensor: The modified SAE activations modified by the trainable parameters.
            """
            return self.trainable_ia3(tensor)
            
        self.sae.add_hook('hook_sae_acts_post', ia3_hook)
    
    def _verify_trainable_params(self):
        """Verify that only trainable_ia3 is trainable"""
        trainable_params = [name for name, param in self.named_parameters() if param.requires_grad]
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found in model")
        
        if len(trainable_params) > 1:
            raise ValueError(f"Found multiple trainable parameters: {trainable_params}. Expected only trainable_ia3")
            
        if not any("trainable_ia3" in param_name for param_name in trainable_params):
            raise ValueError(f"trainable_ia3 not found in trainable parameters: {trainable_params}")
    
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
