import wandb
import torch
import numpy as np
from transformers.trainer_callback import TrainerCallback

class PEFTParameterTrackingCallback(TrainerCallback):
    """
    A callback that tracks individual parameter values over time for PEFT methods.
    Optimized for tracking small parameter vectors where each element is important.
    """
    
    def __init__(self, peft_param_prefix=None, param_desc=None):
        """
        Args:
            peft_param_prefix (list, optional): List of parameter name prefixes to track.
                If None, will use common PEFT parameter prefixes.
            param_desc (dict, optional): Dictionary mapping param keys to descriptions.
        """
        self.peft_param_prefix = peft_param_prefix or ["lora", "adapter", "prefix", "prompt", "ia3"]
        # Keep track of parameters we've seen to maintain consistent tracking
        self.tracked_params = {}
        self.param_desc = param_desc or {}
    
    def _is_peft_param(self, param_name):
        """Check if parameter is a PEFT parameter based on naming."""
        return any(prefix in param_name.lower() for prefix in self.peft_param_prefix)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize parameter tracking at the beginning of training."""
        if not model or not state.is_world_process_zero:
            return
            
        # Find trainable PEFT parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not self._is_peft_param(name):
                raise ValueError(f'there are non-peft parameters training! {name}')

            param_data = param.data.detach().cpu().numpy().flatten()
            
            # Store initial parameter data for reference
            self.tracked_params[name] = {
                'shape': param.shape,
                'size': param.numel(),
                'indices': list(range(len(param_data)))
            }
            
            # Log initial parameter values
            param_dict = {}
            for i, value in enumerate(param_data):
                base_param_i_name = f'param/{name}/{i}'
                desc = self.param_desc.get(base_param_i_name, "")
                param_name = f"{base_param_i_name}/{desc}"
                param_dict[param_name] = float(value)
            
            wandb.log(param_dict, step=0)
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log individual parameter values at logging steps."""
        if not model or not state.is_world_process_zero or not wandb.run:
            return
            
        # Dictionary to log current parameter values
        param_dict = {}
        
        # Track trainable PEFT parameters that we identified at the beginning
        for name, param in model.named_parameters():
            if name in self.tracked_params and param.requires_grad:
                param_data = param.data.detach().cpu().numpy().flatten()
                
                # Log each individual parameter value
                for i, value in enumerate(param_data):
                    base_param_i_name = f"param/{name}/{i}"
                    desc = self.param_desc.get(base_param_i_name, "")
                    param_name = f"{base_param_i_name}/{desc}"
                    param_dict[param_name] = float(value)
                
                # If parameter has a gradient, track that too
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy().flatten()
                    for i, value in enumerate(grad_data):
                        base_grad_i_name = f"grad/{name}/{i}"
                        desc = self.param_desc.get(f"param/{name}/{i}", "")
                        grad_name = f"{base_grad_i_name}/{desc}"
                        param_dict[grad_name] = float(value)
        
        # Log all the values
        wandb.log(param_dict, step=state.global_step)