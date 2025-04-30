import wandb
import torch
import numpy as np
from transformers import TrainerCallback

class PEFTParameterHistogramCallback(TrainerCallback):
    """
    A callback that tracks actual parameter values for parameter-efficient fine-tuning methods.
    Only tracks parameters that have requires_grad=True.
    """
    
    def __init__(self, peft_param_prefix=None, max_individual_values=50):
        """
        Args:
            peft_param_prefix (list, optional): List of parameter name prefixes to track.
                If None, will try to auto-detect PEFT parameters using common prefixes.
            max_individual_values (int): Maximum number of individual values to track per parameter.
        """
        self.peft_param_prefix = peft_param_prefix or ["lora", "adapter", "prefix", "prompt", "ia3"]
        self.max_individual_values = max_individual_values
    
    def _is_peft_param(self, param_name):
        """Check if parameter is a PEFT parameter based on naming."""
        return any(prefix in param_name.lower() for prefix in self.peft_param_prefix)
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log actual parameter values when logging occurs."""
        if not model or not state.is_world_process_zero or not wandb.run:
            return
            
        # Dictionary to store parameter values
        param_values = {}
        
        # Collect all trainable PEFT parameter values
        for name, param in model.named_parameters():
            # Only track parameters that:
            # 1. Are PEFT parameters (based on name)
            # 2. Have requires_grad=True (are being trained)
            if self._is_peft_param(name) and param.requires_grad:
                # Get the actual parameter values as numpy array
                param_data = param.data.detach().cpu().numpy().flatten()
                
                # Store param shape information
                original_shape = list(param.shape)
                param_values[f"peft_params/{name}/shape"] = str(original_shape)
                
                # Use histogram instead of raw values list
                param_values[f"peft_params/{name}/histogram"] = wandb.Histogram(param_data)
                
                # For small parameters, also log individual values for better tracking
                if len(param_data) <= self.max_individual_values:
                    for i, value in enumerate(param_data):
                        param_values[f"peft_params/{name}/value_{i}"] = float(value)
                
                # Log summary statistics as well
                param_values[f"peft_params/{name}/mean"] = float(np.mean(param_data))
                param_values[f"peft_params/{name}/std"] = float(np.std(param_data))
                param_values[f"peft_params/{name}/min"] = float(np.min(param_data))
                param_values[f"peft_params/{name}/max"] = float(np.max(param_data))
                
                # Log gradient information if available
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy().flatten()
                    param_values[f"peft_grads/{name}/histogram"] = wandb.Histogram(grad_data)
                    
                    # Log individual gradient values for small parameters
                    if len(grad_data) <= self.max_individual_values:
                        for i, value in enumerate(grad_data):
                            param_values[f"peft_grads/{name}/value_{i}"] = float(value)
                    
                    # Log gradient summary statistics
                    param_values[f"peft_grads/{name}/mean"] = float(np.mean(grad_data))
                    param_values[f"peft_grads/{name}/std"] = float(np.std(grad_data))
        
        # Log the parameter values to wandb
        wandb.log(param_values, step=state.global_step)

        # Log the total number of trainable parameters
        if state.global_step == 0 or not hasattr(self, 'logged_trainable_params'):
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.log({"trainable_parameters": trainable_params}, step=state.global_step)
            self.logged_trainable_params = True