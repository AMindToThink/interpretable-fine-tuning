from transformers.trainer_callback import TrainerCallback

# class ParameterPlottingCallback(TrainerCallback):
#     def on_log(self, args, state, control, model, logs=None, **kwargs):

import wandb
import torch

class PEFTParameterTrackingCallback(TrainerCallback):
    """
    A callback that tracks actual parameter values for parameter-efficient fine-tuning methods.
    Designed for small adapter models where seeing individual parameter values is valuable.
    Only tracks parameters that have requires_grad=True.
    """
    
    def __init__(self, peft_param_prefix=None, max_individual_values=50):
        """
        Args:
            peft_param_prefix (list, optional): List of parameter name prefixes to track.
                If None, will try to auto-detect PEFT parameters using common prefixes.
            max_individual_values (int): Maximum number of individual values to track per parameter.
        """
        self.peft_param_prefix = peft_param_prefix or ["lora", "adapter", "prefix", "prompt"]
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
                # Get the actual parameter values
                param_data = param.data.detach().cpu().flatten().tolist()
                
                # Store param shape information
                original_shape = list(param.shape)
                param_values[f"peft_params/{name}/shape"] = str(original_shape)
                
                # For visualization purposes, also log the full parameter as array
                param_values[f"peft_params/{name}/values"] = param_data
                
                # For small parameters, also log individual values for better tracking
                if len(param_data) <= self.max_individual_values:
                    for i, value in enumerate(param_data):
                        param_values[f"peft_params/{name}/{i}"] = value
                
                # Log gradient information if available
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().flatten().tolist()
                    param_values[f"peft_grads/{name}/values"] = grad_data
                    
                    # Log individual gradient values for small parameters
                    if len(grad_data) <= self.max_individual_values:
                        for i, value in enumerate(grad_data):
                            param_values[f"peft_grads/{name}/{i}"] = value
        
        # Log the parameter values to wandb
        wandb.log(param_values, step=state.global_step)

        # Log the total number of trainable parameters (this is useful information)
        if state.global_step == 0 or not hasattr(self, 'logged_trainable_params'):
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_values["trainable_parameters"] = trainable_params
            self.logged_trainable_params = True