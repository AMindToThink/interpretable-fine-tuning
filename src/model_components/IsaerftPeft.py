#%%
# from elsewhere
import torch
from torch import nn
from transformers import AutoConfig
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from sae_lens import HookedSAETransformer, SAE
from peft.utils import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from functools import partial

# Handle imports differently based on how the file is being used
try:
    # When imported as a module
    from .ResidualBlock import ResidualBlock
    from .IsaerftConfig import IsaerftConfig
    from .IsaerftIA3 import IsaerftIA3
except ImportError:
    # When run directly as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_components.ResidualBlock import ResidualBlock
    from IsaerftConfig import IsaerftConfig
    from IsaerftIA3 import IsaerftIA3

# 1. Add your PEFT type to the PeftType enum
if not hasattr(PeftType, "ISAERFT"):
    # This adds the ISAERFT type to the PeftType enum
    PeftType.ISAERFT = "ISAERFT"

def resid_hook(sae_acts, hook, residual_block):
    """Runs the input through a trainable resnet (ResidualBlock).

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point
        residual_block (ResidualBlock): The residual block to apply

    Returns:
        Tensor: The modified SAE activations modified by the trainable parameters.
    """
    return residual_block(sae_acts)
#%%


class IsaerftModel(BaseTuner):
    """Implementation of the ISAERFT model"""
    # TODO: would be better if I used a list of hooks plus the context manager instead of permanently adding them. 
    def __init__(self, model, config, adapter_name):
        super().__init__(model=model, peft_config=config, adapter_name=adapter_name)
        self.model = model
        self.device = next(model.parameters()).device
        self.config = AutoConfig.from_pretrained(model.cfg.tokenizer_name)
        self.peft_config = config
        self.active_adapter = adapter_name
        self.warnings_issued = {}
        
        # Store SAEs that have been added to the model
        self.saes = {}
        self.trainable_blocks = nn.ModuleDict()  
        
        # Initialize adapter state
        self._adapter_layers_enabled = True
        
        # Create trainable blocks for each target hook
        self.setup_trainable_blocks()

    # <Required for BaseTuner>
    @staticmethod
    def _prepare_adapter_config(isaerft_config, model_config):
        # I decided this logic didn't look important

        # if peft_config.target_modules is None:
        #     if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
        #         raise ValueError("Please specify `target_modules` in `peft_config`")
        #     peft_config.target_modules = set(
        #         TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        #     )
        return isaerft_config
    
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overridden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            current_key (`str`):
                The key of the current target being adapted.
        """
        return
    def resize_token_embeddings(self, num_new_tokens, *args, **kwargs):
        print(f"{num_new_tokens=}")
        assert num_new_tokens == self.config.vocab_size
    @classmethod
    def _load_pretrained_saes_yaml(cls):
        """Load the pretrained_saes.yaml file from the SAE lens package"""
        try:
            from sae_lens import SAE
            import yaml
            import os
            import importlib.resources as pkg_resources
            
            # Try to load the pretrained_saes.yaml file
            try:
                # First try to get it from the sae_lens package using the newer approach
                try:
                    # For Python 3.9+
                    with pkg_resources.files('sae_lens').joinpath('pretrained_saes.yaml').open('r') as f:
                        pretrained_saes = yaml.safe_load(f)
                except AttributeError:
                    # Fallback for older Python versions
                    yaml_content = pkg_resources.read_text('sae_lens', 'pretrained_saes.yaml')
                    pretrained_saes = yaml.safe_load(yaml_content)
            except Exception:
                # Fallback to looking in common locations
                yaml_paths = [
                    os.path.join(os.path.dirname(SAE.__file__), 'pretrained_saes.yaml'),
                    'pretrained_saes.yaml'
                ]
                for path in yaml_paths:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            pretrained_saes = yaml.safe_load(f)
                        break
                else:
                    raise FileNotFoundError("Could not find pretrained_saes.yaml")
            
            return pretrained_saes
        except Exception as e:
            print(f"Error loading pretrained_saes.yaml: {e}")
            return None
    def _find_potential_sae_matches(self, release_pattern, id_pattern, pretrained_saes):
        try:
            # Find matching SAEs in the yaml file
            model_name = self.model.cfg.model_name.lower()
            
            # First, create a list of all potential matches without importing
            potential_matches = []
            # Find all matching collections based on release pattern and model
            for collection_name, collection_data in pretrained_saes.items():
                if (release_pattern in collection_name and 
                    collection_data.get('model', '').lower() == model_name):
                    
                    # Find all SAEs in this collection that match the id pattern
                    for sae_info in collection_data.get('saes', []):
                        sae_id = sae_info.get('id', '')
                        if id_pattern in sae_id or id_pattern == "":
                            potential_matches.append((collection_name, sae_id))
            
            # Print the list of potential matches
            if potential_matches:
                print(f"Found {len(potential_matches)} potential SAEs matching pattern ({release_pattern}, {id_pattern}):")
                for release, sae_id in potential_matches:
                    print(f"  - {release}: {sae_id}")
            else:
                print(f"No SAEs found matching pattern ({release_pattern}, {id_pattern})")
            return potential_matches
        except Exception as e:
            raise ValueError(f"An error occured when getting the sae releases and ids: {e}")

    def _find_and_import_saes(self, potential_matches):
        """Find and import SAEs matching the given patterns"""
        matching_saes = []
        
        try:
            # Now import the matching SAEs
            for release, sae_id in potential_matches:
                try:
                    # Import the SAE
                    sae, _, _ = SAE.from_pretrained(
                        release=release,
                        sae_id=sae_id
                    )
                    sae.use_error_term = True
                    sae = sae.to(self.device)
                    # Freeze the SAE parameters
                    for param in sae.parameters():
                        param.requires_grad = False
                    # Add the SAE to the model
                    self.model.add_sae(sae)
                    matching_saes.append(sae)
                    self.saes[sae.name] = sae
                    print(f"Added SAE: {sae.name} from {release}")
                except Exception as e:
                    print(f"Failed to import SAE {sae_id} from {release}: {e}")
        
        except Exception as e:
            raise ValueError(f"Error while trying to import SAEs: {e}")
        
        return matching_saes
    
    def _target_hooks_to_saes(self, target_hooks):
        # Load the pretrained SAEs YAML file once
        pretrained_saes = self._load_pretrained_saes_yaml()
        all_matching_saes = []
        
        # Process each target hook pattern (release, id) pair
        for release_pattern, id_pattern in target_hooks:
            matching_saes = []
            # import pdb;pdb.set_trace()
            # If no matching SAEs found, try to import them
            potential_matches = self._find_potential_sae_matches(release_pattern=release_pattern, id_pattern=id_pattern, pretrained_saes=pretrained_saes)
            matching_saes = self._find_and_import_saes(potential_matches)
            # import pdb;pdb.set_trace() 
            if not matching_saes:
                raise ValueError(f"No SAE found or could be imported for pattern ({release_pattern}, {id_pattern})")
            
            all_matching_saes.extend(matching_saes)
        return all_matching_saes

    def _sanitize_name(self, name):
        """Convert a name with dots to a valid module name by replacing dots with underscores"""
        return name.replace(".", "_").replace("/", "_")
    
    def reset(self):
        """Reset all trainable blocks to their initial state."""
        for block in self.trainable_blocks.values():
            block.initialize_parameters()

    def setup_trainable_blocks(self):
        """Set up the trainable blocks based on the configuration. Also works as a reset."""
        config = self.peft_config['default']
        
        # Clear existing state
        self.remove_hooks()  # Remove existing hooks
        self.saes = {}  # Clear the SAEs dictionary
        self.trainable_blocks = nn.ModuleDict()  # Create fresh ModuleDict
        
        # Process each target hook pattern (release, id) pair
        all_matching_saes = self._target_hooks_to_saes(target_hooks=config.target_hooks)
        
        # Now create a block for each unique matching SAE
        for sae in all_matching_saes:
            # Get the feature dimension from the SAE
            feature_dim = sae.cfg.d_sae
            sanitized_name = self._sanitize_name(sae.name)
            
            # Create either an IA3 block or ResidualBlock based on the config
            if config.ia3:
                block = IsaerftIA3(
                    sae=sae,
                    name=f"ia3_block_{sanitized_name}"
                ).to(self.device)
            else:
                block = ResidualBlock(
                    input_dim=feature_dim,
                    hidden_layers=config.depth,
                    hidden_dim=config.hidden_size,
                    name=f"residual_block_{sanitized_name}"
                ).to(self.device)
            block.requires_grad_(True)
            self.trainable_blocks[sanitized_name] = block
        self.add_hooks()
    
    def add_hooks(self):
        """Add hooks to the SAEs in the model"""
        for hook_name, sae in self.saes.items():
            # Remove any existing hooks
            sae.remove_all_hook_fns()
            
            # Create a partial function with the corresponding ResidualBlock
            trainable_hook = partial(
                resid_hook, 
                residual_block=self.trainable_blocks[self._sanitize_name(hook_name)]
            )
            
            # Add the hook to the SAE
            sae.add_hook('hook_sae_acts_post', trainable_hook)
    
    def remove_hooks(self):
        """Remove hooks from the SAEs"""
        for sae in self.saes.values():
            sae.remove_all_hook_fns()
    
    def forward(self, *args, **kwargs):
        """Forward pass that adds hooks before and removes them after"""
        # Add hooks before forward pass
        # self.add_hooks()
        # import pdb;pdb.set_trace()
        # Run the model's forward pass
        if "input_ids" in kwargs:
            input_tensor = kwargs.pop("input_ids")  # Use pop to remove it
        elif args:
            input_tensor = args[0]
            args = args[1:]  # Remove the first argument
        else:
            input_tensor = None
        output = self.model(input_tensor, *args, **kwargs, return_type='both')
        
        # Remove hooks after forward pass (optional, can keep them if needed)
        # self.remove_hooks()
        
        return output
    
    def get_trainable_parameters(self):
        """Get the trainable parameters of the model"""
        return self.trainable_blocks.parameters()

    def _set_adapter_layers(self, enabled=True):
        """Set the adapter layers to enabled or disabled state"""
        self._adapter_layers_enabled = enabled
        if enabled:
            self.add_hooks()
        else:
            self.remove_hooks()

    def enable_adapter_layers(self):
        """Enable all adapter layers"""
        self._set_adapter_layers(True)
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=True)

    def disable_adapter_layers(self):
        """Disable all adapter layers"""
        self._set_adapter_layers(False)
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=False)

    def _check_target_module_exists(self, *args, **kwargs):
        pass

    def _mark_only_adapters_as_trainable(self, *args, **kwargs):
        pass

    def inject_adapter(self, model, adapter_name, low_cpu_mem_usage=False):
        # No-op: we manage our own adapters and hooks
        return

class IsaerftPeft(PeftModel):
    def __init__(self, model, config: IsaerftConfig, adapter_name="default"):
        # Validate config
        assert config.__class__.__name__ == "IsaerftConfig", f"Expected IsaerftConfig, got {type(config)}"
        assert hasattr(config, 'is_prompt_learning'), "Config must have is_prompt_learning attribute"
        assert not config.is_prompt_learning

        # Register your model class in the mapping
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ISAERFT] = IsaerftModel
        
        # Freeze the base model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Call parent constructor
        super().__init__(model, config, adapter_name)
        
        # Verify initialization was successful
        assert hasattr(self, 'peft_config'), "PeftModel initialization failed: missing peft_config"
        assert adapter_name in self.peft_config, f"Adapter '{adapter_name}' not found in peft_config"
    
    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save the trainable blocks to the specified directory.
        
        Args:
            save_directory (str): Directory where the model should be saved
            **kwargs: Additional arguments passed to the underlying save methods
        """
        import os
        import json
        import torch
        
        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        # import pdb;pdb.set_trace()
        # Save the config
        self.peft_config[self.active_adapter].save_pretrained(save_directory)
        
        # Save each trainable block's state dict
        for name, block in self._modules['base_model'].trainable_blocks.items():
            # Create a path for this block
            block_path = os.path.join(save_directory, f"{name}.pt")
            # Save the state dict
            torch.save(block.state_dict(), block_path)
        
        # Save a manifest of all blocks
        block_names = list(self._modules['base_model'].trainable_blocks.keys())
        with open(os.path.join(save_directory, "blocks_manifest.json"), "w") as f:
            json.dump({"block_names": block_names}, f, indent=2)
        
        print(f"Model saved to {save_directory}")
        return save_directory

    def merge_and_unload(self):
        """Part of the Trainer init (eg ORPOTrainer) is to squish Peft into the model. We don't really have that, so do nothing"""
        return self

    def forward(self, *args, **kwargs):
        """Override the forward method to directly use the base model"""
        # Skip the problematic PeftModel.forward implementation
        # import pdb;pdb.set_trace()
        # Pop use_cache if present since model doesn't support it
        assert not kwargs.pop('use_cache', None), "Sorry, HookedSAETransformer doesn't have use_cache. Please make it either false or don't give it."
        # import pdb;pdb.set_trace()
        return self.base_model(*args, **kwargs)
    
    def get_base_model(self):
        """Override to avoid the problematic code in the parent class"""
        return self.base_model
    
    def get_trainable_parameters(self):
        """Get the trainable parameters of the model"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param



#%%
if __name__ == "__main__":
    import importlib
    import torch
    from transformers import AutoTokenizer
    from sae_lens import HookedSAETransformer
    importlib.reload(sys.modules['IsaerftConfig'])
    #%%
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading HookedSAETransformer...")
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device=device).to(device)
    print(f"Model loaded: {model.cfg.model_name}")
    #%%
    # Create IsaerftConfig
    print("Creating IsaerftConfig...")
    config = IsaerftConfig(
        target_hooks=[
            ("gemma-scope-2b-pt-res-canonical", "layer_25/width_16k/canonical"),  # Match all SAEs in layer 25
        ],
        depth=-1  # Bias-only for simplicity
    )
    print(f"Config created with target hooks: {config.target_hooks}")
    #%%
    # Create IsaerftPeft model
    print("Creating IsaerftPeft model...")
    peft_model = IsaerftPeft(model, config)
    print("IsaerftPeft model created")

    # Check which SAEs were added
    print(f"\nAdded {len(peft_model.saes)} SAEs:")
    for name in peft_model.saes:
        print(f"  - {name}")

    # Check trainable blocks
    print(f"\nCreated {len(peft_model.trainable_blocks)} trainable blocks:")
    for name, block in peft_model.trainable_blocks.items():
        print(f"  - {name}: {block}")
    #%%
    # Test with a simple input
    print("\nTesting forward pass...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    text = "Hello!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"Input shape: {input_ids.shape}")

    # Before forward pass
    peft_model.train()  # Set to training mode

    # Forward pass
    output = peft_model(input_ids)
    print(f"Output shape: {output.logits.shape}")
    #%%
    print(output.logits.sum())
    #%%
    # Compute loss and backpropagate
    print("\nTesting backward pass...")
    loss = output.logits.abs().sum()
    print(f"Loss: {loss.item()}")
    loss.backward()
    print(peft_model.trainable_blocks['blocks_25_hook_resid_post'].sequential[0].bias.grad.norm().item())
    #%%
    # Check if gradients are flowing to the trainable parameters
    print("\nChecking gradients:")
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None
            grad_norm = torch.norm(param.grad).item() if has_grad else 0
            print(f"  - {name}: has_grad={has_grad}, grad_norm={grad_norm:.4f}")

    print("\nTest completed successfully!")
    # %%
    #%%
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            print(param.shape)
    # %%

    # Test adapter enabling/disabling
    print("\nTesting adapter enabling/disabling...")
    
    # Randomize trainable parameters to ensure we can detect adapter effects
    print("Randomizing trainable parameters...")
    with torch.no_grad():
        for param in peft_model.parameters():
            if param.requires_grad:
                param.data = torch.randn_like(param) * 0.1  # Small scale to avoid extreme values
    
    # Get initial output with adapters enabled
    output_enabled = peft_model(input_ids)
    initial_logits = output_enabled.logits.detach().clone()
    
    # Disable adapters
    print("Disabling adapters...")
    peft_model.disable_adapter_layers()
    
    # Get output with adapters disabled
    output_disabled = peft_model(input_ids)
    disabled_logits = output_disabled.logits.detach().clone()
    
    # Check that outputs are different
    logits_diff = torch.norm(initial_logits - disabled_logits).item()
    print(f"Logits difference after disabling: {logits_diff:.4f}")
    assert logits_diff > 0, "Disabling adapters should change the output"
    
    # Re-enable adapters
    print("Re-enabling adapters...")
    peft_model.enable_adapter_layers()
    
    # Get output with adapters re-enabled
    output_reenabled = peft_model(input_ids)
    reenabled_logits = output_reenabled.logits.detach().clone()
    
    # Check that outputs match initial state
    logits_diff = torch.norm(initial_logits - reenabled_logits).item()
    print(f"Logits difference after re-enabling: {logits_diff:.4f}")
    assert logits_diff < 1e-6, "Re-enabling adapters should restore original output"
    
    print("Adapter enabling/disabling test completed successfully!")
