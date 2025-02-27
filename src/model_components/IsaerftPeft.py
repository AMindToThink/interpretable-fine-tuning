#%%
# from elsewhere
import torch
from torch import nn
from transformers import AutoConfig
from peft import PeftModel
from sae_lens import HookedSAETransformer, SAE
from peft.utils import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from functools import partial

# Handle imports differently based on how the file is being used
try:
    # When imported as a module
    from .ResidualBlock import ResidualBlock
    from .IsaerftConfig import IsaerftConfig
except ImportError:
    # When run directly as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_components.ResidualBlock import ResidualBlock
    from IsaerftConfig import IsaerftConfig

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
class IsaerftModel(nn.Module):
    """Implementation of the ISAERFT model"""
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.config = AutoConfig.from_pretrained(model.cfg.tokenizer_name)
        self.peft_config = {adapter_name: config}
        self.active_adapter = adapter_name
        self.warnings_issued = {}
        
        # Store SAEs that have been added to the model
        self.saes = {}
        
        # Create trainable blocks for each target hook
        self.setup_trainable_blocks()
    def resize_token_embeddings(self, num_new_tokens, *args):
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
                        sae_id=sae_id,
                        device=next(self.model.parameters()).device
                    )
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

    def setup_trainable_blocks(self):
        """Set up the trainable blocks based on the configuration"""
        config = self.peft_config[self.active_adapter]['default']
        
        self.trainable_blocks = nn.ModuleDict()
        
        # Process each target hook pattern (release, id) pair
        all_matching_saes = self._target_hooks_to_saes(target_hooks=config.target_hooks)
        
        # Now create a ResidualBlock for each unique matching SAE
        for sae in all_matching_saes:
            # Get the feature dimension from the SAE
            feature_dim = sae.cfg.d_sae
            sanitized_name = self._sanitize_name(sae.name)
            
            # Create a ResidualBlock based on the config
            block = ResidualBlock(
                input_dim=feature_dim,
                hidden_layers=config.depth,
                hidden_dim=config.hidden_size,
                name=f"residual_block_{sanitized_name}"
            ).to(next(self.model.parameters()).device)
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
        
        # Run the model's forward pass
        output = self.model(*args, **kwargs)
        
        # Remove hooks after forward pass (optional, can keep them if needed)
        # self.remove_hooks()
        
        return output
    
    def get_trainable_parameters(self):
        """Get the trainable parameters of the model"""
        return self.trainable_blocks.parameters()

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
    
    def forward(self, *args, **kwargs):
        """Override the forward method to directly use the base model"""
        # Skip the problematic PeftModel.forward implementation
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading HookedSAETransformer...")
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device=device)
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
    text = "Hello, world!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"Input shape: {input_ids.shape}")

    # Before forward pass
    peft_model.train()  # Set to training mode

    # Forward pass
    output = peft_model(input_ids)
    print(f"Output shape: {output.shape}")
    #%%
    print(output.sum())
    #%%
    # Compute loss and backpropagate
    print("\nTesting backward pass...")
    loss = output.abs().sum()
    print(f"Loss: {loss.item()}")
    loss.backward()
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
