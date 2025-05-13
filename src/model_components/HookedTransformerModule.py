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

if __name__ == '__main__':
    import torch
    from sae_lens import HookedSAETransformer, SAE
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading HookedSAETransformer...")
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device=device).to(device)
    print(f"Model loaded: {model.cfg.model_name}")

    # Load SAE
    print("Loading SAE...")
    sae_release = "gemma-scope-2b-pt-res-canonical"
    model_sae_id = 'layer_20/width_16k/canonical'
    test_sae, sae_dict, _ = SAE.from_pretrained(release=sae_release, sae_id=model_sae_id)
    test_sae = test_sae.to(device)
    # print(f"SAE loaded: {test_sae.name}")

    # Create IsaerftIA3Model
    print("Creating IsaerftIA3Model...")
    peft_model = IsaerftIA3Model(model, test_sae)
    print("IsaerftIA3Model created")

    # Test with a simple input
    print("\nTesting forward pass...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    text = "Hello!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    output = peft_model(input_ids)
    print(f"Output shape: {output.logits.shape}")
    print(f"Output sum: {output.logits.sum().item()}")
    
    from pprint import pprint
    pprint([(n, type(m)) for n, m in peft_model.named_modules()])
    from peft import IA3Config, get_peft_model
    config = IA3Config(
        task_type="CAUSAL_LM",
        # Use a list of exact module names to target
        target_modules=["blocks.20.hook_resid_post.hook_sae_acts_post"],
        # Similarly exact name for feedforward module
        feedforward_modules=[],
        init_ia3_weights=True
    )
    ia3_model = get_peft_model(peft_model, config)
    ia3_model.print_trainable_parameters()

