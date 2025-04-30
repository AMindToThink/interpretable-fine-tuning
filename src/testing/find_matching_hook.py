import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sae_lens import HookedSAETransformer
import numpy as np
from tqdm import tqdm

def get_hooked_tensor(model_name="google/gemma-2-2b", device="cpu"):
    """Get the tensor from HookedSAETransformer with hook at blocks.20.hook_mlp_out"""
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    
    # Create a hook that captures the tensor
    captured_tensor = None
    def hook_fn(tensor, *, hook):
        nonlocal captured_tensor
        captured_tensor = tensor.clone()
        return tensor
    
    # Add hook at layer 20's MLP output
    model.add_hook("blocks.20.hook_mlp_out", hook_fn)
    
    # Get output
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        model(inputs.input_ids)
    
    return captured_tensor

def find_matching_hook():
    device = "cpu" if torch.cpu.is_available() else "cpu"
    model_name = "google/gemma-2-2b"
    
    # Get the reference tensor from HookedSAETransformer
    print("Getting reference tensor from HookedSAETransformer...")
    reference_tensor = get_hooked_tensor(model_name, device)
    print(f"\nReference tensor at blocks.20.hook_mlp_out:")
    print(f"Shape: {reference_tensor.shape}")
    print(f"Mean: {reference_tensor.mean().item():.4f}")
    print(f"Std: {reference_tensor.std().item():.4f}")
    
    # Create PyTorch model
    print("\nCreating PyTorch model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Set up hooks to compare tensors directly
    print("Searching for matching hook points...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    def hook_fn(module, input, output, hook_path):
        # Get the first tensor from input tuple if it exists
        if not isinstance(input, tuple) or len(input) == 0:
            return output
            
        tensor = input[0]
        if not isinstance(tensor, torch.Tensor):
            return output
        
        if tensor.shape == reference_tensor.shape:
            mean_diff = abs(tensor.mean().item() - reference_tensor.mean().item())
            std_diff = abs(tensor.std().item() - reference_tensor.std().item())
            
            if mean_diff < 0.1 and std_diff < 0.1:
                print(f"\nPotential match found at: {hook_path}")
                print(f"Shape: {tensor.shape}")
                print(f"Mean: {tensor.mean().item():.4f} (ref: {reference_tensor.mean().item():.4f})")
                print(f"Std: {tensor.std().item():.4f} (ref: {reference_tensor.std().item():.4f})")
                
                diff = torch.abs(tensor - reference_tensor)
                print(f"Mean difference: {diff.mean().item():.4f}")
                print(f"Max difference: {diff.max().item():.4f}")
        
        return output
    
    # Register hooks for all modules
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Module)]
    for name, module in tqdm(modules, desc="Registering hooks"):
        module.register_forward_hook(lambda m, i, o, p=name: hook_fn(m, i, o, p))
    
    # Run the model once
    with torch.no_grad():
        model(inputs.input_ids)

if __name__ == "__main__":
    find_matching_hook() 