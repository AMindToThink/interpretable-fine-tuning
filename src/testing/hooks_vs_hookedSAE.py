# Script which compares using pytorch hooks and using HookedSAETransformer to check that they are the same.

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from transformer_lens.hook_points import HookPoint

def create_hooked_model(model_name="google/gemma-2-2b", device="cuda"):
    """Create a HookedSAETransformer model with a hook at layer 20"""
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    # import pdb;pdb.set_trace()
    # Create a simple hook that just adds 1 to the activations
    def hook_fn(tensor, *, hook):
        print(tensor)
        print(f"HookedSAE hook: Input shape {tensor.shape}, mean {tensor.mean().item():.4f}")
        result = tensor + 1.0
        print(f"HookedSAE hook: Output shape {result.shape}, mean {result.mean().item():.4f}")
        return result
    
    def print_hook(tensor, hook):
        print(tensor)
        return tensor
    # Add hook at layer 20's MLP output
    model.add_hook("blocks.20.hook_resid_post", hook_fn)
    return model

def create_pytorch_model(model_name="google/gemma-2-2b", device="cuda"):
    """Create a PyTorch model with a hook at layer 20"""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    # import pdb;pdb.set_trace()
    # Create a hook that adds 1 to the activations
    def hook_fn(module, input, output):
        # import pdb;pdb.set_trace()
        # output is a tuple, we need to modify the first element
        print(f"PyTorch hook: Input shape {input[0].shape}, mean {input[0].mean().item():.4f}")
        print(f"PyTorch hook: Output shape {output[0].shape}, mean {output[0].mean().item():.4f}")
        # modified = (output[0] + 1.0,) + output[1:]
        # print(f"PyTorch hook: Modified shape {modified[0].shape}, mean {modified[0].mean().item():.4f}")
        print(output)
        return (output[0] + 1.0,) + output[1:]
    
    # Register hook at layer 20's MLP output
    # model.get_submodule('model.layers.20.mlp').register_forward_hook(hook_fn)
    model.model.layers[20].register_forward_hook(hook_fn)
    assert model.get_submodule('model.layers.20') == model.model.layers[20]

    return model

def compare_models():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    model_name = "google/gemma-2-2b"
    
    # Create both models
    hooked_model = create_hooked_model(model_name, device)
    pytorch_model = create_pytorch_model(model_name, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test input
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Get outputs
    with torch.no_grad():
        print("\nRunning HookedSAE model:")
        hooked_output = hooked_model(inputs.input_ids)
        print("\nRunning PyTorch model:")
        pytorch_output = pytorch_model(inputs.input_ids).logits
    
    # Compare outputs
    print("\nOutput shapes:")
    print(f"HookedSAETransformer: {hooked_output.shape}")
    print(f"PyTorch model: {pytorch_output.shape}")
    
    # Compare values
    diff = torch.abs(hooked_output - pytorch_output)
    print("\nMaximum absolute difference:", diff.max().item())
    print("Mean absolute difference:", diff.mean().item())
    
    # Print some statistics about the outputs
    print("\nOutput statistics:")
    print(f"HookedSAE mean: {hooked_output.mean().item():.4f}, std: {hooked_output.std().item():.4f}")
    print(f"PyTorch mean: {pytorch_output.mean().item():.4f}, std: {pytorch_output.std().item():.4f}")
    
    return hooked_output, pytorch_output

if __name__ == "__main__":
    compare_models()
