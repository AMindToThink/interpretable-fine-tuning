# Script which compares using pytorch hooks and using HookedSAETransformer to check that they are the same.

import torch
from torch import Tensor
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from transformer_lens.hook_points import HookPoint
from functools import partial

def create_hooked_model(model_name, sae, device="cuda"):
    """Create a HookedSAETransformer model with a hook at layer 20"""
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    model.add_sae(sae)
    return model

def create_pytorch_model(model_name, sae, device="cuda"):
    """Create a PyTorch model with a hook at layer 20"""
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # import pdb;pdb.set_trace()
    # Create a hook that adds 1 to the activations
    def hook_fn(module, input, output):
        # import pdb;pdb.set_trace()
        # output is a tuple, we need to modify the first element
        
        sae_out = sae(output[0])
        return (sae_out,) + output[1:]
    
    # Register hook at layer 20's MLP output
    # model.get_submodule('model.layers.20.mlp').register_forward_hook(hook_fn)
    model.model.layers[20].register_forward_hook(hook_fn) # If you target 19 instead of 20, it breaks. The maximum absolute difference becomes 19.155
    
    return model

def compare_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/gemma-2-2b"
    sae_release = 'gemma-scope-2b-pt-res-canonical'
    sae_id = 'layer_20/width_16k/canonical'
    sae_20 = SAE.from_pretrained(
                        sae_release, sae_id, device=str(device)
                    )[0]
    
    def steering_hook(
        activations,#: Float[Tensor],  # Float[Tensor, "batch pos d_in"], Either jaxtyping or lm-evaluation-harness' precommit git script hate a type hint here.
        hook: HookPoint,
        latent_idx: int,
        steering_coefficient: float,
    ) -> Tensor:
        """
        Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
        sequence positions.
        Conceptually, the rows in sae.W_dec represent the expanded forms of "features" (aka concepts) discovered by the SAE.
        Adding that row is increasing that "feature".
        """
        assert (activations >= 0).all(), "oops, why are activations negative"
        print(activations)
        print(activations.shape)
        print()
        activations[:, :, latent_idx] += steering_coefficient
        print(activations)
        return activations
    sae_20.add_hook('hook_sae_acts_post', partial(steering_hook, latent_idx=12082, steering_coefficient=240.0))
    # Create both models
    hooked_model = create_hooked_model(model_name, sae_20, device)
    pytorch_model = create_pytorch_model(model_name, sae_20, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test input
    text = "When I look in the mirror, I see"
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
    
    # Generate tokens from each model
    print("\nGenerating tokens from HookedSAE model:")
    hooked_tokens = hooked_model.generate(
        inputs.input_ids,
        max_new_tokens=20,
        do_sample=False
    )
    hooked_text = tokenizer.decode(hooked_tokens[0])
    print(f"HookedSAE output: {hooked_text}")

    print("\nGenerating tokens from PyTorch model:") 
    pytorch_tokens = pytorch_model.generate(
        inputs.input_ids,
        max_new_tokens=20,
        do_sample=False
    )
    pytorch_text = tokenizer.decode(pytorch_tokens[0])
    print(f"PyTorch output: {pytorch_text}")
    import pdb;pdb.set_trace()
    return hooked_output, pytorch_output

if __name__ == "__main__":
    compare_models()

