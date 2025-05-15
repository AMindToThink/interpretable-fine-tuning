import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add src directory to path

from sae_lens import SAE, HookedSAETransformer
from model_components.IsaerftIA3 import IsaerftIA3
from transformers import AutoTokenizer

def demonstrate_sae_with_ia3():
    """Example showing how to use IsaerftIA3 with real SAEs"""
    
    # Use CPU for this example, or CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_name = "google/gemma-2-2b"
    print(f"Loading model {model_name}...")
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    
    # Load an SAE
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_20/width_16k/canonical"
    print(f"Loading SAE {sae_id} from {sae_release}...")
    
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id
    )
    sae = sae.to(device)
    
    # Create an IA3 component for the SAE
    print(f"Creating IsaerftIA3 component for SAE with {sae.cfg.d_sae} features...")
    ia3 = IsaerftIA3(sae=sae)
    
    # Test input
    print("Setting up a test input...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Run a forward pass through the model normally
    print("Running standard forward pass...")
    with torch.no_grad():
        base_output = model(inputs.input_ids)
    
    # Now let's hook up the IA3 component to the SAE
    def ia3_hook(tensor, *, hook):
        # Apply the IA3 component to the SAE features
        return ia3(tensor)
    
    # Add hook to SAE
    print("Adding IA3 hook to SAE...")
    model.add_sae(sae)  # Add the SAE to the model
    sae.add_hook('hook_sae_acts_post', ia3_hook)  # Hook the IA3 component
    
    # Run a forward pass with the hooked IA3 component
    print("Running forward pass with IA3 component...")
    with torch.no_grad():
        ia3_output = model(inputs.input_ids)
    
    # Compare outputs
    print("\nOutput shape:", ia3_output.shape)
    
    # See if the outputs are different
    diff = torch.abs(base_output - ia3_output)
    print(f"Max difference between outputs: {diff.max().item():.6f}")
    print(f"Mean difference between outputs: {diff.mean().item():.6f}")
    
    # The outputs should be the same if the IA3 scaling_factors are all zeros
    print("\nIA3 scaling_factors mean:", ia3.scaling_factors.mean().item())
    
    # Let's modify some scaling factors and run again
    print("\nModifying IA3 scaling factors...")
    ia3.scaling_factors.data.fill_(0.5)  # Set all scaling factors to 0.5
    
    # Run a forward pass with modified IA3 parameters
    with torch.no_grad():
        modified_output = model(inputs.input_ids)
    
    # Compare with original output
    mod_diff = torch.abs(base_output - modified_output)
    print(f"Max difference after modification: {mod_diff.max().item():.6f}")
    print(f"Mean difference after modification: {mod_diff.mean().item():.6f}")
    
    # The difference should be larger now that we've modified the scaling factors
    
    return model, sae, ia3

if __name__ == "__main__":
    demonstrate_sae_with_ia3() 