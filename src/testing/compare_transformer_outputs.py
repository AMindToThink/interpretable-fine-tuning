import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sae_lens import HookedSAETransformer

def compare_transformer_outputs():
    device ='cpu'# "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/gemma-2-2b"
    
    # Create both models
    print("Loading models...")
    hooked_model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device) # from_pretrained has a small delta of .3. 
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test input
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"\nInput text: {text}")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Get outputs
    print("\nRunning models...")
    with torch.no_grad():
        hooked_output = hooked_model(inputs.input_ids)
        hf_output = hf_model(inputs.input_ids).logits
    
    # Compare outputs
    print("\nOutput shapes:")
    print(f"HookedSAETransformer: {hooked_output.shape}")
    print(f"HuggingFace model: {hf_output.shape}")
    
    # Compare values
    diff = torch.abs(hooked_output - hf_output)
    print("\nMaximum absolute difference:", diff.max().item())
    print("Mean absolute difference:", diff.mean().item())
    
    # Print some statistics about the outputs
    print("\nOutput statistics:")
    print(f"HookedSAE mean: {hooked_output.mean().item():.4f}, std: {hooked_output.std().item():.4f}")
    print(f"HuggingFace mean: {hf_output.mean().item():.4f}, std: {hf_output.std().item():.4f}")
    
    # Print first few logits for comparison
    print("\nFirst few logits comparison:")
    print("HookedSAE:", hooked_output[0, -1, :5].tolist())
    print("HuggingFace:", hf_output[0, -1, :5].tolist())
    
    return hooked_output, hf_output

if __name__ == "__main__":
    compare_transformer_outputs() 