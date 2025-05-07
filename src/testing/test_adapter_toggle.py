import torch
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer
from ..model_components.IsaerftConfig import IsaerftConfig
from ..model_components.IsaerftPeft import IsaerftPeft

def test_adapter_toggle():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading HookedSAETransformer...")
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device=device).to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    print("Model and tokenizer loaded")

    # Create IsaerftConfig targeting a specific SAE layer
    config = IsaerftConfig(
        target_hooks=[
            ("gemma-scope-2b-pt-res-canonical", "layer_25/width_16k/canonical"),
        ],
        depth=-1,  # Multiple layers
    )

    # Create IsaerftPeft model
    print("Creating IsaerftPeft model...")
    peft_model = IsaerftPeft(model, config)

    # Initialize the adapter weights to large random values to create obvious distortions
    print("Setting adapter weights to large random values...")
    for name, block in peft_model.base_model.trainable_blocks.items():
        # Set all weights in the sequential layers to large random values
        with torch.no_grad():
            for layer in block.sequential:
                if hasattr(layer, 'weight'):
                    # Initialize with large random values
                    layer.weight.data = torch.randn_like(layer.weight) * 10.0
                if hasattr(layer, 'bias'):
                    layer.bias.data = torch.randn_like(layer.bias) * 10.0

    # Test text
    test_prompt = "The quick brown fox jumps over the lazy dog."

    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Test 1: With adapter enabled (should produce garbage)
    print("\n=== Test with adapter ENABLED ===")
    peft_model.enable_adapter_layers()
    output_with_adapter = generate_text(test_prompt)
    print(f"Input: {test_prompt}")
    print(f"Output: {output_with_adapter}")

    # Test 2: With adapter disabled (should produce normal text)
    print("\n=== Test with adapter DISABLED ===")
    peft_model.disable_adapter_layers()
    output_without_adapter = generate_text(test_prompt)
    print(f"Input: {test_prompt}")
    print(f"Output: {output_without_adapter}")

    # Test 3: Re-enable adapter to verify it still produces garbage
    print("\n=== Test with adapter RE-ENABLED ===")
    peft_model.enable_adapter_layers()
    output_readapter = generate_text(test_prompt)
    print(f"Input: {test_prompt}")
    print(f"Output: {output_readapter}")

if __name__ == "__main__":
    test_adapter_toggle()