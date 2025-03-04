# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
print(torch.cuda.device_count())

# %%
# Import libraries
import torch
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer

# %%
# Import our custom ISAERFT components
try:
    # When imported as a module
    from model_components.IsaerftConfig import IsaerftConfig
    from model_components.IsaerftPeft import IsaerftPeft
except ImportError:
    # When run directly as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from model_components.IsaerftConfig import IsaerftConfig
    from model_components.IsaerftPeft import IsaerftPeft

# %%
# Authenticate to Hugging Face
from huggingface_hub import login
login(token=os.environ['HUGGINGFACE_WRITE_KEY'])

# %%
# Define the model
model_name = "google/gemma-2-2b"

# Get the actual device that CUDA is using
if torch.cuda.is_available():
    device = f"cuda:{torch.cuda.current_device()}"
else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

assert 'cuda' in device
print(f"Using device: {device}")

# %%
# Load model
hooked_sae_transformer = HookedSAETransformer.from_pretrained(
    model_name,
).to(device)
assert isinstance(hooked_sae_transformer, HookedSAETransformer)

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')

# %%
# Configure ISAERFT with BiasOnly (depth=-1)
example_releases_ids = {
    "EleutherAI/pythia-70m-deduped":("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post"),
    "google/gemma-2-2b": ("gemma-scope-2b-pt-res-canonical","layer_20/width_16k/canonical")
} 

isaerft_config = IsaerftConfig(
    target_hooks=[
        example_releases_ids[model_name],
    ],
    depth=-1  # Bias-only mode
)

# %%
# Apply ISAERFT to the model
model = IsaerftPeft(hooked_sae_transformer, isaerft_config)

# %%
# Test which parameters are trainable
def count_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    trainable_param_names = []
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_param_names.append(name)
    
    return {
        'trainable_params': trainable_params,
        'all_params': all_params,
        'percent_trainable': 100 * trainable_params / all_params,
        'trainable_param_names': trainable_param_names
    }

# %%
# Run the test
results = count_trainable_parameters(model)
print(f"Total parameters: {results['all_params']:,}")
print(f"Trainable parameters: {results['trainable_params']:,}")
print(f"Percent trainable: {results['percent_trainable']:.4f}%")
print("\nTrainable parameter names:")
for name in results['trainable_param_names']:
    print(f"  - {name}")

# %%
# Verify that only bias parameters are trainable
bias_only = all('bias' in name for name in results['trainable_param_names'])
print(f"\nOnly bias parameters are trainable: {bias_only}")

# Check if any non-bias parameters are trainable
non_bias_trainable = [name for name in results['trainable_param_names'] if 'bias' not in name]
if non_bias_trainable:
    print("\nWARNING: Found non-bias trainable parameters:")
    for name in non_bias_trainable:
        print(f"  - {name}")
else:
    print("\nSuccess! Only bias parameters are trainable.")

# %%
# Add a check for parameter changes with HF Trainer
from transformers import Trainer, TrainingArguments
import copy
import numpy as np

def check_parameter_changes_with_trainer(model, tokenizer):
    # Move model to CPU for this test
    model_cpu = model.to("cpu")
    
    # Create a deep copy of the initial model parameters
    initial_params = {}
    for name, param in model_cpu.named_parameters():
        initial_params[name] = param.detach().clone()
    
    # Import SFTTrainer and datasets
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    # Use a simple existing dataset instead of creating our own
    dataset = load_dataset("stanfordnlp/imdb", split="train").select(range(1))
    
    training_args = SFTConfig(
        max_seq_length=512,
        output_dir="/tmp",
        learning_rate=10,  # High learning rate to ensure changes
        no_cuda=True,
        weight_decay=0.0,
    )
    trainer = SFTTrainer(
        model=model_cpu,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
    )
    trainer.train()
    
    # Check which parameters changed
    changed_params = []
    unchanged_params = []
    
    for name, param in model_cpu.named_parameters():
        # Calculate the difference
        if name in initial_params:
            diff = torch.sum(torch.abs(param.detach() - initial_params[name])).item()
            if diff > 0:
                changed_params.append((name, diff))
            else:
                unchanged_params.append(name)
    
    # Sort by magnitude of change
    changed_params.sort(key=lambda x: x[1], reverse=True)
    
    # Move model back to original device
    model_cpu.to(device)
    import pdb;pdb.set_trace
    return {
        'changed_params': changed_params,
        'unchanged_params': unchanged_params
    }

# %%
# Run the trainer check
print("\n=== Testing parameter changes with HF Trainer ===")
trainer_results = check_parameter_changes_with_trainer(model, tokenizer)

print(f"\nParameters that changed during training ({len(trainer_results['changed_params'])} total):")
for name, diff in trainer_results['changed_params']:
    print(f"  - {name}: change magnitude = {diff:.6f}")

# Check if only bias parameters changed
bias_only_changed = all('bias' in name for name, _ in trainer_results['changed_params'])
print(f"\nOnly bias parameters changed: {bias_only_changed}")

# Check if any non-bias parameters changed
non_bias_changed = [(name, diff) for name, diff in trainer_results['changed_params'] if 'bias' not in name]
if non_bias_changed:
    print("\nWARNING: Found non-bias parameters that changed:")
    for name, diff in non_bias_changed:
        print(f"  - {name}: change magnitude = {diff:.6f}")
else:
    print("\nSuccess! Only bias parameters changed during training.")
# %%
trainer_results['changed_params']

# %%
