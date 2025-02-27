# Following along with https://medium.com/myorder/fine-tuning-pythia-70m-deduped-instruction-following-llms-with-performance-evaluation-3bd0bb33b79
# %%
%load_ext autoreload
%autoreload 2
#%%
# Import libraries
from dataclasses import dataclass
import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
from sae_lens import HookedSAETransformer, SAE
#%%
# Import our custom ISAERFT components
try:
    # When imported as a module
    from model_components.IsaerftConfig import IsaerftConfig
    from model_components.IsaerftPeft import IsaerftPeft
except ImportError:
    # When run directly as a script
    import sys
    import os
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_components.IsaerftConfig import IsaerftConfig
    from model_components.IsaerftPeft import IsaerftPeft
#%%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Use only GPU 0
# import torch
# print(torch.cuda.device_count())  # Should print 1, but doesn't
#%%
# Authenticate to Hugging Face
from huggingface_hub import login

login(token=os.environ['HUGGINGFACE_WRITE_KEY'])

#%%
# Load dataset
dataset = load_dataset(path="trl-lib/ultrafeedback_binarized")

#%%
# Define the model
model_name = "EleutherAI/pythia-70m-deduped" 

# Get the actual device that CUDA is using
if torch.cuda.is_available():
    # Get the current device that's actually being used
    device = f"cuda:{torch.cuda.current_device()}"
else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

assert 'cuda' in device
print(f"Using device: {device}")

#%%
# Model to fine-tune
model = HookedSAETransformer.from_pretrained(
    model_name,
    # torch_dtype=torch.float32,
    # device_map=device
).to(device)
# model.config.use_cache = False
# tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
#%%
chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
### Instruction:
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
### Response:
{{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
### Response:
{% endif %}"""
tokenizer.pad_token = tokenizer.eos_token

tokenizer.chat_template = chat_template
#%%
# non_hooked_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to('cpu')
#%%
# del non_hooked_model
# chat = [
#     { "role": "user", "content": "Write a hello world program" },
# ]
# prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# print(prompt)
# #%%
# _, tokenizer = setup_chat_format(non_hooked_model, tokenizer) # TODO: Figure out if this is a problem that I'm not applying the chat format to the model
#%%
# Apply ISAERFT to the model
from sae_lens import SAE

release = "pythia-70m-deduped-res-sm"
sae_id = "blocks.4.hook_resid_post"
isaerft_config = IsaerftConfig(
    target_hooks=[
        (release, sae_id),
    ],
    depth=-1  # Bias-only for simplicity
)
#%%
from sae_lens import SAE

release = "pythia-70m-deduped-res-sm"
sae_id = "blocks.4.hook_resid_post"
sae = SAE.from_pretrained(release, sae_id)[0]
#%%
print(sae)
#%%
# Apply the ISAERFT adapter
model = IsaerftPeft(model, isaerft_config)
#%%
# Check device placement of model components
def check_device():
    print("Checking device placement of model components...")
    model_device = next(model.base_model.parameters()).device
    import torch
    print(f"{torch.device(device)=}")
    print(f"{model_device=}")
    assert model_device == torch.device(device)
    # Check base model
    print(f"Base model device: {model_device}")

    assert(all(p[1].device == model_device for p in model.named_parameters()))
    text = "Hello, world!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"Input shape: {input_ids.shape}")


    # Forward pass
    output = model(input_ids)
    print(f"Output shape: {output.shape}")
    print(tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True))
    # Run an example through the model to verify forward pass
    print("Testing model forward pass...")

    # Create a simple test input
    test_chat = [
        {"role": "user", "content": "Write a hello world program"}
    ]
    test_prompt = tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    print(test_prompt)
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    print(inputs)
    # Run forward pass
    outputs = model(**inputs)
    print(outputs)
    print("Forward pass successful!")
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Output device: {outputs.logits.device}")
    assert outputs.logits.device == model_device, "Output device doesn't match model device"

#%%

# model, tokenizer = setup_chat_format(model, tokenizer)
#%%
# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "PYTHIA-FT-ORPO-ISAERFT"
finetune_tags = ["smol-course", "module_1", "isaerft"]

#%%
# Train model with ORPO
orpo_args = ORPOConfig(
    # Small learning rate to prevent catastrophic forgetting
    learning_rate=8e-6,
    # Linear learning rate decay over training
    lr_scheduler_type="linear",
    # Maximum combined length of prompt + completion
    max_length=1024,
    # Maximum length for input prompts
    max_prompt_length=512,
    # Controls weight of the odds ratio loss (λ in paper)
    beta=0.1,
    # Batch size for training
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    # Helps with training stability by accumulating gradients before updating
    gradient_accumulation_steps=8,
    # Memory-efficient optimizer for CUDA, falls back to adamw_torch for CPU/MPS
    optim="paged_adamw_8bit" if ("cuda" in device)else "adamw_torch",
    # When to run evaluation
    eval_strategy="steps",
    # Evaluate every 20% of training
    eval_steps=0.2,
    # Log metrics every step
    logging_steps=1,
    # Gradual learning rate warmup
    warmup_steps=10,
    # Disable external logging
    report_to="wandb",
    # Where to save model/checkpoints
    output_dir="./results/",
    # Enable MPS (Metal Performance Shaders) if available
    use_mps_device=device == "mps",
    hub_model_id=finetune_name,
    # Training for a shorter time for this example
    num_train_epochs=(1/4*.25),
    # Ensure device placement is correct
    no_cuda=False,
    dataloader_pin_memory=False,
    dataloader_drop_last=True,
)

#%%
thesae = next(iter(model.saes.items()))[1]
[(k,v) for k,v in thesae.__dict__.items() if hasattr(v, 'device')]
# thesae.W_E

#%%
# Initialize wandb
import wandb
from datetime import datetime
import uuid
wandb.finish()
wandb.login(key=os.environ['WANDB_KEY'])

wandb.init(
    project="pythia70m-orpo-isaerft",
    name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:6]}",
    tags=finetune_tags
)
#%%
# Create the trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"].select(range(10)),
    eval_dataset=dataset["test"].select(range(10)),
    processing_class=tokenizer,
    # label_names=["labels"],  # This is the standard label name for causal language models
)

# Get the actual device the model is on
model_device = next(model.parameters()).device
print(f"Model is on device: {model_device}")

# Add this before training to ensure inputs are on the correct device
def move_inputs_to_device(batch, target_device):
    # Only move tensors that aren't already on the target device
    return {k: (v.to(target_device) if isinstance(v, torch.Tensor) and v.device != target_device else v) 
            for k, v in batch.items()}

# Override the default get_train_dataloader to ensure tensors are on the right device
original_get_train_dataloader = trainer.get_train_dataloader
def get_train_dataloader_with_device():
    # Disable pin_memory since we're handling device placement manually
    original_pin_memory = trainer.args.dataloader_pin_memory
    trainer.args.dataloader_pin_memory = False
    
    dataloader = original_get_train_dataloader()
    original_collate_fn = dataloader.collate_fn
    
    def collate_fn_with_device(*args, **kwargs):
        batch = original_collate_fn(*args, **kwargs)
        return move_inputs_to_device(batch, model_device)
    
    dataloader.collate_fn = collate_fn_with_device
    
    # Restore original pin_memory setting
    trainer.args.dataloader_pin_memory = original_pin_memory
    return dataloader

trainer.get_train_dataloader = get_train_dataloader_with_device

# Do the same for eval dataloader
original_get_eval_dataloader = trainer.get_eval_dataloader
def get_eval_dataloader_with_device(eval_dataset=None):
    # Disable pin_memory since we're handling device placement manually
    original_pin_memory = trainer.args.dataloader_pin_memory
    trainer.args.dataloader_pin_memory = False
    
    dataloader = original_get_eval_dataloader(eval_dataset)
    original_collate_fn = dataloader.collate_fn
    
    def collate_fn_with_device(*args, **kwargs):
        batch = original_collate_fn(*args, **kwargs)
        return move_inputs_to_device(batch, model_device)
    
    dataloader.collate_fn = collate_fn_with_device
    
    # Restore original pin_memory setting
    trainer.args.dataloader_pin_memory = original_pin_memory
    return dataloader

trainer.get_eval_dataloader = get_eval_dataloader_with_device

# Store the original data collator
original_data_collator = trainer.data_collator

# Create a new data collator that wraps the original one
def device_aware_collator(features):
    # Use the original collator to create the batch
    batch = original_data_collator(features)
    # Move the batch to the correct device
    return move_inputs_to_device(batch, model_device)

# Replace the trainer's data collator with our wrapper
trainer.data_collator = device_aware_collator

# Patch the compute_loss method to ensure inputs are on the right device
original_compute_loss = trainer.compute_loss
def compute_loss_with_device_check(model, inputs, return_outputs=False, **kwargs):
    # Ensure all inputs are on the same device as the model
    device_checked_inputs = move_inputs_to_device(inputs, model_device)
    return original_compute_loss(model, device_checked_inputs, return_outputs, **kwargs)

trainer.compute_loss = compute_loss_with_device_check

# Also modify the orpo_args to disable pin_memory
orpo_args.dataloader_pin_memory = False

#%%
# Train the model
trainer.train()

#%%
# Save the model
trainer.save_model(f"./{finetune_name}")

# Finish wandb logging
wandb.finish()

#%%
# Push to hub
try:
    print("Pushing model to hub...")
    trainer.push_to_hub(tags=finetune_tags)
    print("Successfully pushed to hub!")
except Exception as e:
    print(f"Error pushing to hub: {str(e)}")
    # Alternative manual push
    print("Attempting manual push...")
    model.push_to_hub(finetune_name, tags=finetune_tags)
    tokenizer.push_to_hub(finetune_name, tags=finetune_tags)
    print("Manual push completed!")

print("## 💐 You're done!")
print("You've successfully fine-tuned SmolLM2 with ORPO and ISAERFT!")
print("This approach allows you to align the model while only training a small number of parameters.")
#%%