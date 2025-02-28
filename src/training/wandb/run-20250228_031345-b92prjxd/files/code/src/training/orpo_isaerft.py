# %%
%load_ext autoreload
%autoreload 2

import os
# YOU HAVE TO SET CUDA_VISIBLE_DEVICES BEFORE DOING ANY IMPORTS OF cuda-related packages! https://discuss.pytorch.org/t/setting-visible-devices-with-distributed-data-parallel/93230
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch
print(torch.cuda.device_count())  # Should print 1, but doesn't

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
from datetime import datetime
import uuid
run_name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:6]}"

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
# Apply the ISAERFT adapter
model = IsaerftPeft(model, isaerft_config)

#%%

# model, tokenizer = setup_chat_format(model, tokenizer)
#%%
# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "PYTHIA-FT-ORPO-ISAERFT_"+run_name
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
    output_dir="./results/orpo_isaerft/"+run_name,
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
# Initialize wandb
import wandb

# Make sure any previous wandb run is properly closed
wandb.finish()
wandb.login(key=os.environ['WANDB_KEY'])

wandb.init(
    project="pythia70m-orpo-isaerft",
    name=run_name,
    tags=finetune_tags
)
#%%
# Create the trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"].select(range(100)),
    eval_dataset=dataset["test"].select(range(10)),
    processing_class=tokenizer,
    # peft_config=isaerft_config, # don't include this; it is one or the other: model is a HookedSAETransformer and peft_config is used to transform it, or model is an IsaerftPeft and no peft_config needed
    # label_names=["labels"],  # This is the standard label name for causal language models
    # dataset_num_proc=1,
)

#%%
# Train the model
# b /home/cs29824/matthew/interpretable-fine-tuning/.venv/lib/python3.11/site-packages/transformer_lens/components/embed.py:34
# import pdb;pdb.set_trace()
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
print("You've successfully fine-tuned a HookedSAETransformer with ISAERFT!")
print("This approach allows you to align the model by editing interpretable parameters.")
#%%