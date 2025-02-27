# %%
# uncomment these when in notebook mode 
# %load_ext autoreload
# %autoreload 2
#%%
# Import libraries
from tqdm import tqdm
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
import wandb  # Add wandb import
import numpy as np
import json
from datetime import datetime  # Add datetime import
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
wandb.login(key=os.environ['WANDB_KEY'])
#%%
# Load dataset
dataset = load_dataset(path="fnlp/moss-002-sft-data")

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
# Apply the ISAERFT adapter
model = IsaerftPeft(model, isaerft_config)
#%%
# Check device placement of model components

# model, tokenizer = setup_chat_format(model, tokenizer)
#%%
# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "PYTHIA-FT-SFT-ISAERFT"
finetune_tags = ["smol-course", "module_1", "isaerft"]

#%%
# Initialize wandb
wandb.init(
    project="isaerft-finetuning",
    name=finetune_name,
    tags=finetune_tags,
    config={
        "model_name": model_name,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "sae_release": release,
        "sae_id": sae_id,
    }
)

#%%
# Create a simple training loop without preprocessing

# Training hyperparameters
hyperparams = {
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "train_size": 100
}

# Prepare optimizer - only optimize the trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=hyperparams["learning_rate"])

# Use the full training dataset
train_dataset = dataset["train"].select(range(hyperparams["train_size"]))

# Create a validation set (10% of the training data)
train_val_split = int(0.9 * len(train_dataset))
val_dataset = train_dataset.select(range(train_val_split, len(train_dataset)))
train_dataset = train_dataset.select(range(train_val_split))

# Training loop
model.train()
for epoch in range(hyperparams["num_epochs"]):
    total_loss = 0
    
    # Process data in batches
    for i in tqdm(range(0, len(train_dataset), hyperparams["batch_size"])):
        batch_data = train_dataset[i:i+hyperparams["batch_size"]]
        
        # Use plain_text directly instead of instruction/response
        texts = batch_data["plain_text"]
        
        # Tokenize directly without chat formatting
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # inputs["labels"] = inputs["input_ids"].clone()
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / hyperparams["gradient_accumulation_steps"]
        loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if ((i // hyperparams["batch_size"]) + 1) % hyperparams["gradient_accumulation_steps"] == 0 or i >= len(train_dataset) - hyperparams["batch_size"]:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * hyperparams["gradient_accumulation_steps"]
        
        # Print progress
        if (i // hyperparams["batch_size"]) % 10 == 0:
            # print(f"Epoch {epoch+1}/{hyperparams['num_epochs']} | Batch {i//hyperparams['batch_size']}/{len(train_dataset)//hyperparams['batch_size']} | Loss: {loss.item() * hyperparams['gradient_accumulation_steps']:.4f}")
            # Log training loss to wandb
            wandb.log({"train_loss": loss.item() * hyperparams["gradient_accumulation_steps"], 
                      "epoch": epoch + (i/len(train_dataset))})
    
    avg_loss = total_loss / (len(train_dataset) // hyperparams["batch_size"])
    print(f"Epoch {epoch+1}/{hyperparams['num_epochs']} completed | Average Loss: {avg_loss:.4f}")
    
    # Validation after each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_dataset), hyperparams["batch_size"]):
            batch_data = val_dataset[i:i+hyperparams["batch_size"]]
            texts = batch_data["plain_text"]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / (len(val_dataset) // hyperparams["batch_size"])
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Log epoch metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_epoch_loss": avg_loss,
        "val_epoch_loss": avg_val_loss
    })
    
    # Switch back to training mode
    model.train()

#%%
print(len(trainable_params))
print(trainable_params[0].shape)

#%%
# Save the trainable parameter to a human-readable file

# Get the trainable parameter and convert to numpy
param_tensor = trainable_params[0].detach().cpu().numpy()
param_shape = param_tensor.shape
param_list = param_tensor.tolist()  # Convert to list for JSON serialization

# Create a dictionary with metadata and values
param_data = {
    "shape": list(param_shape),
    "hyperparameters": hyperparams,
    "values": param_list,
    "description": f"ISAERFT bias vector for {release}/{sae_id}"
}
output_dir = 'results'
# Save as JSON (human-readable)
param_file_path = f"{output_dir}/trainable_param_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.json"
with open(param_file_path, 'w') as f:
    json.dump(param_data, f, indent=2)

print(f"Trainable parameter saved to {param_file_path}")

# Example of how to load it back:
def load_param_from_json(file_path):
    with open(file_path, 'r') as f:
        param_data = json.load(f)
    
    # Convert back to tensor
    param_tensor = torch.tensor(param_data["values"]).reshape(param_data["shape"])
    return param_tensor

# # Save the fine-tuned model
# output_dir = f"./models/{finetune_name}"
# os.makedirs(output_dir, exist_ok=True)
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# # Optional: Upload to Hugging Face Hub
# model.push_to_hub(finetune_name, tags=finetune_tags, token=os.environ['HUGGINGFACE_WRITE_KEY'])
# tokenizer.push_to_hub(finetune_name, tags=finetune_tags,token=os.environ['HUGGINGFACE_WRITE_KEY'])

# Finish wandb run
wandb.finish()

print(f"Model saved to {output_dir} and uploaded to Hugging Face Hub as {finetune_name}")

# %%
# import pdb;pdb.set_trace()