#%%
# Import libraries
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch
assert 1 == torch.cuda.device_count()  
import wandb
import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from sae_lens import HookedSAETransformer
from huggingface_hub import login
import argparse

# Import our custom ISAERFT components
try:
    # When imported as a module
    from model_components.IsaerftConfig import IsaerftConfig
    from model_components.IsaerftPeft import IsaerftPeft
except ImportError:
    # When run directly as a script
    import sys
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_components.IsaerftConfig import IsaerftConfig
    from model_components.IsaerftPeft import IsaerftPeft

# Authenticate to Hugging Face and Wandb
login(token=os.environ['HUGGINGFACE_WRITE_KEY'])
wandb.login(key=os.environ['WANDB_KEY'])

# Load dataset
dataset = load_dataset(path="fnlp/moss-002-sft-data")

# Define the model
model_name = "EleutherAI/pythia-70m-deduped"

# Get the actual device that CUDA is using
if torch.cuda.is_available():
    device = f"cuda:{torch.cuda.current_device()}"
else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

assert 'cuda' in device
print(f"Using device: {device}")

# Model to fine-tune
model = HookedSAETransformer.from_pretrained(
    model_name,
).to(device)
#%%
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#%%
# # Set up chat template
# chat_template = """{% for message in messages %}
# {% if message['role'] == 'user' %}
# ### Instruction:
# {{ message['content'] }}
# {% elif message['role'] == 'assistant' %}
# ### Response:
# {{ message['content'] }}
# {% endif %}
# {% endfor %}
# {% if add_generation_prompt %}
# ### Response:
# {% endif %}"""
# tokenizer.chat_template = chat_template

# Apply ISAERFT to the model
example_releases_ids = {
    "EleutherAI/pythia-70m-deduped": ("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post"),
    "google/gemma-2-2b": ("gemma-scope-2b-pt-res-canonical", "layer_20/width_16k/canonical")
}

release, sae_id = example_releases_ids[model_name]
isaerft_config = IsaerftConfig(
    target_hooks=[
        example_releases_ids[model_name],
    ],
    depth=-1  # Bias-only for simplicity
)

# Apply the ISAERFT adapter
model = IsaerftPeft(model, isaerft_config)

# Set our name for the finetune to be saved &/ uploaded to
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
finetune_name = model_name.replace('/', '-') + f"_FT-SFT-ISAERFT_{timestamp}"
finetune_tags = ["smol-course", "module_1", "isaerft"]

# Add argument parser
parser = argparse.ArgumentParser(description='Fine-tune a model with ISAERFT')
parser.add_argument('--config', type=str, default='configs/default_hyperparams.json',
                   help='Path to hyperparameters config JSON file')
args = parser.parse_args()

# Load hyperparameters from JSON
with open(args.config, 'r') as f:
    hyperparams = json.load(f)

# Initialize wandb
wandb.init(
    project="isaerft-finetuning",
    tags=finetune_tags,
    config={
        "model_name": model_name,
        "learning_rate": hyperparams["learning_rate"],
        "max_steps": hyperparams["max_steps"],
        "batch_size": hyperparams["batch_size"],
        "gradient_accumulation_steps": hyperparams["gradient_accumulation_steps"],
        "sae_release": release,
        "sae_id": sae_id,
    }
)

# Prepare the dataset
train_dataset = dataset["train"]
if hyperparams["train_size"] > 0:
    train_dataset = train_dataset.select(range(min(hyperparams["train_size"], len(train_dataset))))

train_val_split = int(0.9 * len(train_dataset))
train_split = train_dataset.select(range(train_val_split))
val_split = train_dataset.select(range(train_val_split, len(train_dataset)))

# Preprocess the dataset to extract MOSS responses
def extract_moss_responses(example):
    text = example["plain_text"]
    # Split by [MOSS]: and <eoa> to get MOSS responses
    moss_parts = text.split("[MOSS]: ")
    
    # Skip the first part (before first [MOSS]:)
    moss_responses = []
    for i in range(1, len(moss_parts)):
        if "<eoa>" in moss_parts[i]:
            response = moss_parts[i].split("<eoa>")[0].strip()
            moss_responses.append(response)
    
    # Join all responses with newlines if there are multiple
    return {"text": "\n\n".join(moss_responses)}

# Apply preprocessing
train_dataset = train_split.map(extract_moss_responses)
val_dataset = val_split.map(extract_moss_responses)

# Configure SFT Trainer
sft_config = SFTConfig(
    output_dir=f"./models/{finetune_name}",
    max_steps=hyperparams['max_steps'],  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=hyperparams["batch_size"],  # Set according to your GPU memory capacity
    learning_rate=hyperparams['learning_rate'],  # Common starting point for fine-tuning
    gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
    gradient_checkpointing=False,
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=100,  # Frequency of saving model checkpoints
    evaluation_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=50,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
    report_to="wandb",
)

# Create SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Get the trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Number of trainable parameters: {len(trainable_params)}")
print(f"Shape of first trainable parameter: {trainable_params[0].shape}")

# Save the trainable parameter to a human-readable file
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
os.makedirs(output_dir, exist_ok=True)

# Save as JSON (human-readable)
param_file_path = f"{output_dir}/trainable_param_{datetime.now().strftime('%Y-%m-%d-%H:%M')}.json"
with open(param_file_path, 'w') as f:
    json.dump(param_data, f, indent=2)

print(f"Trainable parameter saved to {param_file_path}")

# Optional: Push to Hugging Face Hub
if os.environ.get('PUSH_TO_HUB', 'false').lower() == 'true':
    trainer.push_to_hub(tags=finetune_tags)
    print(f"Model uploaded to Hugging Face Hub as {finetune_name}")
else:
    print(f"Model saved locally to {trainer.args.output_dir}")

# Finish wandb run
wandb.finish()

# %%
