# Following along with https://medium.com/myorder/fine-tuning-pythia-70m-deduped-instruction-following-llms-with-performance-evaluation-3bd0bb33b79
# %%
# %load_ext autoreload
# %autoreload 2
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
dataset = load_dataset(path="sentence-transformers/eli5")

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

# model, tokenizer = setup_chat_format(model, tokenizer)
#%%
# Set our name for the finetune to be saved &/ uploaded to
finetune_name = f"pythia-70m-isaerft-sft-eli5"
finetune_tags = ["pythia-70m", "sft", "isaerft", "eli5"]

#%%
# Inspect the dataset structure
print("Dataset structure:")
print(dataset["train"][0].keys())
print("\nSample entry:")
print(dataset["train"][0])

#%%
# Create train/validation split
train_val_dataset = dataset["train"].shuffle(seed=42)
train_size = int(0.9 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

train_dataset = train_val_dataset.select(range(train_size))
val_dataset = train_val_dataset.select(range(train_size, len(train_val_dataset)))

# Limit training data for faster training
train_dataset = train_dataset.select(range(min(5000, len(train_dataset))))
val_dataset = val_dataset.select(range(min(500, len(val_dataset))))

print(f"Training on {len(train_dataset)} examples")
print(f"Validating on {len(val_dataset)} examples")

#%%
# Prepare the dataset for training
def preprocess_function(examples):
    """Format examples into prompt-completion pairs and tokenize them."""
    # Format each example as a question-answer pair
    prompts = [f"### Instruction:\n{question}\n\n### Response:\n" for question in examples["question"]]
    answers = examples['answer']  # No need for list comprehension here
    
    # Create inputs for model training
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(len(prompts)):
        # Tokenize prompt and answer separately
        prompt_tokens = tokenizer(prompts[i], truncation=True, max_length=384, return_tensors="pt")
        answer_tokens = tokenizer(answers[i], truncation=True, max_length=128, return_tensors="pt")
        
        # Remove batch dimension
        prompt_input_ids = prompt_tokens["input_ids"][0]
        prompt_attention_mask = prompt_tokens["attention_mask"][0]
        answer_input_ids = answer_tokens["input_ids"][0]
        answer_attention_mask = answer_tokens["attention_mask"][0]
        
        # Combine prompt and answer tokens
        input_ids = torch.cat([prompt_input_ids, answer_input_ids])
        attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask])
        
        # Create labels: -100 for prompt tokens (ignored in loss), actual ids for answer tokens
        labels = torch.clone(input_ids)
        prompt_len = len(prompt_input_ids)
        labels[:prompt_len] = -100
        
        # Convert to lists and add to model_inputs
        model_inputs["input_ids"].append(input_ids.tolist())
        model_inputs["attention_mask"].append(attention_mask.tolist())
        # model_inputs["labels"].append(labels.tolist())
    
    return model_inputs

# Process the datasets
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Processing training dataset",
)

val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Processing validation dataset",
)

#%%
# Configure the training parameters
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir=f"./{finetune_name}",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    report_to="wandb",
    push_to_hub=True,
    hub_model_id=finetune_name,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked
)

#%%
# Initialize the standard Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#%%
# Start training
print("Starting training...")
trainer.train()
print("Training completed!")

#%%
# Initialize wandb
import wandb
from datetime import datetime
import uuid
wandb.finish()
wandb.login(key=os.environ['WANDB_KEY'])

wandb.init(
    project="pythia70m-sft-isaerft",
    name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:6]}",
    tags=finetune_tags
)
#%%


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