# %%
# %load_ext autoreload
# %autoreload 2

import os
# YOU HAVE TO SET CUDA_VISIBLE_DEVICES BEFORE DOING ANY IMPORTS OF cuda-related packages! https://discuss.pytorch.org/t/setting-visible-devices-with-distributed-data-parallel/93230
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true' # if the process hangs, turn this to false.
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
model_name = "google/gemma-2-2b" 
simpler_model_name = model_name.split('/')[1]
from datetime import datetime

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
hooked_sae_transformer = HookedSAETransformer.from_pretrained(
    model_name,
    # torch_dtype=torch.float32,
    # device_map=device
).to(device)
assert isinstance(hooked_sae_transformer, HookedSAETransformer)
# model.config.use_cache = False
# tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')#(model_name)
#%%
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
# tokenizer.pad_token = tokenizer.eos_token

# tokenizer.chat_template = chat_template
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

example_releases_ids = {
    "EleutherAI/pythia-70m-deduped":("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post"),
    "google/gemma-2-2b": ("gemma-scope-2b-pt-res-canonical","layer_20/width_16k/canonical")} 
isaerft_config = IsaerftConfig(
    target_hooks=[
        example_releases_ids[model_name],
    ],
    depth=-1  # Bias-only for simplicity
)
#%%


#%%

# model, tokenizer = setup_chat_format(model, tokenizer)
#%%
# Set our name for the finetune to be saved &/ uploaded to
run_name=f"run-{simpler_model_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
finetune_name = f"{simpler_model_name.upper()}-FT-ORPO-ISAERFT_"+run_name
finetune_tags = ["smol-course", "module_1", "isaerft"]

#%%
# Define sweep configuration
import wandb

sweep_config = {
    'method': 'random',  # Random search over the parameter space
    'metric': {
        'name': 'eval/rewards/margins',  # Metric to optimize
        'goal': 'maximize'    # We want to maximize the reward margin
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 5e-5
        },
        'beta': {
            'values': [0.05, 0.1, 0.15, 0.2]  # Different beta values to try
        }
    }
}

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="orpo-isaerft-sweep")

# Define the training function
def train_model(config=None):
    with wandb.init(config=config, project="orpo-isaerft-sweep", tags=finetune_tags) as run:
        config = wandb.config
        assert isinstance(hooked_sae_transformer, HookedSAETransformer)
        hooked_sae_transformer.reset_saes()
        # Apply the ISAERFT adapter
        model = IsaerftPeft(hooked_sae_transformer, isaerft_config)
        # Get a unique name for this run
        current_run_name = f"run-{simpler_model_name}-lr{config.learning_rate:.1e}-beta{config.beta}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        current_finetune_name = f"{simpler_model_name.upper()}-FT-ORPO-ISAERFT_{current_run_name}"
        
        # Train model with ORPO
        orpo_args = ORPOConfig(
            # Use the learning rate from sweep config
            learning_rate=config.learning_rate,
            # Linear learning rate decay over training
            lr_scheduler_type="linear",
            # Maximum combined length of prompt + completion
            max_length=1024,
            # Maximum length for input prompts
            max_prompt_length=512,
            # Controls weight of the odds ratio loss (λ in paper) - from sweep config
            beta=config.beta,
            # Batch size for training
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            # Helps with training stability by accumulating gradients before updating
            gradient_accumulation_steps=8,
            # Memory-efficient optimizer for CUDA, falls back to adamw_torch for CPU/MPS
            optim="paged_adamw_8bit" if ("cuda" in device) else "adamw_torch",
            # When to run evaluation
            eval_strategy="steps",
            # Evaluate every 20% of training
            eval_steps=0.2,
            # Log metrics every step
            logging_steps=1,
            # Gradual learning rate warmup
            warmup_steps=10,
            # Use wandb for logging
            report_to="wandb",
            # Where to save model/checkpoints
            output_dir=f"./results/orpo_isaerft/{current_run_name}",
            # Enable MPS (Metal Performance Shaders) if available
            use_mps_device=device == "mps",
            hub_model_id=current_finetune_name,
            # Training for a shorter time for this example
            num_train_epochs=2,
            # Ensure device placement is correct
            no_cuda=False,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            dataloader_num_workers=4,
        )
        
        # Create the trainer
        trainer = ORPOTrainer(
            model=model,
            args=orpo_args,
            train_dataset=dataset["train"].select(range(1000)),
            eval_dataset=dataset["test"].select(range(100)),
            processing_class=tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(f"./results/{current_finetune_name}")
        
        # Only push the best model to hub (optional)
        # You could add logic here to only push if this is the best run so far
        try:
            print("Pushing model to hub...")
            trainer.push_to_hub(tags=finetune_tags + [f"lr_{config.learning_rate}", f"beta_{config.beta}"])
            print("Successfully pushed to hub!")
        except Exception as e:
            print(f"Error pushing to hub: {str(e)}")
            # Alternative manual push
            print("Attempting manual push...")
            model.push_to_hub(current_finetune_name, tags=finetune_tags + [f"lr_{config.learning_rate}", f"beta_{config.beta}"])
            tokenizer.push_to_hub(current_finetune_name, tags=finetune_tags + [f"lr_{config.learning_rate}", f"beta_{config.beta}"])
            print("Manual push completed!")

# Run the sweep
wandb.agent(sweep_id, train_model, count=10)  # Run 10 experiments

print("## 💐 Sweep completed!")
print("You've successfully run a hyperparameter sweep for fine-tuning a HookedSAETransformer with ISAERFT!")
print("Check your wandb dashboard to see the results and find the best hyperparameters.")
#%%