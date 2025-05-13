# %% [markdown]
# Refer to https://colab.research.google.com/drive/1OLHNWBl6cA3h9KwoaFZx56YAwx_5S6B0?usp=sharing for a self-contained notebook for training Eleuther's Pythia 70m on https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca using IA3 acting on a single position.

# %%
# %reload_ext autoreload
# %autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 
import torch
print("number of cuda devices visible: ",  torch.cuda.device_count())  # Should print 1

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add src directory to path

# %%
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, ModelConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import CodeCarbonCallback
from peft import IA3Config, get_peft_model
import randomname
from sae_lens import SAE, HookedSAETransformer
from functools import partial
import wandb
# %%
from callbacks import *
from model_components.IsaerftPeft import IsaerftPeft
from model_components.IsaerftConfig import IsaerftConfig
from copy import deepcopy
from torch import Tensor
from transformer_lens.hook_points import HookPoint

# %%
save_path = 'results/IA3_Results/isaerft'

#%%

# %%
model_name = "google/gemma-2-2b"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model using HookedSAETransformer instead of AutoModelForCausalLM
model = HookedSAETransformer.from_pretrained(model_name, device=device).to(device)
print(f"Model loaded: {model.cfg.model_name}")

# %%
sae_release = "gemma-scope-2b-pt-res-canonical"
model_sae_id = 'layer_20/width_16k/canonical'

# %%
# Create IsaerftConfig
config = IsaerftConfig(
    target_hooks=[
        (sae_release, model_sae_id),  # Match the SAE we loaded
    ],
    depth=None,  
    ia3=True
)

# Create IsaerftPeft model
peft_model = IsaerftPeft(model, config)

# Verify trainable parameters
print("\nTrainable parameters:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"Name: {name}, Shape: {param.shape}")

# %%
tracking_callback = PEFTParameterTrackingCallback(peft_param_prefix=['trainable_blocks'])
histogram_callback = PEFTParameterHistogramCallback(peft_param_prefix=['trainable_blocks'])

# %%
wandb.init(project="ISAERFT_visualization")

# %%
import time
import randomname

wandb.finish()
# Get current timestamp in the desired format
timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

# Combine random name with timestamp
run_name = randomname.get_name() + "_" + timestamp

# %%
def doSFT():
    dataset = load_dataset("iamtarun/code_instructions_120k_alpaca", split="train")

    # %%
    def preprocess_function(example):
        example['prompt'] = example['instruction'] + "\ninput:\n" + example['input']
        example['completion'] = example['output']
        return example

    # %%
    # make the dataset a prompt-completion dataset https://huggingface.co/docs/trl/en/dataset_formats
    dataset = dataset.map(preprocess_function)

    # %%
    dataset = dataset.select_columns(['prompt', 'completion'])
    lets_overfit:bool = False
    if lets_overfit:
        batch_size=64
        small_dataset = dataset.select(range(batch_size))
    train_dataset = small_dataset if lets_overfit else dataset
    training_args = SFTConfig(
        max_length=512,
        output_dir=save_path + "/" + run_name,
        run_name=run_name,
        per_device_train_batch_size=8,
        logging_steps=50,
        learning_rate=5e-3,
        max_steps=5000

    )
    trainer = SFTTrainer(
        peft_model,
        train_dataset=train_dataset,
        args=training_args,
        callbacks=[tracking_callback, histogram_callback, CodeCarbonCallback()]
    )
    trainer.train()

def doDPO():
    # Initialize wandb project
    wandb.init(
        project="isaerft-dpo",
    )

    # Load a preference dataset for DPO training
    dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style")

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Configure DPO training
    training_args = DPOConfig(
        max_length=512,
        output_dir=save_path + "/" + run_name + "_dpo",
        run_name=run_name + "_dpo", 
        per_device_train_batch_size=4,
        logging_steps=5,
        learning_rate=5e-3,
        max_steps=5000,
        gradient_accumulation_steps=2,
        bf16=True,
        logging_first_step=True,
        report_to="wandb",
        beta=0.1,  # DPO-specific parameter controlling deviation from reference model
        loss_type="sigmoid"  # Default DPO loss type
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=[tracking_callback, histogram_callback, CodeCarbonCallback()]
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)

    # Close wandb run
    wandb.finish()

#%%
doDPO()