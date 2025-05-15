# %% [markdown]
# Refer to https://colab.research.google.com/drive/1OLHNWBl6cA3h9KwoaFZx56YAwx_5S6B0?usp=sharing for a self-contained notebook for training Eleuther's Pythia 70m on https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca using IA3 acting on a single position.

# %%
# %reload_ext autoreload
# %autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 
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

import NeuronpediaClient
#%%
neuronpedia_api_key = os.environ['NEURONPEDIA_API_KEY']
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
#%%
# Verify trainable parameters
print("\nTrainable parameters:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"Name: {name}, Shape: {param.shape}")



# %%
myIsaerftIA3 = next(iter(peft_model.trainable_blocks.items()))[1]
tracking_callback = PEFTParameterTrackingCallback(peft_param_prefix=['trainable_blocks'], get_param_desc=myIsaerftIA3.get_feature_description)
histogram_callback = PEFTParameterHistogramCallback(peft_param_prefix=['trainable_blocks'])
del myIsaerftIA3

# %%
import time
import randomname

# Define the hyperparameter sweep configuration
sweep_config = {
    'method': 'bayes',  # Use Bayesian optimization for efficient exploration
    'metric': {
        'name': 'train_loss',  # Use train loss as the primary metric
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 5e-4,
            'max': 1e-1,
            'distribution': 'log_uniform_values'
        },
        'weight_decay': {
            'min': 0.0,
            'max': 0.1,
            'distribution': 'uniform'
        },
        'adam_beta1': {
            'min': 0.8,
            'max': 0.99,
            'distribution': 'uniform'
        },
        'adam_beta2': {
            'min': 0.9,
            'max': 0.999,
            'distribution': 'uniform'
        },
        'gradient_accumulation_steps': {
            'values': [2, 4, 8, 16]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="isaerft-dpo-sweep")

def doDPO(config=None):
    # Initialize wandb run for this trial
    with wandb.init(config=config) as run:
        peft_model.setup_trainable_blocks()
        # Get hyperparameters for this run
        config = wandb.config
        
        # Get current timestamp in the desired format
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        
        # Combine random name with timestamp
        run_name = randomname.get_name() + "_" + timestamp
        
        # Load a preference dataset for DPO training
        dataset = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style")
        
        # Verify dataset structure
        print(f"Dataset keys: {dataset.keys()}")
        print(f"Train dataset size: {len(dataset['train'])}")
        print(f"Test dataset size: {len(dataset['test'])}")

        if tokenizer.chat_template is None:
            tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        # Configure DPO training with hyperparameters from sweep
        training_args = DPOConfig(
            max_length=512,
            output_dir=save_path + "/" + run_name + "_dpo",
            run_name=run_name + "_dpo", 
            per_device_train_batch_size=4,
            logging_steps=2,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            do_eval=False,  # Disable evaluation
            max_steps=4,#500//config.gradient_accumulation_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            bf16=True,
            logging_first_step=True,
            report_to="wandb",
            beta=0.1,  # DPO-specific parameter (keeping fixed as requested)
            loss_type="sigmoid",  # Default DPO loss type
            # Remove evaluation settings
            save_strategy="steps",
            save_steps=100,
            metric_for_best_model="train_loss",
            greater_is_better=False,
        )

        # Initialize DPO trainer
        trainer = DPOTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset["train"],
            processing_class=tokenizer,
            callbacks=[tracking_callback, histogram_callback]
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(training_args.output_dir)

# Run the sweep
wandb.agent(sweep_id, function=doDPO, count=20)  # Run 20 trials