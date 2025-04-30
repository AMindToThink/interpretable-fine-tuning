# %% [markdown]
# Refer to https://colab.research.google.com/drive/1OLHNWBl6cA3h9KwoaFZx56YAwx_5S6B0?usp=sharing for a self-contained notebook for training Eleuther's Pythia 70m on https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca using IA3 acting on a single position.

# %%
%reload_ext autoreload
%autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''  # Use no GPUs because there's someone working on a thesis right now. Of course, change this to '0' or '1' later for real training 
import torch
print(torch.cuda.device_count())  # Should print 0

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add src directory to path

# %%
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import IA3Config, get_peft_model
import randomname
from sae_lens import SAE


# %%
from callbacks import *
from model_components import IsaerftIA3

# %%
save_path = '/results/IA3_Results/isaerft'

#%%

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

# %%
model_name = "EleutherAI/pythia-70m-deduped"

# %%
print(dataset[0])

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# %%
test_release = "pythia-70m-deduped-res-sm"
test_sae_id = "blocks.4.hook_resid_post"
test_sae, sae_dict, _ = SAE.from_pretrained(release=test_release, sae_id=test_sae_id)
#%%
# Print all parameters that require gradients
print("Parameters requiring gradients:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape}")

# %%
from copy import deepcopy
from torch import Tensor
from transformer_lens.hook_points import HookPoint
#%%
# Apply the configuration to your model
def prepare_model():
    result_model = deepcopy(model)
    # Freeze all parameters in the base model
    for param in result_model.parameters():
        param.requires_grad = False
    for param in test_sae.parameters():
        param.requires_grad = False
    # sae added to model, counts as a parameter
    result_model.sae = test_sae
    trainable_isaerftIA3 = IsaerftIA3(test_sae.cfg.d_sae, "sae_IA3_2025-04-29")
    test_sae.trainable_ia3 = trainable_isaerftIA3
    def ia3_hook(sae_acts:Tensor, hook:HookPoint) -> Tensor:
        """Runs the input through a trainable isaerft ia3.

        Args:
            sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
            hook (HookPoint): The transformer-lens hook point

        Returns:
            Tensor: The modified SAE activations modified by the trainable parameters.
        """

        return trainable_isaerftIA3()
    # TODO: Add the SAE
    # TODO: Put the hook into the SAE
    # TODO: Register the hook's trainable component as a parameter of the model
    # TODO: Check that the hook's trainable component is the only trainable parameter of the model.
    return result_model

# %%
# Print all modules with IA3 adapters
for name, module in peft_model.named_modules():
    if "ia3_" in name:
          print(name)

# %%
# Print all modules with IA3 adapters
for name, module in peft_model.named_parameters():
    if "ia3_" in name:
          print(name)

# %%


# %%
tracking_callback = PEFTParameterTrackingCallback()
histogram_callback = PEFTParameterHistogramCallback()

# %%
wandb.init(project="IA3_visualization")


# %%
import time
import randomname

wandb.finish()
# Get current timestamp in the desired format
timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

# Combine random name with timestamp
run_name = randomname.get_name() + "_" + timestamp


# %%
lets_overfit:bool = False
if lets_overfit:
  batch_size=64
  small_dataset = dataset.select(range(batch_size))
train_dataset = small_dataset if lets_overfit else dataset

# %%
# prompt: print the shapes and names of all peft_model parameters which require gradients

for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"Name: {name}, Shape: {param.shape}")


# %%

training_args = SFTConfig(
    max_length=512,
    output_dir=save_path + "/" + run_name,
    run_name=run_name,
    per_device_train_batch_size=64,
    logging_steps=50,
    learning_rate=5e-3,
    max_steps=5000

)
trainer = SFTTrainer(
    peft_model,
    train_dataset=train_dataset,
    args=training_args,
    callbacks=[tracking_callback, histogram_callback]
)
trainer.train()

# %%
dataset


