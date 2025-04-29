# %% [markdown]
# Refer to https://colab.research.google.com/drive/1OLHNWBl6cA3h9KwoaFZx56YAwx_5S6B0?usp=sharing for a self-contained notebook for training Eleuther's Pythia 70m on https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca using IA3 acting on a single position.

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''  # Use no GPUs because there's someone working on a thesis right now. Of course, change this to '0' or '1' later for real training 
import torch
print(torch.cuda.device_count())  # Should print 0

# %%
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import IA3Config, get_peft_model
import randomname



# %%
try:
    # When imported as a module
    from ..callbacks import *
except ImportError:
    # When run directly as a script or in notebook
    import sys
    import os
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from callbacks import *



# %%
save_path = '/results/IA3_Results/isaerft'

# %%
import wandb
import torch
import numpy as np
from transformers import TrainerCallback

class PEFTParameterTrackingCallback(TrainerCallback):
    """
    A callback that tracks individual parameter values over time for PEFT methods.
    Optimized for tracking small parameter vectors where each element is important.
    """

    def __init__(self, peft_param_prefix=None):
        """
        Args:
            peft_param_prefix (list, optional): List of parameter name prefixes to track.
                If None, will use common PEFT parameter prefixes.
        """
        self.peft_param_prefix = peft_param_prefix or ["lora", "adapter", "prefix", "prompt", "ia3"]
        # Keep track of parameters we've seen to maintain consistent tracking
        self.tracked_params = {}

    def _is_peft_param(self, param_name):
        """Check if parameter is a PEFT parameter based on naming."""
        return any(prefix in param_name.lower() for prefix in self.peft_param_prefix)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize parameter tracking at the beginning of training."""
        if not model or not state.is_world_process_zero:
            return

        # Find trainable PEFT parameters
        for name, param in model.named_parameters():
            if self._is_peft_param(name) and param.requires_grad:
                param_data = param.data.detach().cpu().numpy().flatten()

                # Store initial parameter data for reference
                self.tracked_params[name] = {
                    'shape': param.shape,
                    'size': param.numel(),
                    'indices': list(range(len(param_data)))
                }

                # Log initial parameter values
                param_dict = {}
                for i, value in enumerate(param_data):
                    param_dict[f"param/{name}/{i}"] = float(value)

                wandb.log(param_dict, step=0)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log individual parameter values at logging steps."""
        if not model or not state.is_world_process_zero or not wandb.run:
            return

        # Dictionary to log current parameter values
        param_dict = {}

        # Track trainable PEFT parameters that we identified at the beginning
        for name, param in model.named_parameters():
            if name in self.tracked_params and param.requires_grad:
                param_data = param.data.detach().cpu().numpy().flatten()

                # Log each individual parameter value
                for i, value in enumerate(param_data):
                    param_dict[f"param/{name}/{i}"] = float(value)

                # If parameter has a gradient, track that too
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy().flatten()
                    for i, value in enumerate(grad_data):
                        param_dict[f"grad/{name}/{i}"] = float(value)

        # Log all the values
        wandb.log(param_dict, step=state.global_step)

# %%

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
dataset[0]

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# %%
model

# %%
from peft import IA3Config, get_peft_model

# Create IA3 configuration with precise targeting for ONLY the 4th MLP layer (index 3)
config = IA3Config(
    task_type="CAUSAL_LM",
    # Use a list of exact module names to target
    target_modules=["gpt_neox.layers.3.mlp.dense_h_to_4h"],
    # Similarly exact name for feedforward module
    feedforward_modules=["gpt_neox.layers.3.mlp.dense_h_to_4h"],
    init_ia3_weights=True
)

# Apply the configuration to your model
peft_model = get_peft_model(model, config)

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


