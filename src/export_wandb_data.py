import wandb
import torch
from ISaeRFT_Interpreter import ISaeRFT_Interpreter
from sae_lens import SAE

api = wandb.Api()
run = api.run("/matthewkhoriaty-northwestern-university/huggingface/runs/fbpiuaue")
history_df = run.history()

# Filter columns to only get the scaling factors, excluding metadata columns
scaling_factor_cols = [col for col in history_df.columns 
                      if col.startswith('param/sae.trainable_ia3.scaling_factors/') 
                      and not any(x in col for x in ['histogram', 'max', 'min', 'mean', 'std', 'shape'])]

# Get the last row values for the filtered columns
last_values = history_df.iloc[-1][scaling_factor_cols]

# Convert pandas Series to torch tensor
last_values_tensor = torch.tensor(last_values.values, dtype=torch.float32)

# Initialize SAE and interpreter
sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_20/width_16k/canonical", device='cpu')[0]
interpreter = ISaeRFT_Interpreter(sae)

# Get interpretations using different methods
l2_interpretations = interpreter.interpret_bias(last_values_tensor, 'L2', top_k=10, bottom_k=5)
absolute_interpretations = interpreter.interpret_bias(last_values_tensor, 'absolute', top_k=10, bottom_k=5)

print("\nL2 Interpretations:")
print(l2_interpretations)
print("\nAbsolute Interpretations:")
print(absolute_interpretations)

# %%
# Get top 10 and bottom 10 columns by final value
last_values = history_df.iloc[-1][history_df.columns[history_df.columns.str.startswith('param/sae.trainable_ia3.scaling_factors/')]]
sorted_values = last_values.sort_values()

print("Bottom 10 columns by final value:")
for col, val in sorted_values.head(10).items():
    print(f"{col}: {val}")

print("\nTop 10 columns by final value:")
for col, val in sorted_values.tail(10).items():
    print(f"{col}: {val}")
# %%

