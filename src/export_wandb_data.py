#%%
import wandb
import torch
from ISaeRFT_Interpreter import ISaeRFT_Interpreter
from sae_lens import SAE
#%%
api = wandb.Api()
run = api.run("/matthewkhoriaty-northwestern-university/huggingface/runs/fbpiuaue")
history_df = run.history()

# Filter columns to only get the scaling factors, excluding metadata columns
scaling_factor_cols = [col for col in history_df.columns 
                      if col.startswith('param/sae.trainable_ia3.scaling_factors/') 
                      and not any(x in col for x in ['histogram', 'max', 'min', 'mean', 'std', 'shape'])]
#%%
# Get the last row values for the filtered columns
last_values = history_df.iloc[-1][scaling_factor_cols]
#%%
# corresponding indices
indices = [int(col.split('/')[-1]) for col in scaling_factor_cols]

#%%
# Create a dictionary mapping indices to values
index_value_map = dict(zip(indices, last_values))

# Sort by indices
sorted_indices = sorted(indices)
last_values_sorted = [index_value_map[i] for i in sorted_indices]

#%%
# Convert pandas Series to torch tensor
last_values_tensor = torch.tensor(last_values_sorted, dtype=torch.float32)

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

#%%
identity_interpretations = interpreter.interpret_bias(last_values_tensor, 'identity', top_k=10, bottom_k=5)

#%%
# {'top_results': [{'interpretation_type': 'L2', 'rank': 0, 'index': 3969, 'value': 4.399526119232178, 'importance': 4.399526119232178, 'explanation': ['expressions of mixed opinions or sentiments, particularly those conveying ambivalence or contradiction']}, {'interpretation_type': 'L2', 'rank': 1, 'index': 8802, 'value': 4.160761833190918, 'importance': 4.160762310028076, 'explanation': [' punctuation, specifically semicolons and periods']}, {'interpretation_type': 'L2', 'rank': 2, 'index': 5149, 'value': 3.938446044921875, 'importance': 3.938446521759033, 'explanation': ['expressions of doubt or uncertainty']}, {'interpretation_type': 'L2', 'rank': 3, 'index': 12197, 'value': 3.6073243618011475, 'importance': 3.6073248386383057, 'explanation': [' programming constructs and declarations in code']}, {'interpretation_type': 'L2', 'rank': 4, 'index': 14398, 'value': 3.424738645553589, 'importance': 3.424738645553589, 'explanation': [' occurrences of the word "package" in programming context']}, {'interpretation_type': 'L2', 'rank': 5, 'index': 15662, 'value': 3.3516621589660645, 'importance': 3.3516621589660645, 'explanation': ['code structures and syntax indicative of programming or technical documentation']}, {'interpretation_type': 'L2', 'rank': 6, 'index': 10436, 'value': 3.3400869369506836, 'importance': 3.3400866985321045, 'explanation': ['code and programming constructs related to database operations and data manipulation']}, {'interpretation_type': 'L2', 'rank': 7, 'index': 14443, 'value': 3.2343146800994873, 'importance': 3.234314441680908, 'explanation': [' Java data structures and utility classes']}, {'interpretation_type': 'L2', 'rank': 8, 'index': 11925, 'value': -3.2114064693450928, 'importance': 3.2114062309265137, 'explanation': ['exclamatory expressions and emotional responses']}, {'interpretation_type': 'L2', 'rank': 9, 'index': 16126, 'value': 3.0998880863189697, 'importance': 3.0998876094818115, 'explanation': ['colons and their associated structures']}], 'bottom_results': [{'interpretation_type': 'L2', 'rank': 0, 'index': 9599, 'value': -0.005471332464367151, 'importance': 0.005471331533044577, 'explanation': ['expressions of mixed opinions or sentiments, particularly those conveying ambivalence or contradiction']}, {'interpretation_type': 'L2', 'rank': 1, 'index': 920, 'value': 0.002982641803100705, 'importance': 0.0029826422687619925, 'explanation': [' punctuation, specifically semicolons and periods']}, {'interpretation_type': 'L2', 'rank': 2, 'index': 3420, 'value': 0.0020860617514699697, 'importance': 0.002086061518639326, 'explanation': ['expressions of doubt or uncertainty']}, {'interpretation_type': 'L2', 'rank': 3, 'index': 5883, 'value': -0.001594418310560286, 'importance': 0.0015944184269756079, 'explanation': [' programming constructs and declarations in code']}, {'interpretation_type': 'L2', 'rank': 4, 'index': 1324, 'value': -0.0005368277779780328, 'importance': 0.0005368277779780328, 'explanation': [' occurrences of the word "package" in programming context']}]}

print("\nTOP RESULTS")
print("-" * 80)
for result in identity_interpretations['top_results']:
    print(f"Rank {result['rank']} (index {result['index']}):")
    print(f"Value: {result['value']:.4f}")
    print(f"Importance: {result['importance']:.4f}")
    print("Explanation:", result['explanation'][0])
    print()

print("\nBOTTOM RESULTS") 
print("-" * 80)
for result in identity_interpretations['bottom_results']:
    print(f"Rank {result['rank']} (index {result['index']}):")
    print(f"Value: {result['value']:.4f}")
    print(f"Importance: {result['importance']:.4f}")
    print("Explanation:", result['explanation'][0])
    print()

# %%
