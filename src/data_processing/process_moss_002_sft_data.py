from datasets import load_dataset
import re
from tqdm import tqdm
import os
from huggingface_hub import login

# Login to Hugging Face
login(token=os.environ['HUGGINGFACE_WRITE_KEY'])

# Load the original dataset
dataset = load_dataset(path="fnlp/moss-002-sft-data")

def process_conversation(text):
    """Extract instruction-output pairs from a conversation."""
    # Define the pattern to match human and MOSS exchanges
    pattern = r'\[Human\]: (.*?)<eoh> \[MOSS\]: (.*?)<eoa>'
    
    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Convert matches to instruction-output pairs
    pairs = [{"instruction": human.strip(), "output": moss.strip()} 
             for human, moss in matches]
    
    return pairs

# Process the dataset
processed_data = []
for example in tqdm(dataset['train']):
    pairs = process_conversation(example['plain_text'])
    processed_data.extend(pairs)

# Create a new dataset from the processed data
from datasets import Dataset
new_dataset = Dataset.from_list(processed_data)

# Show some statistics
print(f"Original dataset size: {len(dataset['train'])}")
print(f"New dataset size: {len(new_dataset)}")
print(f"Average number of turns per conversation: {len(new_dataset)/len(dataset['train']):.2f}")

# Save the processed dataset locally
# new_dataset.save_to_disk("moss_002_instruction_output")

# Push to the Hugging Face Hub
new_dataset.push_to_hub("AMindToThink/moss-002-sft-data-instruction-output")

# import pdb;pdb.set_trace()