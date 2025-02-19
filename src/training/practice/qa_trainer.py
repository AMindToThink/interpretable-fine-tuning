from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
#%%
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def formatting_prompts_func(example):
    output_texts = []
    import pdb;pdb.set_trace()
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    # args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()