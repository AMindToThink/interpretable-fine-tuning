from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import transformers

dataset = load_dataset("tatsu-lab/alpaca", split="train")

model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-350m")

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input_text}
            
            ### Response:
            {response}
            '''
        else:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            {response}
            '''
        output_text.append(text)

    return output_text

training_args = SFTConfig(
    output_dir="/tmp",
    max_seq_length=256,
    packing=False,
)
trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

trainer.train()