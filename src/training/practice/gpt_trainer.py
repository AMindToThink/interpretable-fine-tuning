from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import torch

def train_gpt():
    # Load the ELI5 dataset
    dataset = load_dataset("sentence-transformers/eli5")
    
    # Initialize model and tokenizer
    model_name = "facebook/opt-350m"  # You can change this to your preferred base model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Format the dataset
    def format_instruction(example):
        """Format the question and answer into a single text string"""
        return {
            "text": f"### Human: {example['question']}\n\n### Assistant: {example['answer']}"
        }
    
    formatted_dataset = dataset["train"].map(format_instruction)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
    )
    
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model("./final_model")

if __name__ == "__main__":
    train_gpt()
