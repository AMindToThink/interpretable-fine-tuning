from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from transformers.trainer_callback import TrainerCallback
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize wandb (set your project name and optionally entity)
wandb.init(project="assistant-training")

class WandbLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Log the metrics (such as the loss) to wandb with the current global step.
            wandb.log(logs, step=state.global_step)
            # Log the training loss
            wandb.log({'train/loss': logs['loss']})

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model,
    args=SFTConfig(
        output_dir="/tmp",
        max_seq_length=512,  # Increase this value (adjust based on your model's limits)
        logging_steps=10,
        report_to=["wandb"]
    ),
    train_dataset=dataset,
    data_collator=collator,
    callbacks=[WandbLossCallback()]
)

trainer.train()