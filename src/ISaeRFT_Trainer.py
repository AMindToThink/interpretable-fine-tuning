from transformers import Trainer, TrainingArguments
from torch.nn import Module
from typing import List, Union, Dict
import torch
from sae_lens import SAE, HookedSAETransformer
import model_components
from model_components import ResidualBlock
import steering_functions

class ISaeRFT_Trainer(Trainer):
    """Custom trainer for training the SAE resnets"""
    
    def __init__(
        self,
        model: Module,
        resnet_list: List[ResidualBlock],
        # trainable_saes: List[tuple[str, str]],
        *args,
        **kwargs
    ):
        """
        Initialize the custom trainer
        
        Args:
            model: The pre-trained model to fine-tune
            trainable_saes: List of tuples (layer_name, sae_name) to train. If None, all parameters will be trainable
            *args, **kwargs: Additional arguments passed to the Trainer base class
        """
        super().__init__(model=model, *args, **kwargs)
        self.resnet_list = resnet_list
        self._freeze_layers()
        # self._make_rft_saes()
        

    # def _make_rft_saes(self):
    #     self.saes = [SAE.from_pretrained(
    #                                                 sae_release, 
    #                                                 sae_id, 
    #                                                 device=str(self.device)
    #                                             )[0] 
    #                             for sae_release, sae_id in trainable_saes]
    #     self.trainable_residual_blocks = []
    #     for sae in self.saes:
            


    def _freeze_layers(self):
        """
        Freeze all layers, not those in the SAEs.
        
        """
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    
    def get_trainable_parameters(self) -> Dict[str, bool]:
        """
        Get a dictionary of parameter names and their trainable status
        
        Returns:
            Dict mapping parameter names to boolean indicating if they're trainable
        """
        trainable_params = {}
        for resnet in self.resnet_list:
            for name, param in resnet.named_parameters():
                param_name = f"{resnet.name}.{name}"
                assert param_name not in trainable_params, "Be sure to give each resnet a unique name."
                trainable_params[param_name] = param.requires_grad
        return trainable_params

# Example usage:
def setup_fine_tuning(
    model_name: str,
    trainable_saes: List[tuple[str, str]],
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 3
):
    """
    Set up fine-tuning with custom trainable parameters
    
    Args:
        model_name: Name of the pre-trained model
        trainable_saes: List of tuples (layer_name, sae_name) to train
        output_dir: Directory to save results
        batch_size: Training batch size
        num_epochs: Number of training epochs
    """
    # Load model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=f"{output_dir}/logs",
    )
    
    # Initialize custom trainer
    trainer = ISaeRFT_Trainer(
        model=model,
        trainable_saes=trainable_saes,
        args=training_args,
        # Add other necessary arguments like:
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # data_collator=data_collator,
    )
    
    # Print trainable parameters
    trainable_params = trainer.get_trainable_parameters()
    print("\nTrainable parameters:")
    for name, is_trainable in trainable_params.items():
        if is_trainable:
            print(f"✓ {name}")
    
    return trainer, model, tokenizer

# Example of how to use:
if __name__ == "__main__":
    # Example: Only fine-tune specific SAEs in specific layers
    trainable_saes = [
        ("layer.11", "sae_0"),  # SAE 0 in last transformer layer
        ("layer.10", "sae_1"),  # SAE 1 in second-to-last layer
        ("classifier", "linear")  # Keep classifier trainable
    ]
    
    trainer, model, tokenizer = setup_fine_tuning(
        model_name="bert-base-uncased",
        trainable_saes=trainable_saes,
        output_dir="./fine_tuned_model"
    )
    
    # trainer.train() would start the training