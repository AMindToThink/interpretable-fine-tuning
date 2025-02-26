# Import PeftConfig
from peft import PeftConfig

from dataclasses import asdict, dataclass, field
from typing import Union, List, Optional

@dataclass
class IsaerftConfig(PeftConfig):
    """
    This class will hold the hyperparameters of an [`ISaeRFTModel`]
    
    Args:
        target_hooks (`Union[List[str], str]`):
            Which of the HookedSAETransformer hook locations should have components added to them during fine-tuning
            # Todo: the neat pattern-matching thing for selecting layers. Also the "exclude_modules" analogue.
        depth (`int`):
            How many layers for the FFNNs to have. -1 only trains a bias, 0 only trains a linear layer. defaults to 1.
        hidden_size (`Optional[int]`):
            The hidden size of the FFNN. depth must be > 0 for this to be provided.
        lora_r (`Optional[int]`):
            The rank of the matrices for the FFNN. Depth must be greater than -1. # Todo: find good hyperparameters for the defaults
        # Todo: add args initialization type, dropout, checkpointing, and other parameters supported in the LoraConfig https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
    """
    
    # Fields with defaults must come after
    peft_type: str = field(default="ISAERFT", metadata={"help": "PEFT type"})
    task_type: str = field(default="CAUSAL_LM", metadata={"help": "Task type"})
    
    # IsaerftConfig specific fields
    target_hooks: Union[List[str], str] = field(default="", metadata={'help':"you really need to put something here, I just didn't want to deal with dataclass errors so I put the empty string as the default"})
    depth: int = field(default=-1, metadata={"help": "How many layers for the FFNNs to have. -1 only trains a bias, 0 only trains a linear layer. defaults to -1."})
    hidden_size: Optional[int] = field(default=None, metadata={'help':'The hidden size of the FFNN. depth must be > 0 for this to be provided.'})
    lora_r: Optional[int] = field(default=None, metadata={'help':'The rank of the matrices for the FFNN. Depth must be greater than -1.'}) # Todo: find an actually good lora_r value here
    
    def __post_init__(self):
        super().__post_init__()
        if self.depth < 1 and (self.hidden_size is not None):
            raise ValueError("`hidden_size` must be None if there are no hidden parameters! (ie, when `depth` is less than 1)")
        if self.depth < 0 and (self.lora_r is not None):  
            raise ValueError("`lora_r` must be None if there are no matrices! (ie, when `depth` is less than 0)")
        if isinstance(self.target_hooks, list) and len(self.target_hooks) == 0:
            raise ValueError("`target_hooks` must have a value")
        elif isinstance(self.target_hooks, str) and not self.target_hooks:
            raise ValueError("`target_hooks` must have a value")



