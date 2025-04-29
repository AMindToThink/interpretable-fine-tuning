# Import PeftConfig
from peft.config import PeftConfig
from peft.utils import PeftType

from dataclasses import asdict, dataclass, field
from typing import Union, List, Optional
# 1. Add your PEFT type to the PeftType enum
if not hasattr(PeftType, "ISAERFT"):
    # This adds the ISAERFT type to the PeftType enum
    PeftType.ISAERFT = "ISAERFT"
    
@dataclass
class IsaerftConfig(PeftConfig):
    """
    This class will hold the hyperparameters of an [`ISaeRFTModel`]
    
    Args:
        target_hooks (`List[tuple[str, str]]`):
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
    is_prompt_learning:bool=field(default=False, init=False, metadata={"help": "Whether this is a prompt learning method. ISaeRFT is not."})
    
    target_hooks: List[tuple[str, str]] = field(default_factory=list, metadata={'help':"List of (release, id) tuples to match SAEs, e.g. [('gemma-scope-2b-pt-res-canonical', 'layer_25/width_16k/canonical')] or [('res', '')] for all residual for that model"})
    depth: int = field(default=-1, metadata={"help": "How many layers for the FFNNs to have. -1 only trains a bias, 0 only trains a linear layer. defaults to -1."})
    hidden_size: Optional[int] = field(default=None, metadata={'help':'The hidden size of the FFNN. depth must be > 0 for this to be provided.'})
    lora_r: Optional[int] = field(default=None, metadata={'help':'The rank of the matrices for the FFNN. Depth must be greater than -1.'}) # Todo: find an actually good lora_r value here
    
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ISAERFT;
        if self.depth < 1 and (self.hidden_size is not None):
            raise ValueError("`hidden_size` must be None if there are no hidden parameters! (ie, when `depth` is less than 1)")

        if self.depth < 0 and (self.lora_r is not None):  
            raise ValueError("`lora_r` must be None if there are no matrices! (ie, when `depth` is less than 0)")
        if len(self.target_hooks) == 0:

            raise ValueError("`target_hooks` must have a value")

        if not (all(len(pair) == 2 for pair in self.target_hooks)):
            raise ValueError("`target_hooks` must be a list of pairs.")

        
if __name__ == "__main__":
    # Test valid configurations
    print("Testing valid configurations...")
    
    # Test basic config with bias only
    basic_config = IsaerftConfig(
        target_hooks=[("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post")]
    )
    print("Basic config (bias only) created successfully")

    # Test config with linear layer
    linear_config = IsaerftConfig(
        target_hooks=[("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post")],
        depth=0,
        lora_r=8
    )
    print("Linear layer config created successfully")

    # Test config with hidden layers
    ffnn_config = IsaerftConfig(
        target_hooks=[("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post")],
        depth=2,
        hidden_size=128,
        lora_r=8
    )
    print("FFNN config created successfully")

    # Test invalid configurations
    print("\nTesting invalid configurations...")
    
    try:
        # Should fail: hidden_size provided with bias-only
        IsaerftConfig(
            target_hooks=[("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post")],
            depth=-1,
            hidden_size=128
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Should fail: lora_r provided with bias-only
        IsaerftConfig(
            target_hooks=[("pythia-70m-deduped-res-sm", "blocks.4.hook_resid_post")],
            depth=-1,
            lora_r=8
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Should fail: empty target_hooks
        IsaerftConfig(
            target_hooks=[]
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Should fail: malformed target_hooks
        IsaerftConfig(
            target_hooks=[("only_one_element")]
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nAll tests completed!")



