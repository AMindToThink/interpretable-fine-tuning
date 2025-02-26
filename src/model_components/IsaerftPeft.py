# from elsewhere
import torch
from torch import nn
from peft import PeftModel
from sae_lens import HookedSAETransformer
from peft.utils import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
# from my folder
from IsaerftConfig import IsaerftConfig

# 1. Add your PEFT type to the PeftType enum
if not hasattr(PeftType, "ISAERFT"):
    # This adds the ISAERFT type to the PeftType enum
    PeftType.ISAERFT = "ISAERFT"

class IsaerftModel(nn.Module):
    """Implementation of the ISAERFT model"""
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.peft_config = {adapter_name: config}
        # Implement your custom PEFT logic here
        
    # Implement necessary methods that BaseTuner would have

class IsaerftPeft(PeftModel):
    def __init__(self, model, config: IsaerftConfig, adapter_name="default"):
        # 2. Register your model class in the mapping
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ISAERFT] = IsaerftModel
        
        super().__init__(model, config, adapter_name)


if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b")
    config = IsaerftConfig(target_hooks='mlp', depth=-1)
    peft = IsaerftPeft(model, config)
    import pdb;pdb.set_trace()