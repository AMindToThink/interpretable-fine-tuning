from torch import Tensor
from transformer_lens.hook_points import HookPoint
from model_components import BiasOnly, ResidualBlock
def debug_steer(sae_acts: Tensor, hook:HookPoint) -> Tensor:
    import pdb; pdb.set_trace()
    pass
    pass
    return sae_acts

def resid_hook(sae_acts:Tensor, hook:HookPoint, residual_block:ResidualBlock) -> Tensor:
    """Runs the input through a trainable resnet (ResidualBlock).

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point

    Returns:
        Tensor: The modified SAE activations modified by the trainable parameters.
    """

    return residual_block(sae_acts)