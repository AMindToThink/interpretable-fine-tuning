import torch
from typing import Callable, Optional, List, Dict
from peft.tuners.tuners_utils import BaseTunerLayer
import torch.nn as nn
import os
import sys
from sae_lens import SAE

# Add parent directory to path to make absolute imports work
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import can work properly
from NeuronpediaClient import NeuronpediaClient

def grad_1_abs(x):
    """A version of absolute value which has a gradient of 1 when x_i = 0 instead of 0. Necessary so that we can initialize our parameters to 0 without losing all our gradients."""
    return torch.where(x < 0, -x, x)

class IsaerftIA3(nn.Module, BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("scaling_factors",)

    def __init__(self, sae: SAE, name:str="", scale_processor:Callable[[torch.Tensor], torch.Tensor]=lambda x:x):
        """Scales the features given to it by multiplying elementwise by a learned vector, scaling_factors. Naturally, scaling_factors is initialized to 1s so that this component acts as the identity during the start of training.
        
        Args:
            sae (SAE): The SAE object whose features will be scaled.
            name (str, optional): Name for this layer. Defaults to "".
            scale_processor: Function which processes the scales and returns new scales. Useful for ensuring that only some values are possible (for example, only positive or only negative). For example, making it torch.abs would break because its derivative is 0 at x=0. 
        
        Example:
            >>> device = "cuda" if torch.cuda.is_available() else "cpu"
            >>> sae_release = 'gemma-scope-2b-pt-res-canonical'
            >>> sae_id = 'layer_20/width_16k/canonical'
            >>> sae_20 = SAE.from_pretrained(sae_release, sae_id, device=str(device))[0] 
            >>> layer = IsaerftIA3(sae=sae_20)  # Create layer with SAE
            >>> x = torch.randn(32, sae_20.cfg.d_sae)  # batch_size=32, features=d_sae
            >>> output = layer(x)  # Shape: (32, d_sae), initially equal to x since scaling_factors starts as ones

        """
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)
        self.name = name
        self.scale_processor = scale_processor
        self.saes = (sae,) # Tuple so that torch doesn't pull any shenanigans
        self.num_features = sae.cfg.d_sae
        self.initialize_parameters()
        self._disable_adapters = False
        self.merged_adapters = []
        
        # Initialize Neuronpedia client if API key is available
        self.feature_descriptions = {}
        self._init_neuronpedia()
    
    def initialize_parameters(self):
        self.scaling_factors = torch.nn.Parameter(torch.zeros(self.num_features))
    
    def _init_neuronpedia(self):
        """Initialize Neuronpedia client if API key is available in environment variables."""
        api_key = os.environ.get("NEURONPEDIA_API_KEY")
        if api_key:
            try:
                self.neuronpedia_client = NeuronpediaClient(api_key)
                
                # Get the SAE
                sae = self.saes[0]
                
                # Use the neuronpedia_id directly if available
                if hasattr(sae.cfg, 'neuronpedia_id') and sae.cfg.neuronpedia_id:
                    # neuronpedia_id typically has format "model-id/identifier"
                    parts = sae.cfg.neuronpedia_id.split('/')
                    if len(parts) >= 2:
                        model_id = parts[0]
                        sae_id = '/'.join(parts[1:])
                        self._fetch_feature_descriptions(model_id, sae_id)
            except Exception as e:
                print(f"Error initializing Neuronpedia client: {e}")
                self.neuronpedia_client = None
        else:
            print("No NEURONPEDIA_API_KEY found in environment variables")
            self.neuronpedia_client = None
    
    def _fetch_feature_descriptions(self, model_id: str, sae_id: str):
        """Fetch feature descriptions from Neuronpedia."""
        if self.neuronpedia_client:
            try:
                descriptions = self.neuronpedia_client.get_feature_desc_dict(model_id, sae_id)
                if descriptions and isinstance(descriptions, list):
                    # Create a dictionary mapping index to description
                    self.feature_descriptions = {int(x['index']): x for x in descriptions}
            except Exception as e:
                print(f"Error fetching feature descriptions: {e}")
    
    def get_feature_description(self, index: tuple) -> Optional[str]:
        """Get the description for a specific feature by index."""
        assert len(index) == 1, "Index for a one dimensional vector like IsaerftIA3 must have len of 1."
        if not self.feature_descriptions:
            return None
        
        feature_data = self.feature_descriptions.get(index[0])
        if not feature_data:
            return None
        
        # Try to extract description from feature data
        if 'description' in feature_data:
            return feature_data['description']
        elif 'explanations' in feature_data and feature_data['explanations']:
            return feature_data['explanations'][0].get('description')
        
        return None
    
    def forward(self, x):
        if self._disable_adapters:
            return x
        return (self.scale_processor(self.scaling_factors) + 1.0) * x

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """Merge the active adapter weights into the base weights"""
        # No-op since we don't have base weights to merge into
        pass

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from the base weights"""
        # No-op since we don't have base weights to unmerge from
        pass

if __name__ == '__main__':
    # Test basic functionality
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load an SAE
    sae_release = 'gemma-scope-2b-pt-res-canonical'
    sae_id = 'layer_20/width_16k/canonical'
    sae_20 = SAE.from_pretrained(sae_release, sae_id, device=str(device))[0]
    
    input_dim = sae_20.cfg.d_sae
    batch_size = 3
    
    # Create test input
    x = torch.randn(batch_size, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Test IA3 layer with SAE
    ia3_layer = IsaerftIA3(sae=sae_20)
    output_ia3 = ia3_layer(x)
    print(f"\nIA3 case:")
    print(f"Output shape: {output_ia3.shape}")
    print(f"Initial output equals input (since scaling starts at 1):", torch.allclose(output_ia3, x))
    
    # Test that scaling works as expected
    ia3_layer.scaling_factors.data.fill_(2.0)  # Set all scaling factors to 2
    output_scaled = ia3_layer(x)
    print(f"After scaling by 2, output equals 3*input:", torch.allclose(output_scaled, 3*x))
    
    # Test adapter disabling
    ia3_layer._disable_adapters = True
    output_disabled = ia3_layer(x)
    print(f"After disabling adapters, output equals input:", torch.allclose(output_disabled, x))
    
    # Test feature description if NEURONPEDIA_API_KEY is set
    if os.environ.get("NEURONPEDIA_API_KEY"):
        print("\nTesting Neuronpedia feature descriptions:")
        desc = ia3_layer.get_feature_description((0,))  # Try to get description for feature 0
        print(f"Feature 0 description: {desc}")

