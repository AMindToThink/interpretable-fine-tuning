import requests
from typing import Optional, Dict, Any

class NeuronpediaClient:
    """Client for fetching feature descriptions from Neuronpedia API."""
    
    def __init__(self, api_key: str, base_url: str = "https://www.neuronpedia.org"):
        """
        Initialize the client.
        
        Args:
            api_key: Your Neuronpedia API key
            base_url: Base URL for the API (defaults to https://www.neuronpedia.org)
        """
        self.base_url = base_url    
        self.headers = {"X-Api-Key": api_key}
    
    def get_feature(self, index: int, neuronpedia_id:str='', model_id: str='', layer: str='') -> Dict[str, Any]:
        """
        Fetch feature data from Neuronpedia.
        
        Args:
            neuronpedia_id: combines the model_id and the layer
            model_id: The model ID (e.g., 'gpt2-small')
            layer: The layer/SAE ID (e.g., '0-res-jb')
            index: The feature index
            
        Returns:
            Dict containing the feature data
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
            KeyError: If the response is missing expected data
        """
        assert (not (bool(neuronpedia_id) and bool(model_id))) and (not (bool(neuronpedia_id) and bool(layer))), "neuronpedia_id replaces both model_id and layer"
        # example: https://www.neuronpedia.org/api/feature/gemma-2-2b/2-gemmascope-mlp-65k/19964 (expressions of happiness and joy)
        neuronpedia_id = f"{model_id}/{layer}" if not neuronpedia_id else neuronpedia_id
        url = f"{self.base_url}/api/feature/{neuronpedia_id}/{index}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_description(self, model_id: str, layer: str, index: int, 
                       preferred_model: str = "claude-3-5-sonnet-20240620") -> Optional[str]:
        """
        Get the auto-generated description for a feature.
        
        Args:
            model_id: The model ID (e.g., 'gpt2-small')
            layer: The layer/SAE ID (e.g., '0-res-jb')
            index: The feature index
            preferred_model: Preferred model for the explanation (defaults to Claude)
            
        Returns:
            The description string, or None if no description is found
            
        Example:
            >>> client = NeuronpediaClient("your-api-key")
            >>> desc = client.get_description("gpt2-small", "0-res-jb", 14057)
            >>> print(desc)
            "references to \"Jedi\" in the context of Star Wars, particularly \"Return of the Jedi\"."
        """
        try:
            feature = self.get_feature(model_id, layer, index)
            
            # First try to get description from preferred model
            for explanation in feature.get('explanations', []):
                if explanation.get('explanationModelName') == preferred_model:
                    return explanation.get('description')
            
            # If preferred model not found, return first available description
            if feature.get('explanations'):
                return feature['explanations'][0].get('description')
                
            return None
            
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"Error fetching description: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    import sys
    api_key = sys.argv[1] 
    client = NeuronpediaClient(api_key)
    description = client.get_description("gpt2-small", "0-res-jb", 14057)
    print(f"Feature description: {description}")