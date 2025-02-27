import json
import torch
from typing import Dict, List, Any, Optional, Union
from ISaeRFT_Interpreter import ISaeRFT_Interpreter
from sae_lens import SAE
import argparse
import os
import sys

def interpret_trainable_params(
    json_data: Union[str, Dict[str, Any]], 
    sae: SAE, 
    interpretation_type: str = 'L2', 
    top_k:int = 20,
    bottom_k:int=20,
    neuronpedia_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Interprets trainable parameters from a JSON object using ISaeRFT_Interpreter.
    
    Args:
        json_data: Either a JSON string or a dictionary containing a 'values' key with a list of numbers
        sae: SAE object to use for interpretation
        interpretation_type: Type of interpretation to use ('L2' or 'absolute')
        top_k: Number of top features to return
        neuronpedia_api_key: API key for Neuronpedia (optional)
        
    Returns:
        List of dictionaries containing interpretation results
    """
    # Parse JSON if string is provided
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided")
    else:
        data = json_data
    
    # Extract values
    if 'values' not in data:
        raise ValueError("JSON data must contain a 'values' key")
    
    values = data['values']
    
    # Convert to tensor
    tensor_values = torch.tensor(values, dtype=torch.float32)
    
    # Create interpreter and interpret
    interpreter = ISaeRFT_Interpreter(sae=sae, neuronpedia_api_key=neuronpedia_api_key)
    
    # Interpret the tensor
    interpretation_results = interpreter.interpret_bias(
        vector=tensor_values,
        interpretation_type=interpretation_type,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    
    return interpretation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret trainable parameters from a JSON file or string")
    parser.add_argument("json_input", help="JSON file path or JSON string containing 'values' key")
    parser.add_argument("--release", default="gemma-scope-2b-pt-res-canonical", help="SAE release name")
    parser.add_argument("--sae_id", default="layer_25/width_16k/canonical", help="SAE ID")
    parser.add_argument("--interpretation_type", default="identity", help="Type of interpretation")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to return")
    parser.add_argument("--bottom_k", type=int, default=20, help="Number of bottom features to return")
    parser.add_argument("--neuronpedia_api_key", help="API key for Neuronpedia (optional)")
    
    args = parser.parse_args()
    
    # Check if input is a file path or a JSON string
    if os.path.isfile(args.json_input):
        with open(args.json_input, 'r') as f:
            json_data = f.read()
    else:
        json_data = args.json_input
    
    # Load SAE model
    try:
        print(f"Loading SAE model: {args.release}, {args.sae_id}")
        mysae = SAE.from_pretrained(release=args.release, 
                                   sae_id=args.sae_id, 
                                   device='cpu')[0]
    except Exception as e:
        print(f"Error loading SAE model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Interpret parameters
    try:
        results = interpret_trainable_params(
            json_data=json_data, 
            sae=mysae,
            interpretation_type=args.interpretation_type,
            top_k=args.top_k,
            neuronpedia_api_key=args.neuronpedia_api_key
        )
        
        # Print results in a readable format
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error interpreting parameters: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
