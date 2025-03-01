import json
import torch
from typing import Dict, List, Any, Optional, Union
from ISaeRFT_Interpreter import ISaeRFT_Interpreter
from sae_lens import SAE
import argparse
import os
import sys

def interpret_trainable_params(
    input_data: Union[str, Dict[str, Any], torch.Tensor], 
    sae: SAE, 
    interpretation_type: str = 'L2', 
    top_k:int = 20,
    bottom_k:int=20,
    neuronpedia_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Interprets trainable parameters from a JSON object or PyTorch tensor using ISaeRFT_Interpreter.
    
    Args:
        input_data: Either a JSON string, dictionary with 'values' key, or a PyTorch tensor
        sae: SAE object to use for interpretation
        interpretation_type: Type of interpretation to use ('L2' or 'absolute')
        top_k: Number of top features to return
        bottom_k: Number of bottom features to return
        neuronpedia_api_key: API key for Neuronpedia (optional)
        
    Returns:
        List of dictionaries containing interpretation results
    """
    # Handle different input types
    if isinstance(input_data, torch.Tensor):
        tensor_values = input_data
    elif isinstance(input_data, str):
        # Check if it's a JSON string
        try:
            data = json.loads(input_data)
            if 'values' not in data:
                raise ValueError("JSON data must contain a 'values' key")
            tensor_values = torch.tensor(data['values'], dtype=torch.float32)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided")
    elif isinstance(input_data, dict):
        # It's a dictionary
        if 'values' not in input_data:
            raise ValueError("JSON data must contain a 'values' key")
        tensor_values = torch.tensor(input_data['values'], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
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
    parser = argparse.ArgumentParser(description="Interpret trainable parameters from a JSON file, JSON string, or PyTorch file")
    parser.add_argument("input_path", help="Path to JSON file, PyTorch (.pt) file, or JSON string containing 'values' key")
    parser.add_argument("--release", default="gemma-scope-2b-pt-res-canonical", help="SAE release name")
    parser.add_argument("--sae_id", default="layer_20/width_16k/canonical", help="SAE ID")
    parser.add_argument("--interpretation_type", default="identity", help="Type of interpretation")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to return")
    parser.add_argument("--bottom_k", type=int, default=20, help="Number of bottom features to return")
    parser.add_argument("--neuronpedia_api_key", help="API key for Neuronpedia (optional)")
    
    args = parser.parse_args()
    
    # Check if input is a file path or a JSON string
    if os.path.isfile(args.input_path):
        file_extension = os.path.splitext(args.input_path)[1].lower()
        if file_extension == '.pt':
            # Load PyTorch tensor
            try:
                loaded_data = torch.load(args.input_path, map_location='cpu')
                # Check what's in the loaded data and extract the tensor
                if isinstance(loaded_data, dict) and 'values' in loaded_data:
                    input_data = loaded_data['values']
                elif isinstance(loaded_data, dict):
                    # Try to find a tensor in the dictionary
                    for key, value in loaded_data.items():
                        if isinstance(value, torch.Tensor):
                            input_data = value
                            break
                    else:
                        # If no tensor found, use the first item
                        input_data = next(iter(loaded_data.values()))
                else:
                    # Assume it's directly a tensor or can be used as is
                    input_data = loaded_data
                
                # Print debug info about the loaded data
                print(f"Loaded PyTorch data type: {type(input_data)}")
                if isinstance(input_data, torch.Tensor):
                    print(f"Tensor shape: {input_data.shape}")
            except Exception as e:
                print(f"Error loading PyTorch file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Assume it's a JSON file
            with open(args.input_path, 'r') as f:
                input_data = f.read()
    else:
        # Assume it's a JSON string
        input_data = args.input_path
    
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
            input_data=input_data, 
            sae=mysae,
            interpretation_type=args.interpretation_type,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
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
