import torch
from safetensors.torch import load_file
import numpy as np
import logging

def detect_vulnerabilities(model_path: str, test_input: torch.Tensor):
    """Loads a SafeTensor model and runs a test input to detect vulnerabilities."""
    logging.basicConfig(filename="fuzz_vuln_log.txt", level=logging.INFO)
    
    try:
        tensors = load_file(model_path)
        
        for name, tensor in tensors.items():
            try:
                output = tensor @ test_input  # Simulated inference step
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logging.warning(f"Detected NaN/Inf in tensor: {name}")
            except RuntimeError as e:
                logging.error(f"Runtime error in tensor {name}: {e}")
            except Exception as e:
                logging.critical(f"Unexpected exception in tensor {name}: {e}")
        
        print("Fuzzing complete. Check fuzz_vuln_log.txt for issues.")
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")

if __name__ == "__main__":
    fuzzed_model_path = "fuzzed_model.safetensors"  # Ensure this exists
    test_input = torch.randn(10, 10)  # Example test input
    detect_vulnerabilities(fuzzed_model_path, test_input)

