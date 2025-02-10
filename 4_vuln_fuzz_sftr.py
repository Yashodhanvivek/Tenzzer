import torch
from safetensors.torch import load_file
import numpy as np
import logging

def detect_vulnerabilities(model_path: str, test_input: torch.Tensor):
    """Loads a SafeTensor model and runs a test input to detect vulnerabilities."""
    logging.basicConfig(filename="org_fuzz_vuln_log.txt", level=logging.INFO)
    
    try:
        tensors = load_file(model_path)
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
        
        for name, tensor in tensors.items():
            try:
                if tensor.dim() < 2:
                    logging.warning(f"Tensor {name} has low dimensions {tensor.shape}, possible corruption.")
                    continue
                
                expected_dim = tensor.shape[1] if tensor.dim() > 1 else tensor.shape[0]
                if expected_dim != test_input.shape[0]:
                    logging.warning(f"Shape mismatch in tensor: {name} ({tensor.shape} vs {test_input.shape})")
                    continue
                
                output = tensor @ test_input  # Simulated inference step
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logging.error(f"Detected NaN/Inf in tensor: {name}, potential vulnerability.")
                
                grad_output = torch.autograd.grad(outputs=output, inputs=tensor, grad_outputs=torch.ones_like(output), allow_unused=True, retain_graph=True)
                if grad_output is None or any(g is None for g in grad_output):
                    logging.critical(f"Gradient anomaly detected in tensor: {name}, potential adversarial attack vector.")
                
                if (tensor.abs() > 1e20).any():
                    logging.critical(f"Extreme value detected in tensor: {name}, potential overflow vulnerability.")
                
            except RuntimeError as e:
                logging.error(f"Runtime error in tensor {name}: {e}")
            except Exception as e:
                logging.critical(f"Unexpected exception in tensor {name}: {e}")
        
        print("Fuzzing complete. Check fuzz_vuln_log.txt for critical vulnerabilities.")
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")

if __name__ == "__main__":
    #fuzzed_model_path = "fuzzed_model.safetensors"
    fuzzed_model_path = "model.safetensors"  # Ensure this exists
    test_input = torch.randn(768, 1)  # Adjusted input size to match expected model dimensions
    detect_vulnerabilities(fuzzed_model_path, test_input)

