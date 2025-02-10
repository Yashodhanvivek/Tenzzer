import os
import random
import numpy as np
from safetensors.torch import load_file, save_file
import torch

def mutate_tensor(tensor: torch.Tensor):
    """Applies random mutations to the tensor while handling different data types."""
    tensor = tensor.clone()
    
    mutation_type = random.choice(["noise", "extreme"])  # Removed bitflip to avoid float issues
    if mutation_type == "noise":
        if tensor.is_floating_point():
            tensor += torch.randn_like(tensor) * 0.1
        else:
            tensor += torch.randint_like(tensor, -10, 10)
    elif mutation_type == "extreme":
        index = tuple(random.randint(0, s - 1) for s in tensor.shape)
        if tensor.is_floating_point():
            tensor[index] = random.choice([-1e10, 1e10])
        else:
            tensor[index] = random.choice([-2**31, 2**31 - 1])
    return tensor

def fuzz_safetensor(file_path: str, output_path: str):
    """Loads, mutates, and saves a fuzzed SafeTensor file."""
    tensors = load_file(file_path)
    fuzzed_tensors = {k: mutate_tensor(v) for k, v in tensors.items()}
    save_file(fuzzed_tensors, output_path)
    print(f"Fuzzed SafeTensor saved to: {output_path}")

if __name__ == "__main__":
    input_path = "model.safetensors"  # Change to your model path
    output_path = "fuzzed_model.safetensors"
    fuzz_safetensor(input_path, output_path)

