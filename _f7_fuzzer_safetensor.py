import torch
import safetensors.torch
import os
import time
import psutil
import random
from transformers import LayoutLMForQuestionAnswering, AutoTokenizer

def mutate_tensor(tensor, mutation_rate=0.1):
    """Mutates a tensor with various strategies."""
    fuzzed_tensor = tensor.clone()
    original_shape = list(tensor.shape)
    original_dtype = tensor.dtype

    if random.random() < mutation_rate:  # Shape mutation
        if original_shape:  # Check if shape is not empty
            dim_to_mutate = random.randint(0, len(original_shape) - 1)
            new_size = max(1, original_shape[dim_to_mutate] + random.randint(-2, 2))  # More controlled shape change
            new_shape = original_shape[:]
            new_shape[dim_to_mutate] = new_size
            try:
                fuzzed_tensor = torch.randn(new_shape, dtype=original_dtype, device=tensor.device)
            except Exception as shape_e:
                print(f"Shape mutation error: {shape_e}")
                return tensor  # Return original tensor if shape mutation fails

    if random.random() < mutation_rate:  # Dtype mutation
        new_dtype = torch.float16 if original_dtype == torch.float32 else torch.float32  # Toggle between float32 and float16
        try:
            fuzzed_tensor = fuzzed_tensor.to(new_dtype)
        except Exception as dtype_e:
            print(f"Dtype mutation error: {dtype_e}")
            return tensor

    if random.random() < mutation_rate:  # Value mutation (NaN/Inf injection)
        fuzzed_tensor = torch.nan_to_num(fuzzed_tensor * random.uniform(-1, 1))  # Clamp values to avoid NaN/Inf in initial tensor
        if random.random() < mutation_rate / 10:  # Lower rate of NaN/Inf injection
            fuzzed_tensor[random.randint(0, fuzzed_tensor.numel() - 1)] = float('nan')  # Inject NaN
        if random.random() < mutation_rate / 10:  # Lower rate of NaN/Inf injection
            fuzzed_tensor[random.randint(0, fuzzed_tensor.numel() - 1)] = float('inf')  # Inject Inf

    return fuzzed_tensor

def load_and_check(model_path, tokenizer):
    try:
        state_dict = safetensors.torch.load_file(model_path)
        model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")
        model.eval()
        model.load_state_dict(state_dict, strict=False)

        # --- Get Valid Input (REPLACE WITH YOUR ACTUAL INPUT LOGIC) ---
        text = "What is the capital of France?"  # Example, replace with your input
        inputs = tokenizer(text, return_tensors="pt")

        try:
            with torch.no_grad():
                output = model(**inputs)
                # --- Check Output (Add your specific checks here) ---
                # print(output)  # Examine the output
                # assert output.logits.shape == torch.Size([1, ...])  # Replace ... with expected output shape
                print("Forward pass successful.")
        except Exception as forward_e:
            print(f"Forward pass failed: {forward_e}")
            return False

        del model
        return True

    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return False

def monitor_resources(pid):
    process = psutil.Process(pid)
    cpu_percent = process.cpu_percent(interval=0.1)
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / (1024 * 1024)
    return cpu_percent, mem_usage_mb

def fuzz_safetensor_data(model_path, num_iterations=1000, tokenizer=None, output_dir="fuzzed_tensors"):
    try:
        original_state_dict = safetensors.torch.load_file(model_path)
    except Exception as e:
        print(f"Error loading original model: {e}")
        return

    model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_iterations):
        print(f"Fuzzing iteration {i+1}")
        fuzzed_state_dict = {}
        for key, tensor in original_state_dict.items():
            fuzzed_state_dict[key] = mutate_tensor(tensor)

        try:
            model.load_state_dict(fuzzed_state_dict, strict=False)
            print("Fuzzing successful, but not necessarily an exploit.")
        except Exception as model_load_e:
            print(f"Model load error: {model_load_e}")
            for key, tensor in fuzzed_state_dict.items():
                print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
            safetensors.torch.save_file(fuzzed_state_dict, os.path.join(output_dir, f"fuzzed_model_fail_{i}_load.safetensors"))
            continue  # Skip the forward pass if model load fails

        try:
            text = "What is the capital of France?"  # Example, replace with your input
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs)

                if torch.isnan(output.logits).any() or torch.isinf(output.logits).any():  # Check logits for NaN/Inf
                    print("NaN/Inf values in output logits!")
                    for key, tensor in fuzzed_state_dict.items():
                        print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
                    safetensors.torch.save_file(fuzzed_state_dict, os.path.join(output_dir, f"fuzzed_model_fail_{i}_nan_inf.safetensors"))
                    continue  # Skip other output checks

                # Add your own output checks here (e.g., shape, specific value ranges)
                # Example:
                # if output.logits.shape != torch.Size([1, 5]):  # Replace with your expected shape
                #     print("Output shape mismatch!")
                #     safetensors.torch.save_file(fuzzed_state_dict, os.path.join(output_dir, f"fuzzed_model_fail_{i}_shape.safetensors"))
                #     continue  # Skip other output checks


        except Exception as forward_e:
            print(f"Forward pass error: {forward_e}")
            for key, tensor in fuzzed_state_dict.items():
                print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
            safetensors.torch.save_file(fuzzed_state_dict, os.path.join(output_dir, f"fuzzed_model_fail_{i}_forward.safetensors"))
            continue  # Skip other output checks


if __name__ == "__main__":
    model_path = "model.safetensors"  # Path to original model (REPLACE THIS)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")  # Initialize tokenizer
    fuzzed_dir = "fuzzed_tensors"  #
