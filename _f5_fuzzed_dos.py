import torch
import safetensors.torch
import os
import time
import psutil
import random
from transformers import LayoutLMForQuestionAnswering, AutoTokenizer

def mutate_shape(tensor):
    original_shape = list(tensor.shape)
    if not original_shape:
        return tensor

    num_dims = len(original_shape)
    dim_to_mutate = random.randint(0, num_dims - 1)

    if random.random() < 0.5:
        new_size = random.randint(1, original_shape[dim_to_mutate] * 3)
    else:
        new_size = random.randint(1, max(1, original_shape[dim_to_mutate] // 3))

    new_shape = original_shape[:]
    new_shape[dim_to_mutate] = new_size
    try:
        return torch.randn(new_shape).to(tensor.device).type(tensor.dtype)
    except Exception as shape_e:
        print(f"Shape corruption caused error: {shape_e}")
        return tensor

def load_and_check(model_path, tokenizer):
    try:
        state_dict = safetensors.torch.load_file(model_path)
        model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")
        model.eval()
        model.load_state_dict(state_dict, strict=False)

        # --- Get Valid Input (REPLACE WITH YOUR ACTUAL INPUT LOGIC) ---
        text = "What is the capital of France?"  # Example, replace with your input
        inputs = tokenizer(text, return_tensors="pt") #Pass tokenizer to get inputs

        try:
            with torch.no_grad():
                output = model(**inputs)
                # --- Check Output (Add your specific checks here) ---
                # print(output) #Examine the output
                # assert output.logits.shape == torch.Size([1, ...]) #Replace ... with expected output shape
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

def fuzz_safetensor_data(model_path, num_iterations=1000, tokenizer=None):
    try:
        original_state_dict = safetensors.torch.load_file(model_path)
    except Exception as e:
        print(f"Error loading original model: {e}")
        return

    model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")
    model.eval()

    for i in range(num_iterations):
        print(f"Fuzzing iteration {i+1}")
        try:
            fuzzed_state_dict = {}
            for key, tensor in original_state_dict.items():
                fuzzed_tensor = tensor.clone()

                # --- Mutations ---
                if random.random() < 0.2:
                    fuzzed_tensor = mutate_shape(tensor)

                if random.random() < 0.1:
                    new_dtype = torch.float16 if tensor.dtype == torch.float32 else torch.float32
                    fuzzed_tensor = fuzzed_tensor.to(new_dtype)

                if random.random() < 0.1:
                    fuzzed_tensor = torch.nan_to_num(fuzzed_tensor * random.uniform(-1, 1))

                fuzzed_state_dict[key] = fuzzed_tensor

            # --- Model Loading and Usage Check ---
            try:
                model.load_state_dict(fuzzed_state_dict, strict=False)
                print("Fuzzing successful, but not necessarily an exploit.")
            except Exception as model_load_e:
                print(f"Error loading fuzzed model: {model_load_e}")
                for key, tensor in fuzzed_state_dict.items():
                  print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
                safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")
                continue

            try:
                # --- Dummy Input (REPLACE THIS WITH YOUR ACTUAL INPUT) ---
                text = "What is your tensor size"  # Example, replace with your input
                inputs = tokenizer(text, return_tensors="pt") #Pass tokenizer to get inputs
                try:
                  with torch.no_grad():
                      output = model(**inputs)
                      print("Forward pass successful. Checking outputs...")

                      if torch.isnan(output).any() or torch.isinf(output).any():
                          print("NaN/Inf values in output!")
                          for key, tensor in fuzzed_state_dict.items():
                            print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
                          safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

                      # --- Add more output checks as needed ---
                      # Example: Check output shape (replace with your expected shape)
                      # expected_shape = torch.Size([1, ...]) # Replace ...
                      # if output.shape != expected_shape:
                      #     print("Output shape mismatch!")
                      #     for key, tensor in fuzzed_state_dict.items():
                      #       print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
                      #     safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

                except Exception as forward_e:
                    print(f"Forward pass failed: {forward_e}")
                    for key, tensor in fuzzed_state_dict.items():
                      print(f"Failing tensor: {key}, Original shape: {original_state_dict[key].shape if key in original_state_dict else 'N/A'}, Fuzzed shape: {tensor.shape}, Original dtype: {original_state_dict[key].dtype if key in original_state_dict else 'N/A'}, Fuzzed dtype: {tensor.dtype}")
                    safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

            except Exception as e:
                print(f"Fuzzing iteration {i+1} failed: {e}")

        except Exception as e:
            print(f"Fuzzing iteration {i+1} failed: {e}")

if __name__ == "__main__":
    model_path = "model.safetensors" #Path to original model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased") #Initialize tokenizer
    fuzzed_dir = "/home/stellaris/aisecurity/safetensor_fuzzer/fuzzed_tensors" # Replace with your fuzzed files directory
    num_files = 0

    if not os.path.exists(fuzzed_dir):
      print(f"Error: Directory '{fuzzed_dir}' not found.")
      exit()

    for filename in os.listdir(fuzzed_dir):
        if filename.endswith(".safetensors"):
            num_files +=
