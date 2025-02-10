import torch
import safetensors.torch
import random

def mutate_shape(tensor):
    original_shape = list(tensor.shape)
    if not original_shape:  # Handle scalar tensor
        return tensor

    num_dims = len(original_shape)
    dim_to_mutate = random.randint(0, num_dims - 1)

    if random.random() < 0.5:  # More aggressive changes
        new_size = random.randint(1, original_shape[dim_to_mutate] * 3)
    else:
        new_size = random.randint(1, max(1, original_shape[dim_to_mutate] // 3))

    new_shape = original_shape[:]  # Create a copy
    new_shape[dim_to_mutate] = new_size
    try:
        return torch.randn(new_shape).to(tensor.device).type(tensor.dtype)
    except Exception as shape_e:
        print(f"Shape corruption caused error: {shape_e}")
        return tensor  # Return original tensor if new shape is invalid


def fuzz_safetensor_data(model_path, num_iterations=1000):
    try:
        original_state_dict = safetensors.torch.load_file(model_path)
    except Exception as e:
        print(f"Error loading original model: {e}")
        return

    for i in range(num_iterations):
        print(f"Fuzzing iteration {i+1}")
        try:
            fuzzed_state_dict = {}
            for key, tensor in original_state_dict.items():
                fuzzed_tensor = tensor.clone()

                # --- Mutations ---
                if random.random() < 0.2:  # Shape mutation
                    fuzzed_tensor = mutate_shape(tensor)

                if random.random() < 0.1:  # Data type change
                    new_dtype = torch.float16 if tensor.dtype == torch.float32 else torch.float32
                    fuzzed_tensor = fuzzed_tensor.to(new_dtype)

                if random.random() < 0.1:  # NaN/Inf injection
                    fuzzed_tensor = torch.nan_to_num(fuzzed_tensor * random.uniform(-1, 1))

                fuzzed_state_dict[key] = fuzzed_tensor

            # --- Model Loading and Usage Check ---
            try:
                safetensors.torch.save_file(fuzzed_state_dict, "fuzzed_model.safetensors") #Save the fuzzed model
                loaded_model = safetensors.torch.load_file("fuzzed_model.safetensors")
                loaded_model.eval()

                # --- Dummy Input (REPLACE THIS WITH YOUR ACTUAL INPUT) ---
                dummy_input = torch.randn(1, 514).long()  # Example - Adapt to LayoutLM!

                try:
                    output = loaded_model(dummy_input)
                    print("Forward pass successful. Checking outputs...")

                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print("NaN/Inf values in output!")
                        print(f"Failing tensor: {key}, Original shape: {tensor.shape}, Fuzzed shape: {fuzzed_tensor.shape}, Original dtype: {tensor.dtype}, Fuzzed dtype: {fuzzed_tensor.dtype}")
                        safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

                    # --- Add more output checks as needed ---
                    # Example: Check output shape (replace with your expected shape)
                    # expected_shape = torch.Size([1, ...]) # Replace ...
                    # if output.shape != expected_shape:
                    #     print("Output shape mismatch!")
                    #     print(f"Failing tensor: {key}, Original shape: {tensor.shape}, Fuzzed shape: {fuzzed_tensor.shape}, Original dtype: {tensor.dtype}, Fuzzed dtype: {fuzzed_tensor.dtype}")
                    #     safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

                except Exception as forward_e:
                    print(f"Forward pass failed: {forward_e}")
                    print(f"Failing tensor: {key}, Original shape: {tensor.shape}, Fuzzed shape: {fuzzed_tensor.shape}, Original dtype: {tensor.dtype}, Fuzzed dtype: {fuzzed_tensor.dtype}")
                    safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

            except Exception as load_e:
                print(f"Fuzzing caused load error: {load_e}")
                print(f"Failing tensor: {key}, Original shape: {tensor.shape}, Fuzzed shape: {fuzzed_tensor.shape}, Original dtype: {tensor.dtype}, Fuzzed dtype: {fuzzed_tensor.dtype}")
                safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

        except Exception as e:
            print(f"Fuzzing iteration {i+1} failed: {e}")


# Example usage (REPLACE with your model path):
model_path = "model.safetensors"
fuzz_safetensor_data(model_path, num_iterations=50)  # Or more iterations
