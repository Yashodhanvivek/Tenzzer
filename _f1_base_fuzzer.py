import torch
import safetensors.torch
import random

def fuzz_safetensor_data(model_path, num_iterations=1000):
    """
    A simplified conceptual fuzzer for safetensor files.
    This is for demonstration and educational purposes only.
    """
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
                # Fuzz tensor data (simplified example)
                fuzzed_tensor = tensor.clone()  # Important: Create a copy
                # Example fuzzing strategies (choose or combine):
                if random.random() < 0.1: # 10% chance
                    # Introduce shape mismatches (intentionally corrupt shapes)
                    new_shape = list(tensor.shape)
                    if len(new_shape) > 0:
                      new_shape[0] = random.randint(1, 100)  # Corrupt first dimension
                    try:
                        fuzzed_tensor = torch.randn(new_shape).to(tensor.device).type(tensor.dtype)
                    except Exception as shape_e:
                        print(f"Shape corruption caused error: {shape_e}")
                        continue #Skip to next tensor

                if random.random() < 0.1:
                    # Change data types (e.g., float32 to float16)
                    new_dtype = torch.float16 if tensor.dtype == torch.float32 else torch.float32
                    fuzzed_tensor = fuzzed_tensor.to(new_dtype)

                if random.random() < 0.1:
                    # Introduce NaN or Inf values (carefully)
                    fuzzed_tensor = torch.nan_to_num(fuzzed_tensor * random.uniform(-1, 1)) #Try to introduce NaN or Inf

                fuzzed_state_dict[key] = fuzzed_tensor

            # Try loading the fuzzed state dict.
            # IMPORTANT: Don't actually *use* this fuzzed model in a critical system.
            # This code is only for *finding* potential issues.
            try:
              safetensors.torch.save_file(fuzzed_state_dict, "fuzzed_model.safetensors") #Save the fuzzed model
              loaded_model = safetensors.torch.load_file("fuzzed_model.safetensors") #Load it back
              print("Fuzzing successful, but not necessarily an exploit.")
            except Exception as load_e:
                print(f"Fuzzing caused load error: {load_e}")
                # Log the details of the fuzzed data that caused the error.
                # This is how you would identify potential vulnerabilities.
                print(f"Failing tensor: {key}, Original shape: {tensor.shape}, Fuzzed shape: {fuzzed_tensor.shape}, Original dtype: {tensor.dtype}, Fuzzed dtype: {fuzzed_tensor.dtype}")
                # Optionally save the failing case for later analysis
                # safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")

        except Exception as e:
            print(f"Fuzzing iteration {i+1} failed: {e}")


# Example usage (replace with your model path):
model_path = "model.safetensors"
fuzz_safetensor_data(model_path, num_iterations=50) #Reduce iterations for testing
import random

def mutate_shape(tensor):
    original_shape = list(tensor.shape)
    if not original_shape: #Handle scalar tensor
        return tensor
    
    num_dims = len(original_shape)
    dim_to_mutate = random.randint(0, num_dims - 1)

    #More aggressive mutations
    if random.random() < 0.5: #50% chance
      new_size = random.randint(1, original_shape[dim_to_mutate] * 3) #Larger changes
    else:
      new_size = random.randint(1, max(1,original_shape[dim_to_mutate] // 3)) #Smaller changes but can be 0

    new_shape = original_shape[:] #Create a copy
    new_shape[dim_to_mutate] = new_size
    try:
        return torch.randn(new_shape).to(tensor.device).type(tensor.dtype)
    except Exception as shape_e:
        print(f"Shape corruption caused error: {shape_e}")
        return tensor #Return original tensor if new shape is invalid

#In the fuzzing loop
fuzzed_tensor = mutate_shape(tensor)

