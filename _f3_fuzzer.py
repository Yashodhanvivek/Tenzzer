import torch
import safetensors.torch
import random
from transformers import LayoutLMForQuestionAnswering  # Or your LayoutLM model class

# ... (mutate_shape function remains the same) ...

def fuzz_safetensor_data(model_path, num_iterations=1000):
    # 1. Create the model instance *first*
    model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased") #Initialize the model
    model.eval() #Set to eval mode
    for i in range(num_iterations):
      print(f"Fuzzing iteration {i+1}")
      try:
          original_state_dict = safetensors.torch.load_file(model_path)

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

          # --- Load the *fuzzed* state dict into the model ---
          try:
            model.load_state_dict(fuzzed_state_dict, strict=False) #Load into the model
            print("Fuzzing successful, but not necessarily an exploit.")
          except Exception as model_load_e:
            print(f"Error loading fuzzed model: {model_load_e}")
            print(f"Failing tensor: {key}, Original shape: {tensor.shape}, Fuzzed shape: {fuzzed_tensor.shape}, Original dtype: {tensor.dtype}, Fuzzed dtype: {fuzzed_tensor.dtype}")
            safetensors.torch.save_file(fuzzed_state_dict, f"fuzzed_model_fail_{i}.safetensors")
            continue #Skip to next iteration

          # --- Model Usage Check ---
          try:
              # --- Dummy Input (REPLACE THIS WITH YOUR ACTUAL INPUT) ---
              dummy_input = torch.randn(1, 514).long()  # Example - Adapt to LayoutLM!
              try:
                output = model(dummy_input)
                print("Forward pass successful. Checking outputs...")
                # ... (rest of the output checks)
              except Exception as forward_e:
                  print(f"Forward pass failed: {forward_e}")
                  # ... (logging and saving failing cases)
          except Exception as e:
              print(f"Fuzzing iteration {i+1} failed: {e}")
      except Exception as e:
          print(f"Fuzzing iteration {i+1} failed: {e}")


# Example usage (REPLACE with your model path):
model_path = "model.safetensors"
fuzz_safetensor_data(model_path, num_iterations=50)  # Or more iterations
