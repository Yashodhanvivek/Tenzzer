import torch
import safetensors.torch
import os
import time
import psutil
import random
from transformers import LayoutLMForQuestionAnswering, AutoTokenizer

# ... (mutate_shape, load_and_check, monitor_resources, fuzz_safetensor_data functions - same as before)

if __name__ == "__main__":
    model_path = "model.safetensors"  # Path to original model (REPLACE THIS)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")  # Initialize tokenizer
    fuzzed_dir = "/home/stellaris/aisecurity/safetensor_fuzzer/fuzzed_tensors/"  # Replace with your fuzzed files directory (REPLACE THIS)

    if not os.path.exists(fuzzed_dir):
        print(f"Error: Directory '{fuzzed_dir}' not found.")
        exit()

    fuzzed_files = [f for f in os.listdir(fuzzed_dir) if f.endswith(".safetensors")]
    num_files = len(fuzzed_files)

    if num_files == 0:
        print(f"Error: No .safetensors files found in '{fuzzed_dir}'.")
        exit()

    print(f"Found {num_files} fuzzed files. Starting DoS and Fuzzing test...")

    # --- DoS Test ---
    start_time_dos = time.time()
    process_id = os.getpid()

    try:
        for filename in fuzzed_files:  # Iterate through the list of files
            file_path = os.path.join(fuzzed_dir, filename)
            load_successful = load_and_check(file_path, tokenizer)
            if load_successful:
                cpu, mem = monitor_resources(process_id)
                print(f"CPU: {cpu:.1f}%, Memory: {mem:.1f} MB")
            else:
                print("Skipping resource monitoring due to load/forward pass failure.")

    except KeyboardInterrupt:
        print("DoS test interrupted by user.")

    end_time_dos = time.time()
    print(f"Time taken for DoS test: {end_time_dos - start_time_dos} seconds")

    # Summary of Resources (DoS test)
    try:
        process = psutil.Process(process_id)
        cpu_usage = process.cpu_times()
        print(f"Total CPU Time (user, DoS): {cpu_usage.user:.2f} seconds")
        print(f"Total CPU Time (system, DoS): {cpu_usage.system:.2f} seconds")
        mem_info = process.memory_full_info()
        peak_mem_mb = mem_info.peak_rss / (1024 * 1024)
        print(f"Peak Memory Usage (DoS): {peak_mem_mb:.2f} MB")
    except psutil.NoSuchProcess:
        print("Process ended before resource statistics could be collected.")

    # --- Fuzzing ---
    start_time_fuzz = time.time()
    fuzz_safetensor_data(model_path, num_iterations=50, tokenizer=tokenizer)  # Call the fuzzing function
    end_time_fuzz = time.time()
    print(f"Time taken for Fuzzing: {end_time_fuzz - start_time_fuzz} seconds")
