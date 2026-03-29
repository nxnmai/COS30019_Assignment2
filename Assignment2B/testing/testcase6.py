import os
import sys
import time
import torch
import numpy as np

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.base_model import BaseTimeSeriesModel

def run_tc06(model: BaseTimeSeriesModel, x_test: np.ndarray):
    """
    TC06: Hardware Efficiency - Inference Latency Test.
    Measures how fast the model predicts a single sample.
    """
    print("\n>>> Running TC06: Inference Latency Test...")
    
    # Detect hardware (utilizing your Dell XPS 15 GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TC06] Computing Device: {device}")
    
    # Warm-up run to initialize GPU kernels
    _ = model.predict(x_test[:1], device=device)
    
    # Start measuring time
    start_time = time.perf_counter()
    
    # Predict a single sample
    _ = model.predict(x_test[:1], device=device)
    
    end_time = time.perf_counter()
    
    # Calculate latency in milliseconds
    latency_ms = (end_time - start_time) * 1000
    
    print(f"[TC06] Result: Inference Latency = {latency_ms:.2f} ms")
    return latency_ms

if __name__ == "__main__":
    print("TC06 script loaded.")