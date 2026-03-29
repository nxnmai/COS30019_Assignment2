import os
import sys
import numpy as np

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.base_model import BaseTimeSeriesModel

def run_tc09(
    model: BaseTimeSeriesModel,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 10,
    seed: int = 42  # Added default seed for reproducibility
) -> dict:
    """
    TC09: Random Sample Consistency Test (Fixed Seed).

    Evaluates whether the model produces consistent predictions across 
    randomly selected individual samples. 
    Using a fixed seed ensures consistent results across different runs.
    """
    print(f"\n>>> Running TC09: Random Sample Consistency Test ({num_samples} samples)...")
    print(f"[TC09] Using Random Seed: {seed}")

    # Set the seed for NumPy to ensure 'np.random.choice' picks the same indices every time
    np.random.seed(seed)

    # Guard: cannot sample more than available data
    if num_samples > len(x_test):
        print(f"[TC09] WARNING: num_samples ({num_samples}) > dataset size ({len(x_test)}).")
        num_samples = len(x_test)

    # Select unique random indices (will be the same indices every run thanks to the seed)
    indices = np.random.choice(len(x_test), size=num_samples, replace=False)

    maes = []
    for idx in indices:
        x_sample = x_test[idx : idx + 1]
        y_sample = y_test[idx : idx + 1]

        pred = model.predict(x_sample)
        mae  = float(np.mean(np.abs(pred.flatten() - y_sample.flatten())))
        maes.append(mae)

    maes = np.array(maes)

    # Statistical Calculations
    mae_variance = float(np.var(maes))
    mae_std      = float(np.std(maes))
    mae_mean     = float(np.mean(maes))
    mae_min      = float(np.min(maes))
    mae_max      = float(np.max(maes))

    # Threshold Logic
    VARIANCE_THRESHOLD = (mae_mean * 1.0) ** 2 
    passed = mae_variance < VARIANCE_THRESHOLD

    print(f"[TC09] Samples evaluated : {num_samples}")
    print(f"[TC09] MAE mean          : {mae_mean:.6f}")
    print(f"[TC09] MAE variance      : {mae_variance:.8f}")
    print(f"[TC09] Result            : {'PASSED' if passed else 'FAILED'}")

    return {
        "mae_variance": mae_variance,
        "mae_std": mae_std,
        "mae_mean": mae_mean,
        "passed": passed,
        "selected_indices": indices.tolist() # Useful for report documentation
    }

if __name__ == "__main__":
    print("TC09 script loaded.")