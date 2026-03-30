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
    seed: int = 42
) -> dict:
    """
    TC09: Random Sample Consistency Test (Fixed Seed).
    """
    print(f"\n>>> Running TC09: Random Sample Consistency Test ({num_samples} samples)...")
    print(f"[TC09] Using Random Seed: {seed}")

    np.random.seed(seed)

    if num_samples > len(x_test):
        num_samples = len(x_test)

    indices = np.random.choice(len(x_test), size=num_samples, replace=False)

    maes = []
    for idx in indices:
        x_sample = x_test[idx : idx + 1]
        y_sample = y_test[idx : idx + 1]

        pred = model.predict(x_sample)
        mae  = float(np.mean(np.abs(pred.flatten() - y_sample.flatten())))
        maes.append(mae)

    maes = np.array(maes)
    mae_variance = float(np.var(maes))
    mae_mean     = float(np.mean(maes))

    # Variance threshold check (should be stable for consistent models)
    VARIANCE_THRESHOLD = (max(0.1, mae_mean) * 0.5) ** 2 
    passed = mae_variance < VARIANCE_THRESHOLD

    print(f"[TC09] Samples evaluated : {num_samples}")
    print(f"[TC09] MAE mean          : {mae_mean:.6f}")
    print(f"[TC09] MAE variance      : {mae_variance:.8f}")
    
    status = "PASSED" if passed else "FAILED"
    print(f"[TC09] Result: {status} (MAE variance={mae_variance:.8f})")

    return {
        "mae_variance": mae_variance,
        "mae_mean": mae_mean,
        "passed": passed
    }

if __name__ == "__main__":
    from testing.test_utils import get_test_args, load_test_context
    args = get_test_args()
    try:
        ctx = load_test_context(args.model)
        run_tc09(ctx["model"], ctx["x_test"], ctx["y_test"])
    except Exception as e:
        print(f"[TC09] Result: FAILED (Execution Error: {e})")
        sys.exit(1)