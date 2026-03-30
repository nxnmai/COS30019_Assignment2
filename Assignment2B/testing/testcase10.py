import os
import sys
import numpy as np

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.base_model import BaseTimeSeriesModel

def run_tc10(
    model: BaseTimeSeriesModel,
    initial_window: np.ndarray,
    steps: int = 4,
    target_feature_index: int = 0,
    interval_minutes: int = 15,
) -> dict:
    """
    TC10: Recursive Multi-step Forecasting (Sliding Window).
    """
    horizon_minutes = steps * interval_minutes
    print(f"\n>>> Running TC10: Recursive Multi-step Forecasting ({horizon_minutes} minutes)...")

    if initial_window.ndim != 3 or initial_window.shape[0] != 1:
        raise ValueError(f"initial_window shape {initial_window.shape} invalid")

    predictions  = []
    current_input = initial_window.copy()
    passed        = True

    for step in range(1, steps + 1):
        try:
            raw_pred   = model.predict(current_input)
            pred_value = float(raw_pred.flatten()[0])
            predictions.append(pred_value)

            # Build a new timestep row; keep all features from the last known row
            new_row = current_input[:, -1:, :].copy()
            new_row[0, 0, target_feature_index] = pred_value

            # Shift
            current_input = np.concatenate([current_input[:, 1:, :], new_row], axis=1)

            print(f"[TC10] Step {step:>2d} (+{step * interval_minutes:>3d} min): volume = {pred_value:.4f}")

        except Exception as exc:
            print(f"[TC10] ERROR at step {step}: {exc}")
            passed = False
            break

    result_array = np.array(predictions)
    print(f"[TC10] Result: {'PASSED' if passed else 'FAILED'} (Forecast horizon reached: {horizon_minutes} min)")

    return {
        "predictions":     result_array,
        "horizon_minutes": horizon_minutes,
        "passed":          passed,
    }

if __name__ == "__main__":
    from testing.test_utils import get_test_args, load_test_context
    args = get_test_args()
    try:
        ctx = load_test_context(args.model)
        # Use first sample from x_test as initial window
        run_tc10(ctx["model"], ctx["x_test"][:1], steps=4)
    except Exception as e:
        print(f"[TC10] Result: FAILED (Execution Error: {e})")
        sys.exit(1)