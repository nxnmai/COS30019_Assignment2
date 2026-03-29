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

    Uses the model's own predictions as future inputs to forecast
    multiple time steps ahead — e.g. 4 steps × 15 min = 60 min ahead.

    This is the actual 'sliding window' test:
        input:  [t-n, ..., t-1, t]
        step 1: predict t+1  → slide window to [t-n+1, ..., t, t+1_pred]
        step 2: predict t+2  → slide window further, and so on.

    Args:
        model:                Trained model implementing BaseTimeSeriesModel.
        initial_window:       Seed input of shape (1, window_size, num_features).
        steps:                Number of 15-minute intervals to forecast ahead.
        target_feature_index: Column index of the traffic volume feature in the
                              feature array. Defaults to 0. Change this if your
                              preprocessing puts volume at a different column.
        interval_minutes:     Length of each time step in minutes (default: 15).

    Returns:
        A dict containing:
            - 'predictions'     (np.ndarray): Forecasted values, shape (steps,).
            - 'horizon_minutes' (int):        Total forecast horizon in minutes.
            - 'passed'          (bool):       True if all steps completed without error.
    """
    horizon_minutes = steps * interval_minutes
    print(f"\n>>> Running TC10: Recursive Multi-step Forecasting "
          f"(next {horizon_minutes} minutes / {steps} steps)...")

    # --- Input validation ---
    if initial_window.ndim != 3 or initial_window.shape[0] != 1:
        raise ValueError(
            f"[TC10] initial_window must have shape (1, window_size, num_features), "
            f"got {initial_window.shape}"
        )

    window_size  = initial_window.shape[1]
    num_features = initial_window.shape[2]

    if target_feature_index >= num_features:
        raise ValueError(
            f"[TC10] target_feature_index={target_feature_index} is out of range "
            f"for num_features={num_features}. Check your preprocessing pipeline."
        )

    if steps < 1:
        raise ValueError(f"[TC10] steps must be >= 1, got {steps}.")

    predictions  = []
    current_input = initial_window.copy()  # (1, window_size, num_features)
    passed        = True

    for step in range(1, steps + 1):
        try:
            # --- Predict next step ---
            raw_pred   = model.predict(current_input)          # (1, 1) or (1,)
            pred_value = float(raw_pred.flatten()[0])
            predictions.append(pred_value)

            # --- Slide the window forward ---
            # Build a new timestep row; keep all features from the last known row
            # so that engineered features (hour, day_of_week, etc.) are preserved,
            # then overwrite only the target feature with the predicted value.
            new_row = current_input[:, -1:, :].copy()          # (1, 1, num_features)
            new_row[0, 0, target_feature_index] = pred_value

            # Shift: drop the oldest timestep, append the new predicted row
            current_input = np.concatenate(
                [current_input[:, 1:, :], new_row], axis=1
            )                                                   # still (1, window_size, num_features)

            print(f"[TC10] Step {step:>2d} (+{step * interval_minutes:>3d} min): "
                  f"predicted volume = {pred_value:.4f}")

        except Exception as exc:
            print(f"[TC10] ERROR at step {step}: {exc}")
            passed = False
            break

    result_array = np.array(predictions)

    print(f"\n[TC10] Forecast horizon : {horizon_minutes} minutes")
    print(f"[TC10] Steps completed  : {len(predictions)} / {steps}")
    print(f"[TC10] Forecasted values: {np.round(result_array, 4).tolist()}")
    print(f"[TC10] Result           : {'PASSED' if passed else 'FAILED'}")

    return {
        "predictions":     result_array,
        "horizon_minutes": horizon_minutes,
        "passed":          passed,
    }


if __name__ == "__main__":
    print("TC10 script loaded.")