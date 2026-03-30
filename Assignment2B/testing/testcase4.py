import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.base_model import BaseTimeSeriesModel


def run_tc04(model: BaseTimeSeriesModel, test_df: pd.DataFrame, x_test: np.ndarray, y_test: np.ndarray):
    """
    TC04: Morning and Evening Peak Hour Sensitivity Test.
    Morning: 07:00 - 09:00 | Evening: 16:00 - 18:00
    """
    print("\n>>> Running TC04: Peak Hour Sensitivity Test...")
    
    # Ensure the dataframe index is datetime type for filtering
    test_df.index = pd.to_datetime(test_df.index)
    
    # Define peak hour masks
    morning_mask = (test_df.index.hour >= 7) & (test_df.index.hour <= 9)
    evening_mask = (test_df.index.hour >= 16) & (test_df.index.hour <= 18)
    peak_mask = morning_mask | evening_mask
    
    # Filter test data for peak hours
    x_peak = x_test[peak_mask]
    y_peak = y_test[peak_mask]
    
    if len(x_peak) == 0:
        print("[TC04] Result: No peak hour data found in the provided dataset.")
        return 0.0
        
    # Model inference
    predictions = model.predict(x_peak)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions.flatten() - y_peak.flatten()))
    
    print(f"[TC04] Result: Peak Hour MAE = {mae:.4f}")
    return mae


if __name__ == "__main__":
    from testing.test_utils import get_test_args, load_test_context
    args = get_test_args()
    try:
        ctx = load_test_context(args.model)
        run_tc04(ctx["model"], ctx["test_df"], ctx["x_test"], ctx["y_test"])
    except Exception as e:
        print(f"[TC04] Result: FAILED (Execution Error: {e})")
        sys.exit(1)