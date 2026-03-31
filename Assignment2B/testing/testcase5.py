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

def run_tc05(model: BaseTimeSeriesModel, test_df: pd.DataFrame, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """
    TC05: Weekend vs Weekday Performance Comparison.
    """
    print("\n>>> Running TC05: Weekend vs Weekday Performance...")
    
    # Dayofweek: Monday=0, Sunday=6. Weekends are 5 and 6.
    base_mask_we = (test_df.index.dayofweek >= 5) # Already a numpy array / index
    base_mask_wd = (~base_mask_we)
    
    # Use np.resize to cycle the base_masks across all sensors for the total test length.
    # This is more robust to slight mismatches than np.tile.
    weekend_mask = np.resize(base_mask_we, len(x_test))
    weekday_mask = np.resize(base_mask_wd, len(x_test))
    
    # Evaluate Weekdays
    x_wd, y_wd = x_test[weekday_mask], y_test[weekday_mask]
    preds_wd = model.predict(x_wd)
    mae_wd = np.mean(np.abs(preds_wd.flatten() - y_wd.flatten()))
    
    # Evaluate Weekends
    x_we, y_we = x_test[weekend_mask], y_test[weekend_mask]
    preds_we = model.predict(x_we)
    mae_we = np.mean(np.abs(preds_we.flatten() - y_we.flatten()))
    
    print(f"[TC05] Weekday MAE: {mae_wd:.4f} | Weekend MAE: {mae_we:.4f}")
    
    # Custom Result String formatting for GUI Parser
    print(f"[TC05] Result: Weekday MAE={mae_wd:.4f}, Weekend MAE={mae_we:.4f}")
    return mae_wd, mae_we

if __name__ == "__main__":
    from testing.test_utils import get_test_args, load_test_context
    args = get_test_args()
    try:
        ctx = load_test_context(args.model)
        run_tc05(ctx["model"], ctx["test_df"], ctx["x_test"], ctx["y_test"])
    except Exception as e:
        print(f"[TC05] Result: FAILED (Execution Error: {e})")
        sys.exit(1)