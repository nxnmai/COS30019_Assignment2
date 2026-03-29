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
    Evaluates if the model generalizes well across different day types.
    """
    print("\n>>> Running TC05: Weekend vs Weekday Performance...")
    
    # Ensure index is datetime
    test_df.index = pd.to_datetime(test_df.index)
    
    # Dayofweek: Monday=0, Sunday=6. Weekends are 5 and 6.
    weekend_mask = test_df.index.dayofweek >= 5
    weekday_mask = ~weekend_mask
    
    # Evaluate Weekdays
    x_wd, y_wd = x_test[weekday_mask], y_test[weekday_mask]
    preds_wd = model.predict(x_wd)
    mae_wd = np.mean(np.abs(preds_wd.flatten() - y_wd.flatten()))
    
    # Evaluate Weekends
    x_we, y_we = x_test[weekend_mask], y_test[weekend_mask]
    preds_we = model.predict(x_we)
    mae_we = np.mean(np.abs(preds_we.flatten() - y_we.flatten()))
    
    print(f"[TC05] Weekday MAE: {mae_wd:.4f} | Weekend MAE: {mae_we:.4f}")
    return mae_wd, mae_we

if __name__ == "__main__":
    print("TC05 script loaded.")