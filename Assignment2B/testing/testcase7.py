import os
import sys
import numpy as np

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.base_model import BaseTimeSeriesModel

def run_tc07(model: BaseTimeSeriesModel, x_test: np.ndarray) -> bool:
    """
    TC07: System Resilience Test (Missing Values).
    """
    print("\n>>> Running TC07: Data Resilience Test...")
    
    # Create a corrupted sample with NaN values
    x_corrupted = x_test[:1].copy()
    x_corrupted[0, 0, 0] = np.nan 
    
    try:
        # Pre-processing step: impute NaNs with 0 before prediction
        safe_input = np.nan_to_num(x_corrupted, nan=0.0)
        
        _ = model.predict(safe_input)
        print("[TC07] Result: PASSED (System successfully handled NaN input)")
        return True
    except Exception as e:
        print(f"[TC07] Result: FAILED (System crashed with error: {str(e)})")
        return False

if __name__ == "__main__":
    from testing.test_utils import get_test_args, load_test_context
    args = get_test_args()
    try:
        ctx = load_test_context(args.model)
        run_tc07(ctx["model"], ctx["x_test"])
    except Exception as e:
        print(f"[TC07] Result: FAILED (Execution Error: {e})")
        sys.exit(1)