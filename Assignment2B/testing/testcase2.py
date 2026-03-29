import os
import sys
import numpy as np

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.base_model import BaseTimeSeriesModel


def run_tc02(
    model: BaseTimeSeriesModel,
    seq_len: int,
    input_dim: int,
) -> bool:
    """
    TC02: Model Input / Output Consistency Check.

    Verifies that the model:
        1. Accepts a correctly shaped input (1, seq_len, input_dim) without crashing.
        2. Returns exactly ONE scalar prediction per sample (flexible shape check).
        3. The prediction is a finite number — not NaN or infinity.
        4. Handles a larger batch (batch_size=8) correctly as well.

    Args:
        model:     A trained (or freshly initialised) model.
        seq_len:   Window / sequence length used during training.
        input_dim: Number of input features per timestep.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("\n>>> Running TC02: Model I/O Tensor Check...")

    all_passed = True

    # ------------------------------------------------------------------
    # Check A: single-sample inference (batch_size = 1)
    # ------------------------------------------------------------------
    dummy_single = np.random.rand(1, seq_len, input_dim).astype(np.float32)

    try:
        pred_single = model.predict(dummy_single)
    except Exception as exc:
        print(f"[TC02] FAILED (single sample): Model crashed. Error: {exc}")
        return False

    # FIX: accept any shape that contains exactly 1 element.
    # (1, 1), (1,), and scalar 0-d arrays are all valid scalar outputs.
    if pred_single.size != 1:
        print(f"[TC02] FAILED (single sample): Expected 1 output value, "
              f"got shape {pred_single.shape} ({pred_single.size} values).")
        all_passed = False
    else:
        scalar_val = float(pred_single.flatten()[0])
        if not np.isfinite(scalar_val):
            print(f"[TC02] FAILED (single sample): Prediction is non-finite: {scalar_val}")
            all_passed = False
        else:
            print(f"[TC02] PASSED (single sample): shape={pred_single.shape}, "
                  f"value={scalar_val:.6f}")

    # ------------------------------------------------------------------
    # Check B: batch inference (batch_size = 8)
    # ------------------------------------------------------------------
    batch_size   = 8
    dummy_batch  = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)

    try:
        pred_batch = model.predict(dummy_batch)
    except Exception as exc:
        print(f"[TC02] FAILED (batch={batch_size}): Model crashed. Error: {exc}")
        return False

    if pred_batch.size != batch_size:
        print(f"[TC02] FAILED (batch={batch_size}): Expected {batch_size} output values, "
              f"got shape {pred_batch.shape} ({pred_batch.size} values).")
        all_passed = False
    else:
        flat_vals = pred_batch.flatten()
        if not np.all(np.isfinite(flat_vals)):
            bad = flat_vals[~np.isfinite(flat_vals)]
            print(f"[TC02] FAILED (batch={batch_size}): "
                  f"{len(bad)} non-finite prediction(s) detected.")
            all_passed = False
        else:
            print(f"[TC02] PASSED (batch={batch_size}): shape={pred_batch.shape}, "
                  f"values in [{flat_vals.min():.4f}, {flat_vals.max():.4f}]")

    status = "PASSED" if all_passed else "FAILED"
    print(f"[TC02] Result: {status}")
    return all_passed


if __name__ == "__main__":
    print("TC02 script loaded.")