import os
import sys
import torch
import numpy as np

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_tc01(model_class, model_config: dict, seed: int = 42) -> bool:
    """
    TC01: Model Initialization & Determinism Check.

    Verifies three things:
        1. The model can be instantiated without errors.
        2. No layer is initialised with NaN or all-zero weights.
        3. Initialisation is deterministic — two models created with the
           same seed produce identical weights across ALL layers.

    Args:
        model_class:  The model class to instantiate (e.g. LSTMModel).
        model_config: Keyword arguments forwarded to model_class(**model_config).
        seed:         Random seed used for both instantiations.

    Returns:
        True if all checks pass, False otherwise.
    """
    print(f"\n>>> Running TC01: Model Initialization & Determinism Check (seed={seed})...")

    try:
        # --- Instantiation 1 ---
        torch.manual_seed(seed)
        np.random.seed(seed)
        model1 = model_class(**model_config)

        # --- Instantiation 2 (same seed, fresh RNG state) ---
        torch.manual_seed(seed)
        np.random.seed(seed)
        model2 = model_class(**model_config)

    except Exception as exc:
        print(f"[TC01] FAILED: Could not instantiate model. Error: {exc}")
        return False

    # --- Check 1: model has trainable parameters ---
    params1 = list(model1.named_parameters())
    if not params1:
        print("[TC01] FAILED: Model has no trainable parameters.")
        return False

    # --- Check 2: scan ALL layers for NaN or all-zero weights ---
    nan_layers  = []
    zero_layers = []

    for name, param in params1:
        data = param.data
        if torch.isnan(data).any():
            nan_layers.append(name)
        if torch.all(data == 0):
            zero_layers.append(name)

    if nan_layers:
        print(f"[TC01] FAILED: NaN weights detected in layer(s): {nan_layers}")
        return False

    if zero_layers:
        # All-zero bias vectors are acceptable (common default); all-zero
        # weight matrices are not — they prevent gradient flow entirely.
        # positional_encoding in Transformer is often initialized to zero or near-zero 
        # and non-trainable in some implementations. Exclude it from the check.
        weight_zeros = [n for n in zero_layers if "bias" not in n and "positional_encoding" not in n]
        if weight_zeros:
            print(f"[TC01] FAILED: All-zero weight matrices in layer(s): {weight_zeros}")
            return False
        else:
            print(f"[TC01] NOTE: All-zero bias(es) detected (acceptable): {zero_layers}")

    # --- Check 3: determinism — compare ALL parameter tensors ---
    params2      = dict(model2.named_parameters())
    mismatch     = []

    for name, param in params1:
        if name not in params2:
            mismatch.append(f"{name} (missing in model2)")
            continue
        if not torch.equal(param.data, params2[name].data):
            mismatch.append(name)

    if mismatch:
        print(f"[TC01] FAILED: Non-deterministic weights in layer(s): {mismatch}")
        return False

    total_params = sum(p.numel() for _, p in params1)
    print(f"[TC01] Layers checked    : {len(params1)}")
    print(f"[TC01] Total parameters  : {total_params:,}")
    print(f"[TC01] Result: PASSED — weights are valid and deterministic.")
    return True


if __name__ == "__main__":
    import argparse
    import importlib
    from main import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "gru", "bilstm", "transformer"])
    args = parser.parse_args()

    cfg = load_config("config.ini")
    model_name = args.model.lower()
    
    try:
        # Load model class by name
        module_name = f"models.{model_name}_model"
        module = importlib.import_module(module_name)
        
        if model_name == "bilstm":
            class_name = "BiLSTMRegressor" # Use Regressor as per registry
        elif model_name == "transformer":
            class_name = "TransformerRegressor"
        else:
            class_name = f"{model_name.upper()}Regressor"
            
        model_class = getattr(module, class_name)
        
        # Determine instantiation config
        if model_name == "transformer":
            m_config = {"input_dim": 21, "d_model": 64, "nhead": 4, "num_layers": 2}
        else:
            m_config = {"input_dim": 21, "hidden_size": 64, "num_layers": 2}

        success = run_tc01(model_class, m_config)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[TC01] Result: FAILED — Execution error: {e}")
        sys.exit(1)