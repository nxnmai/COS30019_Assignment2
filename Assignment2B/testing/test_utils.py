import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from main import load_config, _load_model_and_scaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "gru", "bilstm", "transformer"])
    return parser.parse_args()

def load_test_context(model_name: str):
    """Loads config, model, and standard test data."""
    cfg = load_config(str(PROJECT_ROOT / "config.ini"))
    
    # 1. Load model
    try:
        model, scaler, device = _load_model_and_scaler(cfg, model_name=model_name)
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_name}': {e}")
        # Return a dummy context or raise so the test can handle it
        raise e

    # 2. Load dataset
    data_dir = PROJECT_ROOT / "data" / "processed"
    ds_path = data_dir / "test_dataset.npz"
    
    if not ds_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {ds_path}. Please run feature engineering first.")
        
    ds = np.load(ds_path)
    x_test = ds["x"]
    y_test = ds["y"]
    
    # 3. Load full dataframe for time-based tests (TC04, TC05)
    csv_path = PROJECT_ROOT / cfg.get("paths", "clean_scats_csv", fallback="data/processed/clean_scats_data.csv")
    if csv_path.exists():
        # Only load a slice or just enough to match x_test
        # Usually x_test matches the end of the dataframe
        full_df = pd.read_csv(csv_path, parse_dates=["date_time"])
        # We need to align the dataframe with the test set
        # The test set is the last N days.
        test_days = cfg.getint("preprocessing", "test_days", fallback=5)
        # Simplified: take the last len(x_test) rows from the df
        test_df = full_df.iloc[-len(x_test):].copy()
    else:
        test_df = pd.DataFrame()

    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "x_test": x_test,
        "y_test": y_test,
        "test_df": test_df,
        "cfg": cfg
    }
