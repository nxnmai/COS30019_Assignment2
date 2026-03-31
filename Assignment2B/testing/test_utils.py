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
    ds_rel_path = cfg.get("paths", "dataset_npz", fallback="data/processed/datasets.npz")
    ds_path = PROJECT_ROOT / ds_rel_path
    
    if not ds_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {ds_path}. Please run feature engineering first.")
        
    ds = np.load(ds_path)
    # Correct keys as per train.py loader
    x_test = ds["test_x"]
    y_test = ds["test_y"]
    
    # 3. Load full dataframe for time-based tests (TC04, TC05)
    csv_rel_path = cfg.get("paths", "clean_scats_csv", fallback="data/processed/scats_oct2006_clean.csv")
    csv_path = PROJECT_ROOT / csv_rel_path
    
    if csv_path.exists():
        full_df = pd.read_csv(csv_path, parse_dates=["date_time"])
        
        # Determine the test window for ONE stream.
        # We need to know how many samples each stream contributes to x_test.
        num_streams = cfg.getint("preprocessing", "used_stream_count", fallback=1)
        if num_streams > 0 and len(x_test) % num_streams == 0:
            samples_per_stream = len(x_test) // num_streams
        else:
            samples_per_stream = len(x_test) # fallback

        # Take any arbitrary stream (the first one) to get its 5-day test window
        first_stream_id = full_df["movement_id"].iloc[0]
        stream_data = full_df[full_df["movement_id"] == first_stream_id].copy()
        
        # The test_x windows are derived from the end of the stream data.
        # Length is samples_per_stream.
        test_df = stream_data.iloc[-samples_per_stream:].copy()
        test_df = test_df.set_index("date_time")
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
