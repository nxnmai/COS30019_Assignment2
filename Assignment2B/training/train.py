"""Training CLI for LSTM / GRU / BiLSTM / Transformer models."""
from __future__ import annotations

import argparse
import configparser
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from models.base_model import TrainingConfig, resolve_device


# ---------------------------------------------------------------------------
# Model registry — maps CLI name → (module, class_candidates)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "lstm": {
        "module": "models.lstm_model",
        "class_candidates": ["LSTMRegressor"],
        "defaults": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
    },
    "gru": {
        "module": "models.gru_model",
        "class_candidates": ["GRURegressor"],
        "defaults": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
    },
    "bilstm": {
        "module": "models.bilstm_model",
        "class_candidates": ["BiLSTMRegressor"],
        "defaults": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
    },
    "transformer": {
        "module": "models.transformer_model",
        "class_candidates": ["TransformerRegressor"],
        "defaults": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
        },
    },
}


def _load_model_class(module_name: str, class_candidates: list):
    import importlib
    module = importlib.import_module(module_name)
    for name in class_candidates:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise ValueError(f"No model class found in {module_name}. Tried: {class_candidates}")


def _load_datasets(dataset_path: Path, scaler_path: Path):
    """Load pre-built .npz dataset. Build it first if missing."""
    if not dataset_path.exists():
        print(f"[INFO] dataset not found at {dataset_path}. Running feature engineering...")
        from preprocessing.feature_engineering import prepare_datasets
        from preprocessing.data_loader import load_scats_data

        cfg_path = PROJECT_ROOT / "config.ini"
        cfg = configparser.ConfigParser()
        cfg.read(cfg_path)

        raw_xls = (PROJECT_ROOT / cfg.get("paths", "raw_scats_xls")).resolve()
        clean_csv = (PROJECT_ROOT / cfg.get("paths", "clean_scats_csv")).resolve()

        if clean_csv.exists():
            import pandas as pd
            df = pd.read_csv(clean_csv, parse_dates=["date", "date_time"])
        else:
            df = load_scats_data(raw_xls)
            clean_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(clean_csv, index=False)

        lookback = cfg.getint("preprocessing", "lookback", fallback=12)
        train_days = cfg.getint("preprocessing", "train_days", fallback=21)
        val_days = cfg.getint("preprocessing", "val_days", fallback=5)
        test_days = cfg.getint("preprocessing", "test_days", fallback=5)
        min_days = cfg.getint("preprocessing", "min_available_days", fallback=21)

        prepare_datasets(
            dataframe=df,
            lookback=lookback,
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            min_available_days=min_days,
            scaler_output_path=scaler_path,
            dataset_output_path=dataset_path,
        )
        print(f"[INFO] Datasets saved to {dataset_path}")

    data = np.load(dataset_path)
    return (
        data["train_x"], data["train_y"],
        data["val_x"],   data["val_y"],
        data["test_x"],  data["test_y"],
    )


def train(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: str,
    dataset_path: Path,
    scaler_path: Path,
    checkpoint_dir: Path,
) -> None:
    spec = MODEL_REGISTRY.get(model_name.lower())
    if spec is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"  Epochs={epochs}  Batch={batch_size}  LR={learning_rate}  Patience={patience}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load data
    train_x, train_y, val_x, val_y, _, _ = _load_datasets(dataset_path, scaler_path)
    print(f"[DATA] train={train_x.shape}  val={val_x.shape}")

    # Build model
    model_cls = _load_model_class(spec["module"], spec["class_candidates"])
    seq_len = train_x.shape[1]

    # Construct model with defaults (seq_len needed for Transformer only)
    init_kwargs = dict(spec["defaults"])
    if model_name == "transformer":
        init_kwargs["seq_len"] = seq_len
    model = model_cls(**init_kwargs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {model_cls.__name__} — {param_count:,} trainable parameters")

    # Set up training config
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"

    cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        device=resolve_device(device),
        checkpoint_path=checkpoint_path,
        verbose=True,
    )

    # Train
    t0 = time.perf_counter()
    history = model.fit(train_x, train_y, val_x=val_x, val_y=val_y, config=cfg)
    elapsed = time.perf_counter() - t0

    # Summary
    best_val = min(history["val_loss"]) if history["val_loss"] else float("nan")
    print(f"\n[DONE] {model_name.upper()} training complete in {elapsed:.1f}s", flush=True)
    print(f"       Best val_loss = {best_val:.6f}", flush=True)
    print(f"       Checkpoint saved -> {checkpoint_path}\n", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a traffic flow model (LSTM / GRU / BiLSTM / Transformer).")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY), help="Model architecture to train")
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config.ini)")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config.ini)")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config.ini)")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (overrides config.ini)")
    p.add_argument("--device", default=None, help="Device: cpu / cuda / auto (overrides config.ini)")
    p.add_argument("--dataset-path", default=None, help="Path to datasets.npz")
    p.add_argument("--scaler-path", default=None, help="Path to scaler.pkl")
    p.add_argument("--checkpoint-dir", default=None, help="Directory to save model checkpoints")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Load config.ini for defaults
    cfg = configparser.ConfigParser()
    cfg_path = PROJECT_ROOT / "config.ini"
    if cfg_path.exists():
        cfg.read(cfg_path)

    def _get(section: str, key: str, fallback):
        try:
            return cfg.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return str(fallback)

    epochs = args.epochs or int(_get("training", "epochs", 100))
    batch_size = args.batch_size or int(_get("training", "batch_size", 128))
    lr = args.lr or float(_get("training", "learning_rate", 1e-3))
    patience = args.patience or int(_get("training", "patience", 12))
    device = args.device or _get("inference", "device", "auto")

    dataset_path = Path(args.dataset_path) if args.dataset_path else (PROJECT_ROOT / _get("paths", "dataset_npz", "data/processed/datasets.npz")).resolve()
    scaler_path = Path(args.scaler_path) if args.scaler_path else (PROJECT_ROOT / _get("paths", "scaler_path", "data/processed/scaler.pkl")).resolve()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (PROJECT_ROOT / _get("paths", "checkpoints_dir", "data/processed/checkpoints")).resolve()

    train(
        model_name=args.model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        patience=patience,
        device=device,
        dataset_path=dataset_path,
        scaler_path=scaler_path,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
