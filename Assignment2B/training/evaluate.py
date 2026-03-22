from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting optional in headless env
    plt = None

from models.base_model import resolve_device, torch

DEFAULT_MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "transformer": {
        "module": "models.transformer_model",
        "class_candidates": ["TransformerRegressor", "TransformerModel"],
        "checkpoint": "data/processed/checkpoints/transformer.pt",
    },
    "lstm": {
        "module": "models.lstm_model",
        "class_candidates": ["LSTMRegressor", "LSTMModel", "LSTM"],
        "checkpoint": "data/processed/checkpoints/lstm.pt",
    },
    "gru": {
        "module": "models.gru_model",
        "class_candidates": ["GRURegressor", "GRUModel", "GRU"],
        "checkpoint": "data/processed/checkpoints/gru.pt",
    },
}


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-3, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _load_model_class(module_name: str, class_candidates: Iterable[str]):
    module = importlib.import_module(module_name)
    for class_name in class_candidates:
        model_cls = getattr(module, class_name, None)
        if model_cls is not None:
            return model_cls

    # Fallback: choose the first class defined in module that looks like a model.
    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and any(token in obj.__name__.lower() for token in ("lstm", "gru", "transformer")):
            return obj
    raise ValueError(f"Could not find a model class in module '{module_name}'.")


def _load_checkpoint_model(model_name: str, model_spec: Dict[str, object], checkpoint_path: Path, device: str):
    model_cls = _load_model_class(
        module_name=str(model_spec["module"]),
        class_candidates=model_spec["class_candidates"],  # type: ignore[arg-type]
    )

    if hasattr(model_cls, "load"):
        model, _ = model_cls.load(checkpoint_path, map_location=device)
        return model

    # Fallback for non-base-model implementations.
    model = model_cls()
    if torch is None:
        raise ImportError("PyTorch is required to load model checkpoints.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict(model, x: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    if hasattr(model, "predict"):
        return model.predict(x, batch_size=batch_size, device=device).reshape(-1)

    if torch is None:
        raise ImportError("PyTorch is required for model inference.")

    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            yhat = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yhat)
    if not preds:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds)


def evaluate_models(
    dataset_path: str | Path = "data/processed/datasets.npz",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    output_dir: str | Path = "data/processed/evaluation",
    model_names: Optional[List[str]] = None,
    checkpoint_overrides: Optional[Dict[str, str]] = None,
    device: str = "auto",
    batch_size: int = 1024,
    plot_window: int = 288,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    scaler_path = Path(scaler_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)
    test_x = data["test_x"]
    test_y = data["test_y"].reshape(-1)

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        y_true = scaler.inverse_transform(test_y.reshape(-1, 1)).reshape(-1)
    else:
        y_true = test_y

    selected_models = model_names or ["lstm", "gru", "transformer"]
    checkpoint_overrides = checkpoint_overrides or {}
    run_device = resolve_device(device)

    metrics_rows = []
    predictions_for_plot: Dict[str, np.ndarray] = {}

    for model_name in selected_models:
        if model_name not in DEFAULT_MODEL_SPECS:
            print(f"[WARN] Unknown model '{model_name}'. Skipping.")
            continue

        model_spec = DEFAULT_MODEL_SPECS[model_name]
        checkpoint_path = Path(checkpoint_overrides.get(model_name) or str(model_spec["checkpoint"]))
        if not checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found for {model_name}: {checkpoint_path}. Skipping.")
            continue

        model = _load_checkpoint_model(model_name, model_spec, checkpoint_path, run_device)
        pred_scaled = _predict(model, test_x, device=run_device, batch_size=batch_size)
        if scaler is not None:
            y_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        else:
            y_pred = pred_scaled

        model_metrics = {
            "model": model_name,
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape_percent": mape(y_true, y_pred),
        }
        metrics_rows.append(model_metrics)
        predictions_for_plot[model_name] = y_pred

    if not metrics_rows:
        raise RuntimeError("No model could be evaluated. Check available checkpoints and model implementations.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    metrics_csv = output_dir / "metrics_table.csv"
    metrics_md = output_dir / "metrics_table.md"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_md.write_text(metrics_df.to_markdown(index=False), encoding="utf-8")

    if plt is None:
        print("[WARN] matplotlib is not installed. Skipping plot generation.")
    else:
        n_models = len(predictions_for_plot)
        n = min(plot_window, len(y_true))
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=True)
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, pred) in zip(axes, predictions_for_plot.items()):
            ax.plot(y_true[:n], label="Actual", linewidth=2)
            ax.plot(pred[:n], label=f"{model_name.upper()} Predicted", linewidth=1.5)
            ax.set_title(f"{model_name.upper()} Predicted vs Actual (first {n} test points)")
            ax.set_ylabel("Flow (vehicles/hour)")
            ax.grid(alpha=0.25)
            ax.legend()

        axes[-1].set_xlabel("Test Time Step")
        fig.tight_layout()
        combined_plot = output_dir / "predicted_vs_actual.png"
        fig.savefig(combined_plot, dpi=160)
        plt.close(fig)

        # Also save per-model overlays for report insertion.
        for model_name, pred in predictions_for_plot.items():
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(y_true[:n], label="Actual", linewidth=2)
            ax.plot(pred[:n], label=f"{model_name.upper()} Predicted", linewidth=1.5)
            ax.set_title(f"{model_name.upper()} Predicted vs Actual")
            ax.set_xlabel("Test Time Step")
            ax.set_ylabel("Flow (vehicles/hour)")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"pred_vs_actual_{model_name}.png", dpi=160)
            plt.close(fig)

    summary = {
        "dataset_path": str(dataset_path),
        "scaler_path": str(scaler_path) if scaler is not None else None,
        "evaluated_models": list(predictions_for_plot.keys()),
        "metrics_csv": str(metrics_csv),
        "plot_window": int(plot_window),
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return metrics_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LSTM/GRU/Transformer checkpoints.")
    parser.add_argument("--dataset-path", default="data/processed/datasets.npz")
    parser.add_argument("--scaler-path", default="data/processed/scaler.pkl")
    parser.add_argument("--output-dir", default="data/processed/evaluation")
    parser.add_argument("--models", default="lstm,gru,transformer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--plot-window", type=int, default=288)
    parser.add_argument("--lstm-checkpoint", default="")
    parser.add_argument("--gru-checkpoint", default="")
    parser.add_argument("--transformer-checkpoint", default="")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    overrides = {
        key: value
        for key, value in {
            "lstm": args.lstm_checkpoint,
            "gru": args.gru_checkpoint,
            "transformer": args.transformer_checkpoint,
        }.items()
        if value
    }

    metrics_df = evaluate_models(
        dataset_path=args.dataset_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
        model_names=model_names,
        checkpoint_overrides=overrides,
        device=args.device,
        batch_size=args.batch_size,
        plot_window=args.plot_window,
    )
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
