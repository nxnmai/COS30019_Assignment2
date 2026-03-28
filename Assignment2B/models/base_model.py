from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - import guard for environments without torch
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for model training/inference. Install torch before using models/*."
        )


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    min_delta: float = 1e-5
    device: str = "cpu"
    checkpoint_path: str | Path = "data/processed/checkpoints/model.pt"
    verbose: bool = True


def resolve_device(device: str = "cpu") -> str:
    _require_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def build_data_loader(
    x: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    _require_torch()
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if torch is not None:

    class BaseTimeSeriesModel(nn.Module):
        """
        Base class for sequence-to-one regressors.
        Subclasses only implement __init__ + forward.
        """

        def __init__(self) -> None:
            super().__init__()

        def get_init_kwargs(self) -> Dict[str, Any]:
            """
            Override in subclasses to persist constructor kwargs in checkpoints.
            """
            return {}

        def fit(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            val_x: Optional[np.ndarray] = None,
            val_y: Optional[np.ndarray] = None,
            config: Optional[TrainingConfig] = None,
        ) -> Dict[str, list]:
            config = config or TrainingConfig()
            device = resolve_device(config.device)
            self.to(device)

            train_loader = build_data_loader(train_x, train_y, batch_size=config.batch_size, shuffle=True)
            val_loader = None
            if val_x is not None and val_y is not None and len(val_x) > 0:
                val_loader = build_data_loader(val_x, val_y, batch_size=config.batch_size, shuffle=False)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            checkpoint_path = Path(config.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            history: Dict[str, list] = {"train_loss": [], "val_loss": []}
            best_val = float("inf")
            epochs_without_improvement = 0

            for epoch in range(1, config.epochs + 1):
                self.train()
                train_loss_total = 0.0
                train_batches = 0

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    preds = self(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()

                    train_loss_total += float(loss.item())
                    train_batches += 1

                train_loss = train_loss_total / max(train_batches, 1)
                history["train_loss"].append(train_loss)

                if val_loader is None:
                    val_loss = train_loss
                else:
                    self.eval()
                    val_loss_total = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb = xb.to(device)
                            yb = yb.to(device)
                            preds = self(xb)
                            loss = criterion(preds, yb)
                            val_loss_total += float(loss.item())
                            val_batches += 1
                    val_loss = val_loss_total / max(val_batches, 1)
                history["val_loss"].append(val_loss)

                improved = val_loss < (best_val - config.min_delta)
                if improved:
                    best_val = val_loss
                    epochs_without_improvement = 0
                    self.save(
                        checkpoint_path,
                        extra={
                            "epoch": epoch,
                            "best_val_loss": best_val,
                            "history": history,
                        },
                    )
                else:
                    epochs_without_improvement += 1

                if config.verbose:
                    print(
                        f"Epoch {epoch:03d}/{config.epochs} "
                        f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                        flush=True,
                    )

                if config.patience > 0 and epochs_without_improvement >= config.patience:
                    if config.verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break

            # Ensure model state is the best checkpoint before returning.
            if checkpoint_path.exists():
                self.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
                self.to(device)
                self.eval()

            return history

        def predict(
            self,
            x: np.ndarray,
            batch_size: int = 1024,
            device: str = "cpu",
        ) -> np.ndarray:
            device = resolve_device(device)
            self.to(device)
            self.eval()

            loader = build_data_loader(x, y=None, batch_size=batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(device)
                    yhat = self(xb).detach().cpu().numpy()
                    preds.append(yhat)
            if not preds:
                return np.empty((0, 1), dtype=np.float32)
            return np.concatenate(preds, axis=0)

        def save(self, file_path: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
            payload = {
                "state_dict": self.state_dict(),
                "model_class": self.__class__.__name__,
                "init_kwargs": self.get_init_kwargs(),
                "extra": extra or {},
            }
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, file_path)

        @classmethod
        def load(
            cls,
            file_path: str | Path,
            map_location: str = "cpu",
            **override_kwargs: Any,
        ) -> Tuple["BaseTimeSeriesModel", Dict[str, Any]]:
            _require_torch()
            file_path = Path(file_path)
            checkpoint = torch.load(file_path, map_location=map_location)
            init_kwargs = checkpoint.get("init_kwargs", {})
            init_kwargs.update(override_kwargs)
            model = cls(**init_kwargs)  # type: ignore[arg-type]
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            return model, checkpoint.get("extra", {})

else:

    class BaseTimeSeriesModel:  # pragma: no cover - fallback when torch is unavailable
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

        def fit(self, *args: Any, **kwargs: Any) -> Dict[str, list]:
            _require_torch()
            return {}

        def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
            _require_torch()
            return np.empty((0, 1), dtype=np.float32)

        def save(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> Tuple["BaseTimeSeriesModel", Dict[str, Any]]:
            _require_torch()
            return cls(), {}
