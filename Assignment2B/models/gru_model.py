"""GRU model for one-step-ahead traffic flow regression."""
from __future__ import annotations

from typing import Any, Dict

from models.base_model import BaseTimeSeriesModel, _require_torch, nn, torch


if torch is not None:

    class GRURegressor(BaseTimeSeriesModel):
        """
        Stacked GRU for one-step-ahead traffic flow regression.

        Mirrors LSTMRegressor architecture but uses GRU cells —
        fewer parameters, typically faster training with comparable accuracy.

        Input shape:  (batch, seq_len, input_dim)
        Output shape: (batch, 1)
        """

        def __init__(
            self,
            input_dim: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout_rate = dropout

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def get_init_kwargs(self) -> Dict[str, Any]:
            return {
                "input_dim": self.input_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_rate,
            }

        def forward(self, x):  # type: ignore[override]
            # x: (batch, seq_len) or (batch, seq_len, input_dim)
            if x.ndim == 2:
                x = x.unsqueeze(-1)
            out, _ = self.gru(x)           # (batch, seq_len, hidden_size)
            last = out[:, -1, :]           # take last timestep
            last = self.dropout(last)
            return self.fc(last)           # (batch, 1)


else:

    class GRURegressor(BaseTimeSeriesModel):  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()
