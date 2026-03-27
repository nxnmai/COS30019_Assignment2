"""Bidirectional LSTM (BiLSTM) model for one-step-ahead traffic flow regression."""
from __future__ import annotations

from typing import Any, Dict

from models.base_model import BaseTimeSeriesModel, _require_torch, nn, torch


if torch is not None:

    class BiLSTMRegressor(BaseTimeSeriesModel):
        """
        Stacked Bidirectional LSTM for one-step-ahead traffic flow regression.

        Processes the input sequence in both forward and backward directions,
        doubling the effective representation width. This captures both
        historical momentum and future-context patterns within the lookback window.

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

            # bidirectional=True → output dim = hidden_size * 2
            self.bilstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            # input to fc is 2× hidden_size (forward + backward)
            self.fc = nn.Linear(hidden_size * 2, 1)

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
            out, _ = self.bilstm(x)        # (batch, seq_len, hidden_size * 2)
            last = out[:, -1, :]           # last timestep (both directions)
            last = self.dropout(last)
            return self.fc(last)           # (batch, 1)


else:

    class BiLSTMRegressor(BaseTimeSeriesModel):  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()
