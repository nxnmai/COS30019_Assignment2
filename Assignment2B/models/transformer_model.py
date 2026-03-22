from __future__ import annotations

from typing import Any, Dict

from models.base_model import BaseTimeSeriesModel, _require_torch, nn, torch


if torch is not None:

    class TransformerRegressor(BaseTimeSeriesModel):
        """
        Transformer encoder for one-step-ahead traffic flow regression.

        Input shape:  (batch, seq_len=12, input_dim=1)
        Output shape: (batch, 1)
        """

        def __init__(
            self,
            input_dim: int = 1,
            seq_len: int = 12,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.seq_len = seq_len
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout

            self.input_projection = nn.Linear(input_dim, d_model)
            self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.output_head = nn.Linear(d_model, 1)

        def get_init_kwargs(self) -> Dict[str, Any]:
            return {
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
            }

        def forward(self, x):  # type: ignore[override]
            if x.ndim == 2:
                x = x.unsqueeze(-1)
            if x.ndim != 3:
                raise ValueError(f"Expected 3D input (batch, seq, features), got shape {tuple(x.shape)}")

            seq_len = x.size(1)
            if seq_len > self.seq_len:
                raise ValueError(f"Input sequence length {seq_len} exceeds configured seq_len {self.seq_len}")

            h = self.input_projection(x)
            h = h + self.positional_encoding[:, :seq_len, :]
            h = self.encoder(h)
            h = self.norm(h)

            # Sequence-to-one regression using last timestep representation.
            last_token = h[:, -1, :]
            return self.output_head(last_token)


else:

    class TransformerRegressor(BaseTimeSeriesModel):  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()
