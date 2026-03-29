from __future__ import annotations
 
import logging
from typing import Any, Dict
 
from models.base_model import BaseTimeSeriesModel, _require_torch, nn, torch
 
logger = logging.getLogger(__name__)
 
 
if torch is not None:
 
    class GRURegressor(BaseTimeSeriesModel):
        """
        Stacked GRU with dual temporal pooling and a residual skip path.
 
        Parameters
        ----------
        input_dim : int
            Number of input features per time step. For the basic SCATS
            pipeline this is 1 (raw flow count, normalised upstream).
            Increase when time-of-day or day-of-week features are added.
        hidden_size : int
            GRU hidden units. Default 32 keeps total params ≈ 14k,
            appropriate for the one-month SCATS dataset size.
        num_layers : int
            Number of stacked GRU layers. Inter-layer dropout is applied
            when num_layers > 1; a warning is logged otherwise.
        dropout : float
            Dropout probability applied between GRU layers and inside the
            MLP head. Slightly higher default (0.3) to regularise the
            small SCATS dataset.
        head_hidden_dim : int
            Width of the hidden layer in the MLP regression head.
 
        Notes
        -----
        ``get_init_kwargs`` returns keys that match ``__init__`` parameter
        names exactly — this is required for checkpoint save/reload via
        ``BaseTimeSeriesModel.load()``.
        """
 
        def __init__(
            self,
            input_dim: int = 1,
            hidden_size: int = 32,        # tuned down: ~14k params on SCATS
            num_layers: int = 2,
            dropout: float = 0.3,         # slightly higher: regularise small dataset
            head_hidden_dim: int = 16,    # tuned down for same reason
        ) -> None:
            super().__init__()
 
            # Store attributes under the SAME names as __init__ parameters.
            # get_init_kwargs() returns these verbatim, so the keys must match
            # the parameter names exactly for checkpoint reload to work.
            self.input_dim       = input_dim
            self.hidden_size     = hidden_size
            self.num_layers      = num_layers
            self.dropout         = dropout        # NOT self.dropout_rate
            self.head_hidden_dim = head_hidden_dim
 
            if num_layers == 1 and dropout > 0.0:
                logger.warning(
                    "GRURegressor: num_layers=1 disables inter-layer dropout "
                    "(PyTorch limitation). dropout=%.2f only applies inside "
                    "the MLP head.",
                    dropout,
                )
 
            # ── Recurrent backbone ────────────────────────────────────────────
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
 
            # ── Residual projection ───────────────────────────────────────────
            # Maps last-step hidden (B, H) → (B, 2H) to match the concat dim,
            # then added element-wise before LayerNorm.
            # Initialised near-zero so the skip starts quiet and only
            # contributes once the GRU features prove insufficient.
            feature_dim = hidden_size * 2
            self.residual_proj = nn.Linear(hidden_size, feature_dim, bias=False)
 
            # ── Normalisation + MLP head ──────────────────────────────────────
            self.norm = nn.LayerNorm(feature_dim)
 
            self.head = nn.Sequential(
                nn.Linear(feature_dim, head_hidden_dim),
                nn.GELU(),            # smoother gradient than ReLU for regression
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
 
            self._reset_parameters()
 
        # ── Weight initialisation ─────────────────────────────────────────────
 
        def _reset_parameters(self) -> None:
            """
            Initialise weights for stable early training.
 
            Strategy
            --------
            - GRU input weights  : Xavier uniform  (balances fan-in/fan-out)
            - GRU hidden weights : Orthogonal      (prevents hidden-state
                                                     explosion / collapse)
            - GRU biases         : Zero
            - Residual proj      : Near-zero normal (skip is quiet at init)
            - Head Linear 0      : Kaiming normal  (suited to GELU)
            - Head Linear 1      : Xavier uniform  (output layer)
            - All biases         : Zero
            """
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
 
            # Residual skip starts quiet
            nn.init.normal_(self.residual_proj.weight, std=0.02)
 
            # MLP head
            for i, module in enumerate(self.head):
                if isinstance(module, nn.Linear):
                    if i == 0:
                        # Kaiming (fan-in) for the GELU-activated first layer
                        nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    else:
                        nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
 
        # ── Serialisation — key names MUST match __init__ params exactly ──────
 
        def get_init_kwargs(self) -> Dict[str, Any]:
            """
            Return constructor arguments needed to rebuild this model from a
            checkpoint. Keys must match ``__init__`` parameter names exactly.
 
            Called by ``BaseTimeSeriesModel.save()`` / ``.load()``.
 
            .. important::
               The key ``"dropout"`` matches the ``__init__`` parameter name.
               Do **not** change it to ``"dropout_rate"`` — that would break
               checkpoint reload because Python would pass an unexpected kwarg.
            """
            return {
                "input_dim":       self.input_dim,
                "hidden_size":     self.hidden_size,
                "num_layers":      self.num_layers,
                "dropout":         self.dropout,        # matches param name
                "head_hidden_dim": self.head_hidden_dim,
            }
 
        # ── Human-readable summary ────────────────────────────────────────────
 
        def __repr__(self) -> str:
            """
            Display a concise summary including total / trainable parameter
            counts. Shown in the GUI Train tab log and in the report's
            model-comparison section.
 
            Example output
            --------------
            GRURegressor(input_dim=1, hidden_size=32, num_layers=2,
                         dropout=0.3, head_hidden_dim=16)
              | params: 14,145 total, 14,145 trainable
            """
            total     = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return (
                f"GRURegressor("
                f"input_dim={self.input_dim}, "
                f"hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, "
                f"dropout={self.dropout}, "
                f"head_hidden_dim={self.head_hidden_dim}"
                f")\n  | params: {total:,} total, {trainable:,} trainable"
            )
 
        # ── Forward pass ──────────────────────────────────────────────────────
 
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            """
            Parameters
            ----------
            x : torch.Tensor
                Shape ``(batch, seq_len)`` or ``(batch, seq_len, input_dim)``.
                The SCATS pipeline passes ``(batch, lookback, 1)`` where
                ``lookback`` is set in ``config.ini`` (default 12 steps = 3 h).
 
            Returns
            -------
            torch.Tensor
                Shape ``(batch, 1)``. Predicted flow on the normalised scale.
                The caller (``evaluate.py`` / ``inference.py``) applies
                ``scaler.inverse_transform()`` before:
 
                  1. Computing MAE / RMSE / MAPE / R² metrics.
                  2. Feeding the travel-time formula:
                     ``speed = max(min_spd, speed_limit × (1 − flow / 1800))``
            """
            # Accept (B, T) shorthand — treat as single-feature sequence
            if x.ndim == 2:
                x = x.unsqueeze(-1)                       # (B, T, 1)
 
            # GRU forward pass
            out, _ = self.gru(x)                          # (B, T, hidden_size)
 
            # Dual temporal pooling — complementary views of the sequence
            last_feat = out[:, -1, :]                     # (B, H)  recency bias
            mean_feat = out.mean(dim=1)                   # (B, H)  global context
 
            # Concat → (B, 2H), then add residual skip from last-step features
            feat = torch.cat([last_feat, mean_feat], dim=1)        # (B, 2H)
            feat = feat + self.residual_proj(last_feat)            # residual add
 
            # Normalise and regress
            feat = self.norm(feat)                        
            return self.head(feat)                       
 
 
else:
 
    class GRURegressor(BaseTimeSeriesModel):  
        """Stub raised when PyTorch is not installed."""
 
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()
 
