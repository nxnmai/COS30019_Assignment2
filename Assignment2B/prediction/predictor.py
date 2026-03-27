"""
Unified predictor interface.

Wraps model loading, scaler handling, and profile-based fallback into a single
reusable object. Used by the Streamlit GUI and any script that needs a clean
predict_flow(site_id, datetime) API.
"""
from __future__ import annotations

import configparser
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(cfg: configparser.ConfigParser, key: str) -> Path:
    raw = cfg.get("paths", key)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class TrafficPredictor:
    """
    High-level interface for traffic flow prediction.

    Usage
    -----
    predictor = TrafficPredictor(model_name="lstm")
    flow = predictor.predict_flow(site_id="2000", dt=datetime(2006, 10, 15, 8, 30))
    """

    def __init__(
        self,
        model_name: str = "lstm",
        config_path: str | Path = "config.ini",
    ) -> None:
        self.model_name = model_name.lower()
        cfg = configparser.ConfigParser()
        resolved_cfg = Path(config_path)
        if not resolved_cfg.is_absolute():
            resolved_cfg = (PROJECT_ROOT / resolved_cfg).resolve()
        cfg.read(resolved_cfg)

        self._lookback = cfg.getint("preprocessing", "lookback", fallback=12)
        self._device = cfg.get("inference", "device", fallback="auto")

        # Load scaler
        scaler_path = _resolve(cfg, "scaler_path")
        self._scaler = self._load_scaler(scaler_path)

        # Load model
        self._model = self._load_model(cfg)

        # Build site profiles (fallback when model is unavailable)
        clean_csv = _resolve(cfg, "clean_scats_csv")
        raw_xls = _resolve(cfg, "raw_scats_xls")
        self._stream_profiles, self._site_profiles, self._global_profile = (
            self._build_profiles(clean_csv, raw_xls)
        )

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_scaler(self, scaler_path: Path):
        try:
            import joblib
            if scaler_path.exists():
                return joblib.load(scaler_path)
        except Exception:
            pass
        return None

    def _load_model(self, cfg: configparser.ConfigParser):
        from main import MODEL_SPECS, _load_model_class

        spec = MODEL_SPECS.get(self.model_name)
        if spec is None:
            return None
        checkpoint_key = str(spec["checkpoint_key"])
        try:
            checkpoint_path = _resolve(cfg, checkpoint_key)
        except Exception:
            return None
        if not checkpoint_path.exists():
            return None
        try:
            from models.base_model import resolve_device
            device = resolve_device(self._device)
            cls = _load_model_class(
                module_name=str(spec["module"]),
                class_candidates=spec["class_candidates"],  # type: ignore
            )
            if hasattr(cls, "load"):
                model, _ = cls.load(checkpoint_path, map_location=device)
                return model
        except Exception:
            pass
        return None

    def _build_profiles(
        self, clean_csv: Path, raw_xls: Path
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        from preprocessing.data_loader import FLOW_COLUMNS

        if clean_csv.exists():
            try:
                df = pd.read_csv(clean_csv)
            except Exception:
                return {}, {}, np.zeros(96)
        elif raw_xls.exists():
            from preprocessing.data_loader import load_scats_data
            df = load_scats_data(raw_xls)
        else:
            return {}, {}, np.zeros(96)

        stream_profiles: Dict[str, np.ndarray] = {}
        site_profiles: Dict[str, np.ndarray] = {}

        for stream_id, grp in df.groupby("movement_id", sort=True):
            vals = grp[FLOW_COLUMNS].to_numpy(dtype=float)
            stream_profiles[str(stream_id)] = np.nan_to_num(
                np.nanmean(vals, axis=0), nan=0.0
            )

        for site_id, grp in df.groupby("scats_number", sort=True):
            vals = grp[FLOW_COLUMNS].to_numpy(dtype=float)
            site_profiles[str(site_id)] = np.nan_to_num(
                np.nanmean(vals, axis=0), nan=0.0
            )

        if stream_profiles:
            global_profile = np.nanmean(
                np.vstack(list(stream_profiles.values())), axis=0
            )
        else:
            global_profile = np.zeros(96)

        return stream_profiles, site_profiles, np.nan_to_num(global_profile, nan=0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_flow(self, site_id: str, dt: Any) -> float:
        """
        Predict traffic flow (vehicles/hour) for `site_id` at datetime `dt`.

        Falls back to historical mean profile when model is not loaded.
        """
        import pandas as _pd

        ts = _pd.Timestamp(dt)
        interval_idx = (ts.hour * 60 + ts.minute) // 15

        # Choose profile
        if site_id in self._site_profiles:
            profile = self._site_profiles[site_id]
        else:
            profile = self._global_profile

        # History window
        hist_idx = [((interval_idx - self._lookback + i) % 96) for i in range(self._lookback)]
        history = profile[hist_idx]

        if self._model is None or self._scaler is None:
            return max(float(profile[interval_idx % 96]), 0.0)

        try:
            x_scaled = self._scaler.transform(
                history.reshape(-1, 1)
            ).reshape(1, self._lookback, 1).astype(np.float32)

            if hasattr(self._model, "predict"):
                pred_scaled = self._model.predict(x_scaled, batch_size=1, device=self._device).reshape(-1)[0]
            else:
                from models.base_model import resolve_device, torch
                device = resolve_device(self._device)
                self._model.eval()
                with torch.no_grad():
                    xb = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
                    pred_scaled = float(self._model(xb).detach().cpu().numpy().reshape(-1)[0])

            pred = self._scaler.inverse_transform(
                np.array([[pred_scaled]], dtype=np.float32)
            ).reshape(-1)[0]
            return max(float(pred), 0.0)
        except Exception:
            return max(float(profile[interval_idx % 96]), 0.0)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def scaler_loaded(self) -> bool:
        return self._scaler is not None
