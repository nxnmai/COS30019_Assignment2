from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing.data_loader import FLOW_COLUMNS, load_scats_data


@dataclass
class DatasetSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    scaler: MinMaxScaler
    metadata: Dict[str, object]


def _make_windows(sequence: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    if sequence.ndim != 1:
        raise ValueError("Sequence for windowing must be 1D")
    if len(sequence) <= lookback:
        return np.empty((0, lookback, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    x, y = [], []
    for end_idx in range(lookback, len(sequence)):
        start_idx = end_idx - lookback
        x.append(sequence[start_idx:end_idx])
        y.append(sequence[end_idx])
    x_arr = np.asarray(x, dtype=np.float32).reshape(-1, lookback, 1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return x_arr, y_arr


def _daily_matrix_for_stream(
    stream_df: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    fill_missing_days: bool,
) -> Tuple[pd.DataFrame, int]:
    daily = (
        stream_df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")[FLOW_COLUMNS]
        .reindex(all_dates)
    )
    available_days = int(daily.notna().any(axis=1).sum())

    if fill_missing_days:
        daily = daily.astype(float).interpolate(axis=0, limit_direction="both").ffill().bfill()
    else:
        daily = daily.fillna(0.0)

    return daily, available_days


def _fit_scaler(train_sequences: List[np.ndarray]) -> MinMaxScaler:
    if not train_sequences:
        raise ValueError("No training sequences available for scaler fitting.")
    train_values = np.concatenate(train_sequences).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(train_values)
    return scaler


def prepare_datasets(
    dataframe: pd.DataFrame,
    lookback: int = 12,
    train_days: int = 21,
    val_days: int = 5,
    test_days: int = 5,
    min_available_days: int = 21,
    fill_missing_days: bool = True,
    scaler_output_path: str | Path = "data/processed/scaler.pkl",
    dataset_output_path: str | Path = "data/processed/datasets.npz",
    metadata_output_path: str | Path = "data/processed/dataset_metadata.json",
) -> DatasetSplits:
    """
    Convert cleaned SCATS rows into lookback-window model datasets.

    Train/Val/Test split is day-based: 21 / 5 / 5.
    Scaler is fit strictly on train data and saved for inference inverse-transform.
    """

    if "date" not in dataframe.columns:
        raise ValueError("Input dataframe must contain a 'date' column.")

    df = dataframe.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "movement_id"]).reset_index(drop=True)
    for col in FLOW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    all_dates = pd.DatetimeIndex(sorted(df["date"].dropna().unique()))
    required_days = train_days + val_days + test_days
    if len(all_dates) < required_days:
        raise ValueError(
            f"Not enough unique dates ({len(all_dates)}) for {train_days}/{val_days}/{test_days} split."
        )

    train_date_idx = slice(0, train_days)
    val_date_idx = slice(train_days, train_days + val_days)
    test_date_idx = slice(train_days + val_days, train_days + val_days + test_days)

    split_sequences_raw: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    skipped_streams: Dict[str, int] = {}
    used_streams: List[str] = []

    for stream_id, stream_df in df.groupby("movement_id", sort=True):
        daily_matrix, available_days = _daily_matrix_for_stream(stream_df, all_dates, fill_missing_days)
        if available_days < min_available_days:
            skipped_streams[str(stream_id)] = available_days
            continue

        values = daily_matrix.to_numpy(dtype=np.float32)
        split_sequences_raw["train"].append(values[train_date_idx].reshape(-1))
        split_sequences_raw["val"].append(values[val_date_idx].reshape(-1))
        split_sequences_raw["test"].append(values[test_date_idx].reshape(-1))
        used_streams.append(str(stream_id))

    if not split_sequences_raw["train"]:
        raise ValueError("No streams passed filtering. Lower min_available_days or inspect raw data quality.")

    scaler = _fit_scaler(split_sequences_raw["train"])

    split_xy: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    for split_name in ("train", "val", "test"):
        for seq in split_sequences_raw[split_name]:
            scaled_seq = scaler.transform(seq.reshape(-1, 1)).reshape(-1)
            x_part, y_part = _make_windows(scaled_seq, lookback=lookback)
            if len(x_part) == 0:
                continue
            split_xy[split_name][0].append(x_part)
            split_xy[split_name][1].append(y_part)

    def _stack(parts: List[np.ndarray], shape_tail: Tuple[int, ...]) -> np.ndarray:
        if not parts:
            return np.empty((0, *shape_tail), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    train_x = _stack(split_xy["train"][0], (lookback, 1))
    train_y = _stack(split_xy["train"][1], (1,))
    val_x = _stack(split_xy["val"][0], (lookback, 1))
    val_y = _stack(split_xy["val"][1], (1,))
    test_x = _stack(split_xy["test"][0], (lookback, 1))
    test_y = _stack(split_xy["test"][1], (1,))

    scaler_output_path = Path(scaler_output_path)
    dataset_output_path = Path(dataset_output_path)
    metadata_output_path = Path(metadata_output_path)
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, scaler_output_path)
    np.savez_compressed(
        dataset_output_path,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )

    metadata = {
        "lookback": lookback,
        "flow_columns": FLOW_COLUMNS,
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "used_stream_count": len(used_streams),
        "used_streams": used_streams,
        "skipped_streams": skipped_streams,
        "date_range": [str(all_dates.min().date()), str(all_dates.max().date())],
        "split_dates": {
            "train": [str(all_dates[train_date_idx].min().date()), str(all_dates[train_date_idx].max().date())],
            "val": [str(all_dates[val_date_idx].min().date()), str(all_dates[val_date_idx].max().date())],
            "test": [str(all_dates[test_date_idx].min().date()), str(all_dates[test_date_idx].max().date())],
        },
        "sample_counts": {
            "train": int(len(train_x)),
            "val": int(len(val_x)),
            "test": int(len(test_x)),
        },
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return DatasetSplits(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        scaler=scaler,
        metadata=metadata,
    )


def inverse_transform(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(values.shape)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build lookback windows + scaler from SCATS data.")
    parser.add_argument("--input-xls", default="data/raw/Scats Data October 2006.xls")
    parser.add_argument("--input-clean-csv", default="data/processed/scats_oct2006_clean.csv")
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--train-days", type=int, default=21)
    parser.add_argument("--val-days", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=5)
    parser.add_argument("--min-available-days", type=int, default=21)
    parser.add_argument("--scaler-out", default="data/processed/scaler.pkl")
    parser.add_argument("--dataset-out", default="data/processed/datasets.npz")
    parser.add_argument("--metadata-out", default="data/processed/dataset_metadata.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    clean_csv_path = Path(args.input_clean_csv)
    if clean_csv_path.exists():
        df = pd.read_csv(clean_csv_path, parse_dates=["date", "date_time"])
    else:
        df = load_scats_data(args.input_xls)
        clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(clean_csv_path, index=False)

    splits = prepare_datasets(
        dataframe=df,
        lookback=args.lookback,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        min_available_days=args.min_available_days,
        scaler_output_path=args.scaler_out,
        dataset_output_path=args.dataset_out,
        metadata_output_path=args.metadata_out,
    )

    print("Prepared datasets:")
    print(f"  train: {splits.train_x.shape} -> {splits.train_y.shape}")
    print(f"  val:   {splits.val_x.shape} -> {splits.val_y.shape}")
    print(f"  test:  {splits.test_x.shape} -> {splits.test_y.shape}")
    print(f"  used streams: {splits.metadata['used_stream_count']}")


if __name__ == "__main__":
    main()
