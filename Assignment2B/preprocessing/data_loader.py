from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import xlrd

FLOW_COLUMNS: List[str] = [f"V{i:02d}" for i in range(96)]
BASE_COLUMNS: List[str] = [
    "SCATS Number",
    "Location",
    "CD_MELWAY",
    "NB_LATITUDE",
    "NB_LONGITUDE",
    "HF VicRoads Internal",
    "VR Internal Stat",
    "VR Internal Loc",
    "NB_TYPE_SURVEY",
    "Date",
]
REQUIRED_COLUMNS: List[str] = ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE", "Date", "V00", "V95"]


def _normalize_header(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _find_header_row(sheet: xlrd.sheet.Sheet, required_columns: Iterable[str]) -> int:
    required = set(required_columns)
    for row_idx in range(min(sheet.nrows, 15)):
        row_headers = {_normalize_header(sheet.cell_value(row_idx, c)) for c in range(sheet.ncols)}
        if required.issubset(row_headers):
            return row_idx
    raise ValueError("Could not find header row in XLS sheet. Expected SCATS/Date/V00/V95 columns.")


def _to_float(value: object) -> float:
    if value in ("", None):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value: object) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_datetime(value: object, datemode: int) -> pd.Timestamp:
    if value in ("", None):
        return pd.NaT
    if isinstance(value, (float, int)):
        try:
            dt = xlrd.xldate.xldate_as_datetime(float(value), datemode)
            return pd.Timestamp(dt)
        except (TypeError, ValueError, xlrd.xldate.XLDateError):
            return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(parsed) if not pd.isna(parsed) else pd.NaT


def _infer_direction(location: str) -> str:
    # Extracts cardinal hint from strings like "... N of ..." / "... SW of ..."
    match = re.search(r"\b([NSEW]{1,2})\s+of\b", location, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNK"


def _format_scats_number(value: object) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    if re.fullmatch(r"\d+(\.0+)?", text):
        return f"{int(float(text)):04d}"
    return text


def load_scats_data(
    file_path: str | Path,
    sheet_name: str = "Data",
    keep_raw_columns: bool = False,
) -> pd.DataFrame:
    """
    Load SCATS October 2006 workbook and return a cleaned dataframe.

    Output columns (default):
      scats_number, movement_id, location, direction_hint, date, date_time,
      nb_latitude, nb_longitude, cd_melway, hf_vicroads_internal, vr_internal_stat,
      vr_internal_loc, nb_type_survey, V00..V95
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    workbook = xlrd.open_workbook(file_path.as_posix())
    try:
        sheet = workbook.sheet_by_name(sheet_name)
    except xlrd.biffh.XLRDError as exc:
        raise ValueError(f"Sheet '{sheet_name}' not found in {file_path.name}") from exc

    header_row = _find_header_row(sheet, REQUIRED_COLUMNS)
    headers = [_normalize_header(sheet.cell_value(header_row, c)) for c in range(sheet.ncols)]
    header_to_idx = {name: idx for idx, name in enumerate(headers) if name}

    missing = [col for col in REQUIRED_COLUMNS if col not in header_to_idx]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected_columns = [col for col in BASE_COLUMNS if col in header_to_idx] + [
        col for col in FLOW_COLUMNS if col in header_to_idx
    ]

    records: List[dict] = []
    for row_idx in range(header_row + 1, sheet.nrows):
        scats_raw = sheet.cell_value(row_idx, header_to_idx["SCATS Number"])
        if str(scats_raw).strip() == "":
            continue

        raw_row = {col: sheet.cell_value(row_idx, header_to_idx[col]) for col in selected_columns}
        date_time = _to_datetime(raw_row.get("Date"), workbook.datemode)
        if pd.isna(date_time):
            continue

        cleaned = {
            "scats_number": _format_scats_number(raw_row.get("SCATS Number")),
            "location": str(raw_row.get("Location", "")).strip(),
            "direction_hint": _infer_direction(str(raw_row.get("Location", "")).strip()),
            "date_time": date_time,
            "date": date_time.normalize(),
            "nb_latitude": _to_float(raw_row.get("NB_LATITUDE")),
            "nb_longitude": _to_float(raw_row.get("NB_LONGITUDE")),
            "cd_melway": str(raw_row.get("CD_MELWAY", "")).strip(),
            "hf_vicroads_internal": _to_int(raw_row.get("HF VicRoads Internal")),
            "vr_internal_stat": _to_int(raw_row.get("VR Internal Stat")),
            "vr_internal_loc": _to_int(raw_row.get("VR Internal Loc")),
            "nb_type_survey": _to_int(raw_row.get("NB_TYPE_SURVEY")),
        }

        if cleaned["vr_internal_loc"] is not None:
            cleaned["movement_id"] = f"{cleaned['scats_number']}_{cleaned['vr_internal_loc']:02d}"
        else:
            safe_location = re.sub(r"[^A-Za-z0-9]+", "_", cleaned["location"]).strip("_").lower()
            cleaned["movement_id"] = f"{cleaned['scats_number']}_{safe_location}"

        for flow_col in FLOW_COLUMNS:
            cleaned[flow_col] = _to_float(raw_row.get(flow_col))

        if keep_raw_columns:
            cleaned.update(raw_row)

        records.append(cleaned)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No usable rows were parsed from {file_path}")

    output_columns = [
        "scats_number",
        "movement_id",
        "location",
        "direction_hint",
        "date",
        "date_time",
        "nb_latitude",
        "nb_longitude",
        "cd_melway",
        "hf_vicroads_internal",
        "vr_internal_stat",
        "vr_internal_loc",
        "nb_type_survey",
        *FLOW_COLUMNS,
    ]
    df = df[output_columns].sort_values(["scats_number", "movement_id", "date"]).reset_index(drop=True)
    return df


def split_by_site(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        site_id: site_df.sort_values(["movement_id", "date"]).reset_index(drop=True)
        for site_id, site_df in dataframe.groupby("scats_number", sort=True)
    }


def save_per_site_csv(dataframe: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for site_id, site_df in split_by_site(dataframe).items():
        site_df.to_csv(output_path / f"site_{site_id}.csv", index=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse SCATS XLS into a clean dataframe.")
    parser.add_argument(
        "--input",
        default="data/raw/Scats Data October 2006.xls",
        help="Path to Scats Data October 2006.xls",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/scats_oct2006_clean.csv",
        help="Output CSV path for consolidated cleaned data",
    )
    parser.add_argument(
        "--per-site-dir",
        default="data/processed/sites",
        help="Directory for per-site CSV exports",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = load_scats_data(args.input)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    save_per_site_csv(df, args.per_site_dir)

    print(f"Parsed rows: {len(df)}")
    print(f"Unique SCATS sites: {df['scats_number'].nunique()}")
    print(f"Unique movement streams: {df['movement_id'].nunique()}")
    print(f"Saved consolidated CSV: {output_csv}")
    print(f"Saved per-site CSVs: {Path(args.per_site_dir)}")


if __name__ == "__main__":
    main()
