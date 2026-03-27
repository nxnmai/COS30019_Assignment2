"""
Graph builder: infers road adjacency from SCATS site lat/lon + location name.

Strategy:
  1. Extract unique SCATS sites with their lat/lon and street name.
  2. Connect two sites if they share the same road name AND are within
     `proximity_km` of each other (defaults to 1.0 km).
  3. Add a second pass connecting any two sites within `close_km` (0.3 km)
     regardless of road name — catches corner intersections between roads.
  4. Output: CSV with columns  source, target, distance_km, intersections
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Road name extraction
# ---------------------------------------------------------------------------

def _extract_road_name(location: str) -> str:
    """
    Extract the primary road name from a SCATS location string.

    Examples:
      "HIGH ST N of PEEL ST" → "HIGH ST"
      "NEPEAN HWY S of OVERTON RD" → "NEPEAN HWY"
    """
    location = str(location).strip().upper()
    # Remove directional qualifiers: "N of", "S of", "NE of", etc.
    parts = re.split(r"\b[NSEW]{1,2}\s+OF\b", location, flags=re.IGNORECASE)
    if parts:
        return parts[0].strip()
    return location


# ---------------------------------------------------------------------------
# Core adjacency builder
# ---------------------------------------------------------------------------

def build_graph_edges(
    scats_df: pd.DataFrame,
    proximity_km: float = 1.0,
    close_km: float = 0.3,
) -> pd.DataFrame:
    """
    Build a directed edge list from the cleaned SCATS dataframe.

    Parameters
    ----------
    scats_df : DataFrame with columns [scats_number, nb_latitude, nb_longitude, location]
    proximity_km : max distance for same-road adjacency
    close_km : max distance for unconditional adjacency (corner intersections)

    Returns
    -------
    DataFrame with columns: source, target, distance_km, intersections
    """
    # Deduplicate: one representative row per site
    site_cols = ["scats_number", "nb_latitude", "nb_longitude", "location"]
    sites = (
        scats_df[site_cols]
        .dropna(subset=["nb_latitude", "nb_longitude"])
        .drop_duplicates(subset=["scats_number"])
        .copy()
    )
    sites["road_name"] = sites["location"].apply(_extract_road_name)
    sites = sites.reset_index(drop=True)

    ids = sites["scats_number"].tolist()
    lats = sites["nb_latitude"].to_numpy(dtype=float)
    lons = sites["nb_longitude"].to_numpy(dtype=float)
    roads = sites["road_name"].tolist()

    edges: List[Tuple[str, str, float]] = []
    seen: set = set()

    n = len(sites)
    print(f"[graph_builder] Building adjacency for {n} unique SCATS sites...")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = (ids[i], ids[j])
            if key in seen:
                continue

            dist = haversine_km(lats[i], lons[i], lats[j], lons[j])

            connect = False
            # Rule 1: same road AND within proximity
            if roads[i] and roads[j] and roads[i] == roads[j] and dist <= proximity_km:
                connect = True
            # Rule 2: very close regardless of road (corner intersections)
            if dist <= close_km:
                connect = True

            if connect:
                edges.append((ids[i], ids[j], dist))
                seen.add(key)

    if not edges:
        raise RuntimeError(
            "No edges were generated. Check proximity thresholds or data quality."
        )

    edge_df = pd.DataFrame(edges, columns=["source", "target", "distance_km"])
    edge_df["intersections"] = 1  # every edge passes through 1 controlled intersection
    edge_df = edge_df.sort_values(["source", "target"]).reset_index(drop=True)

    print(f"[graph_builder] Generated {len(edge_df)} directed edges.")
    return edge_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build SCATS road graph edge list from cleaned CSV.")
    p.add_argument(
        "--input",
        default="data/processed/scats_oct2006_clean.csv",
        help="Path to cleaned SCATS CSV (output of data_loader)",
    )
    p.add_argument(
        "--output",
        default="data/processed/graph_edges.csv",
        help="Output path for the edge list CSV",
    )
    p.add_argument(
        "--proximity-km",
        type=float,
        default=1.0,
        help="Max same-road adjacency distance in km (default: 1.0)",
    )
    p.add_argument(
        "--close-km",
        type=float,
        default=0.3,
        help="Max unconditional adjacency distance in km (default: 0.3)",
    )
    p.add_argument(
        "--raw-xls",
        default="data/raw/Scats Data October 2006.xls",
        help="Fallback raw XLS if clean CSV not found",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.exists():
        df = pd.read_csv(input_path)
    else:
        print(f"[INFO] Clean CSV not found at {input_path}. Loading from raw XLS...")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from preprocessing.data_loader import load_scats_data
        df = load_scats_data(args.raw_xls)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(input_path, index=False)

    edge_df = build_graph_edges(
        scats_df=df,
        proximity_km=args.proximity_km,
        close_km=args.close_km,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(output_path, index=False)
    print(f"[graph_builder] Edge list saved → {output_path}")
    print(edge_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
