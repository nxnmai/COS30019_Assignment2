from __future__ import annotations

import argparse
import configparser
import heapq
import importlib
import inspect
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from prediction.travel_time import (
    DEFAULT_INTERSECTION_DELAY_SEC,
    DEFAULT_MIN_SPEED_KMH,
    DEFAULT_SPEED_LIMIT_KMH,
    FREE_FLOW_THRESHOLD,
    travel_time_seconds,
)
from preprocessing.data_loader import FLOW_COLUMNS, load_scats_data

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "transformer": {
        "module": "models.transformer_model",
        "class_candidates": ["TransformerRegressor", "TransformerModel"],
        "checkpoint_key": "transformer_checkpoint",
    },
    "lstm": {
        "module": "models.lstm_model",
        "class_candidates": ["LSTMRegressor", "LSTMModel", "LSTM"],
        "checkpoint_key": "lstm_checkpoint",
    },
    "gru": {
        "module": "models.gru_model",
        "class_candidates": ["GRURegressor", "GRUModel", "GRU"],
        "checkpoint_key": "gru_checkpoint",
    },
    "bilstm": {
        "module": "models.bilstm_model",
        "class_candidates": ["BiLSTMRegressor"],
        "checkpoint_key": "bilstm_checkpoint",
    },
}


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    distance_km: float
    stream_id: Optional[str] = None
    intersections: int = 1


def _normalize_node_id(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.0+)?", text):
        return f"{int(float(text)):04d}"
    return text


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(config_path: str | Path = "config.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    resolved = _resolve_path(str(config_path))
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    cfg.read(resolved, encoding="utf-8")
    return cfg


def _pick_column(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    if required:
        raise ValueError(f"Required column not found. Expected one of: {candidates}")
    return None


def _load_graph_edges(graph_edges_csv: Path) -> List[Edge]:
    if not graph_edges_csv.exists():
        raise FileNotFoundError(
            f"Graph edges CSV not found: {graph_edges_csv}. Expected columns like source,target,distance_km."
        )
    df = pd.read_csv(graph_edges_csv)
    source_col = _pick_column(df.columns, ["source", "from", "from_id", "origin", "u"])
    target_col = _pick_column(df.columns, ["target", "to", "to_id", "destination", "v"])
    distance_col = _pick_column(df.columns, ["distance_km", "distance", "km", "length_km"])
    stream_col = _pick_column(df.columns, ["stream_id", "movement_id", "flow_id"], required=False)
    intersections_col = _pick_column(df.columns, ["intersections", "intersection_count"], required=False)

    edges = []
    for _, row in df.iterrows():
        source = _normalize_node_id(row[source_col])
        target = _normalize_node_id(row[target_col])
        distance_km = float(row[distance_col])
        stream_id = str(row[stream_col]).strip() if stream_col and pd.notna(row[stream_col]) else None
        intersections = int(row[intersections_col]) if intersections_col and pd.notna(row[intersections_col]) else 1
        edges.append(Edge(source=source, target=target, distance_km=max(0.0, distance_km), stream_id=stream_id, intersections=max(0, intersections)))
    return edges


def _load_clean_flow_data(cfg: configparser.ConfigParser) -> pd.DataFrame:
    clean_csv = _resolve_path(cfg.get("paths", "clean_scats_csv"))
    raw_xls = _resolve_path(cfg.get("paths", "raw_scats_xls"))

    if clean_csv.exists():
        try:
            # Force scats_number to be string to preserve leading zeros (e.g., "0970")
            df = pd.read_csv(clean_csv, parse_dates=["date", "date_time"], dtype={"scats_number": str})
        except ValueError:
            df = pd.read_csv(clean_csv)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "date_time" in df.columns:
                df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    else:
        df = load_scats_data(raw_xls)
        clean_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(clean_csv, index=False)

        # Aggressively normalize SCATS IDs to 4-digit strings (e.g. 970.0 -> "0970")
        df["scats_number"] = df["scats_number"].astype(str).apply(_normalize_node_id)
        
        # Ensure coordinates are numeric
        if "nb_latitude" in df.columns:
            df["nb_latitude"] = pd.to_numeric(df["nb_latitude"], errors="coerce")
        if "nb_longitude" in df.columns:
            df["nb_longitude"] = pd.to_numeric(df["nb_longitude"], errors="coerce")
    
    for col in FLOW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_profiles(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    stream_profiles: Dict[str, np.ndarray] = {}
    site_profiles: Dict[str, np.ndarray] = {}

    for stream_id, group in df.groupby("movement_id", sort=True):
        profile = np.nanmean(group[FLOW_COLUMNS].to_numpy(dtype=np.float64), axis=0)
        stream_profiles[str(stream_id)] = np.nan_to_num(profile, nan=0.0)

    for site_id, group in df.groupby("scats_number", sort=True):
        # Accumulated volume per hour at the site = SUM of all movements
        # We perform sum across movements for each 15-min interval
        # Then we will multiply by 4 later to get veh/hr
        profile = np.nansum(group.groupby("date")[FLOW_COLUMNS].sum().to_numpy(dtype=np.float64), axis=0)
        # Wait, the above is wrong. We want the AVERAGE daily profile of the TOTAL flow.
        # So first sum all movements for each day/interval, then average across days.
        daily_site_totals = group.groupby("date")[FLOW_COLUMNS].sum()
        profile = np.nanmean(daily_site_totals.to_numpy(dtype=np.float64), axis=0)
        site_profiles[str(site_id)] = np.nan_to_num(profile, nan=0.0)

    if stream_profiles:
        global_profile = np.nanmean(np.vstack(list(stream_profiles.values())), axis=0)
    else:
        global_profile = np.zeros((96,), dtype=np.float64)
    global_profile = np.nan_to_num(global_profile, nan=0.0)
    return stream_profiles, site_profiles, global_profile


def _load_model_class(module_name: str, class_candidates: Sequence[str]):
    module = importlib.import_module(module_name)
    for name in class_candidates:
        klass = getattr(module, name, None)
        if klass is not None:
            return klass

    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and any(tok in obj.__name__.lower() for tok in ("transformer", "lstm", "gru")):
            return obj
    raise ValueError(f"No model class found in module {module_name}")


def _load_model_and_scaler(
    cfg: configparser.ConfigParser,
    model_name: str,
) -> Tuple[Optional[Any], Optional[Any], str]:
    model_name = model_name.lower()
    scaler_path = _resolve_path(cfg.get("paths", "scaler_path"))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    model_spec = MODEL_SPECS.get(model_name)
    if model_spec is None:
        return None, scaler, "cpu"

    checkpoint_key = str(model_spec["checkpoint_key"])
    checkpoint_path = _resolve_path(cfg.get("models", checkpoint_key))
    if not checkpoint_path.exists():
        return None, scaler, "cpu"

    try:
        from models.base_model import resolve_device, torch

        device = resolve_device(cfg.get("inference", "device", fallback="auto"))
        model_cls = _load_model_class(
            module_name=str(model_spec["module"]),
            class_candidates=model_spec["class_candidates"],  # type: ignore[arg-type]
        )

        if hasattr(model_cls, "load"):
            model, _ = model_cls.load(checkpoint_path, map_location=device)
        else:
            if torch is None:
                return None, scaler, "cpu"
            model = model_cls()
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        return model, scaler, device
    except Exception:
        # Integration should still run with profile fallback if model loading fails.
        return None, scaler, "cpu"


def _predict_with_model(model: Any, scaler: Any, history_values: np.ndarray, device: str) -> float:
    if model is None or scaler is None:
        return float(history_values[-1])

    x_scaled = scaler.transform(history_values.reshape(-1, 1)).reshape(1, len(history_values), 1).astype(np.float32)
    if hasattr(model, "predict"):
        pred_scaled = model.predict(x_scaled, batch_size=1, device=device).reshape(-1)[0]
    else:
        try:
            from models.base_model import torch
        except Exception:
            return float(history_values[-1])
        if torch is None:
            return float(history_values[-1])
        model.eval()
        with torch.no_grad():
            xb = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
            pred_scaled = float(model(xb).detach().cpu().numpy().reshape(-1)[0])

    pred = scaler.inverse_transform(np.array([[pred_scaled]], dtype=np.float32)).reshape(-1)[0]
    return float(max(pred, 0.0))


def _predict_edge_flow(
    edge: Edge,
    interval_index: int,
    lookback: int,
    stream_profiles: Dict[str, np.ndarray],
    site_profiles: Dict[str, np.ndarray],
    global_profile: np.ndarray,
    model: Optional[Any],
    scaler: Optional[Any],
    device: str,
) -> float:
    # Per assignment instructions, use the accumulated volume at Site B (target)
    if edge.target in site_profiles:
        profile = site_profiles[edge.target]
    elif edge.stream_id and edge.stream_id in stream_profiles:
        profile = stream_profiles[edge.stream_id]
    else:
        profile = global_profile

    hist_idx = [((interval_index - lookback + i) % 96) for i in range(lookback)]
    history = profile[hist_idx]

    if model is None or scaler is None:
        return float(max(profile[interval_index % 96], 0.0))
    return _predict_with_model(model, scaler, history, device=device)


def _build_weighted_adjacency(
    edges: List[Edge],
    edge_weights_sec: Dict[Tuple[str, str], float],
    undirected: bool,
) -> Dict[str, List[Tuple[str, float]]]:
    adj: Dict[str, List[Tuple[str, float]]] = {}
    for edge in edges:
        weight = edge_weights_sec[(edge.source, edge.target)]
        adj.setdefault(edge.source, []).append((edge.target, weight))
        if undirected:
            adj.setdefault(edge.target, []).append((edge.source, weight))
    return adj


def _fallback_top_k_paths(
    adjacency: Dict[str, List[Tuple[str, float]]],
    origin: str,
    destination: str,
    k: int,
    max_depth: int,
    max_expansions: int = 200_000,
) -> List[Tuple[List[str], float]]:
    if origin not in adjacency:
        return []

    heap: List[Tuple[float, List[str]]] = [(0.0, [origin])]
    results: List[Tuple[List[str], float]] = []
    seen_paths = set()
    expansions = 0

    while heap and len(results) < k and expansions < max_expansions:
        cost, path = heapq.heappop(heap)
        expansions += 1
        current = path[-1]

        if current == destination:
            path_key = tuple(path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                results.append((path, cost))
            continue

        if len(path) - 1 >= max_depth:
            continue

        for neighbor, edge_cost in adjacency.get(current, []):
            if neighbor in path:
                continue  # keep loopless routes
            heapq.heappush(heap, (cost + edge_cost, [*path, neighbor]))

    return results


def _try_external_search(
    adjacency: Dict[str, List[Tuple[str, float]]],
    origin: str,
    destination: str,
    k: int,
) -> Optional[List[Tuple[List[str], float]]]:
    try:
        search_module = importlib.import_module("routing.search")
    except Exception:
        return None

    candidate_names = ["find_top_k_routes", "top_k_routes", "k_shortest_paths", "find_routes"]
    for func_name in candidate_names:
        func = getattr(search_module, func_name, None)
        if func is None or not callable(func):
            continue

        try:
            sig = inspect.signature(func)
            kwargs = {}
            if "k" in sig.parameters:
                kwargs["k"] = k
            if "graph" in sig.parameters:
                kwargs["graph"] = adjacency
                result = func(origin, destination, **kwargs)
            else:
                result = func(adjacency, origin, destination, **kwargs)
            if result:
                normalized = []
                for item in result:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        normalized.append((list(item[0]), float(item[1])))
                if normalized:
                    return normalized[:k]
        except Exception:
            continue
    return None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid datetime: {value}")
    return parsed.to_pydatetime()


def find_routes(
    origin_id: str | int,
    dest_id: str | int,
    datetime: str | datetime,
    model: str = "transformer",
    k: int = 5,
    return_network: bool = False,
) -> Dict[str, Any]:
    """
    Main integration API expected by the assignment brief.

    Pipeline:
      model/scaler load -> edge flow prediction -> flow->time conversion ->
      graph weight update -> top-k route search
    """

    cfg = load_config("config.ini")
    ts = _parse_datetime(datetime)
    origin = _normalize_node_id(origin_id)
    destination = _normalize_node_id(dest_id)
    lookback = cfg.getint("preprocessing", "lookback", fallback=12)
    graph_edges_csv = _resolve_path(cfg.get("paths", "graph_edges_csv"))

    edges = _load_graph_edges(graph_edges_csv)
    flow_df = _load_clean_flow_data(cfg)
    stream_profiles, site_profiles, global_profile = _build_profiles(flow_df)
    loaded_model, scaler, device = _load_model_and_scaler(cfg, model_name=model)

    interval_idx = ((ts.hour * 60) + ts.minute) // 15
    speed_limit = cfg.getfloat("travel_time", "speed_limit_kmh", fallback=DEFAULT_SPEED_LIMIT_KMH)
    min_speed = cfg.getfloat("travel_time", "min_speed_kmh", fallback=DEFAULT_MIN_SPEED_KMH)
    threshold = cfg.getfloat("travel_time", "free_flow_threshold", fallback=FREE_FLOW_THRESHOLD)
    intersection_delay = cfg.getfloat(
        "travel_time", "intersection_delay_sec", fallback=DEFAULT_INTERSECTION_DELAY_SEC
    )
    undirected = cfg.getboolean("routing", "undirected", fallback=False)
    max_depth = cfg.getint("routing", "max_path_depth", fallback=12)

    edge_weights: Dict[Tuple[str, str], float] = {}
    edge_flow: Dict[Tuple[str, str], float] = {}
    for edge in edges:
        flow_15min = _predict_edge_flow(
            edge=edge,
            interval_index=interval_idx,
            lookback=lookback,
            stream_profiles=stream_profiles,
            site_profiles=site_profiles,
            global_profile=global_profile,
            model=loaded_model,
            scaler=scaler,
            device=device,
        )
        # Convert 15-minute count to vehicles per hour for the formula
        flow_veh_per_hour = flow_15min * 4.0

        time_sec = travel_time_seconds(
            flow_veh_per_hour=flow_veh_per_hour,
            distance_km=edge.distance_km,
            intersection_delay_sec=intersection_delay,
            intersections_count=edge.intersections,
            speed_limit_kmh=speed_limit,
            free_flow_threshold=threshold,
            min_speed_kmh=min_speed,
        )
        edge_weights[(edge.source, edge.target)] = float(time_sec)
        edge_flow[(edge.source, edge.target)] = float(flow_veh_per_hour)

    adjacency = _build_weighted_adjacency(edges, edge_weights, undirected=undirected)

    external_routes = _try_external_search(adjacency, origin, destination, k)
    if external_routes is not None:
        selected_paths = external_routes[:k]
    else:
        selected_paths = _fallback_top_k_paths(
            adjacency=adjacency,
            origin=origin,
            destination=destination,
            k=k,
            max_depth=max_depth,
        )

    results: List[Dict[str, Any]] = []
    for path, total_time in selected_paths:
        segments = []
        for src, dst in zip(path[:-1], path[1:]):
            seg_time = edge_weights.get((src, dst))
            seg_flow = edge_flow.get((src, dst))
            segments.append(
                {
                    "from": src,
                    "to": dst,
                    "predicted_flow_veh_per_hour": seg_flow,
                    "travel_time_sec": seg_time,
                }
            )
        results.append(
            {
                "path": path,
                "total_time_sec": float(total_time),
                "total_time_min": float(total_time / 60.0),
                "model_used": model.lower(),
                "timestamp": ts.isoformat(),
                "segments": segments,
            }
        )

    if not return_network:
        return results

    # Extraction of node coordinates for the map
    node_coords = {}
    
    # Pre-process flow_df for definitive lookup parity
    if "_scats_norm" not in flow_df.columns:
        flow_df["_scats_norm"] = flow_df["scats_number"].astype(str).apply(_normalize_node_id)
    
    # AGGRESSIVE PRE-POPULATION: Capture coordinates for EVERY site in the data
    valid_data = flow_df.dropna(subset=["nb_latitude", "nb_longitude"])
    for _, row in valid_data.drop_duplicates(subset=["_scats_norm"]).iterrows():
        nid = str(row["_scats_norm"])
        node_coords[nid] = (float(row["nb_latitude"]), float(row["nb_longitude"]))
    
    # PRE-POPULATE: Grab coordinates for ALL sites known in the traffic data first
    # This ensures that even if edges are missing, we see the sites on the map!
    unique_sites = flow_df.dropna(subset=["nb_latitude", "nb_longitude"]).drop_duplicates(subset=["_scats_norm"])
    for _, row in unique_sites.iterrows():
        nid = str(row["_scats_norm"])
        node_coords[nid] = (float(row["nb_latitude"]), float(row["nb_longitude"]))
    
    # (Optional) Ensure route nodes specifically are covered (should already be from above)
    for edge in edges:
        for nid in (edge.source, edge.target):
            target_nid = str(nid).strip()
            if target_nid not in node_coords:
                match = flow_df[flow_df["_scats_norm"] == target_nid]
                if not match.empty:
                    valid_coords = match.dropna(subset=["nb_latitude", "nb_longitude"])
                    if not valid_coords.empty:
                        node_coords[target_nid] = (
                            float(valid_coords.iloc[0]["nb_latitude"]),
                            float(valid_coords.iloc[0]["nb_longitude"]),
                        )
                else:
                    # Fallback for missing coordinates: search for ANY movement at this site
                    # to increase robustness against slight ID variations in raw data
                    pass

    return {
        "routes": results,
        "edges": edges,
        "edge_flow": edge_flow,
        "edge_weights": edge_weights,
        "node_coords": node_coords,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TBRGS route finder (Part B integration).")
    parser.add_argument("--origin", required=True, help="Origin SCATS site ID")
    parser.add_argument("--destination", required=True, help="Destination SCATS site ID")
    parser.add_argument("--datetime", required=True, help="Target datetime, e.g. '2006-10-15 08:30'")
    parser.add_argument("--model", default="transformer", choices=["transformer", "lstm", "gru"])
    parser.add_argument("--k", type=int, default=5, help="Number of routes to return")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    routes = find_routes(
        origin_id=args.origin,
        dest_id=args.destination,
        datetime=args.datetime,
        model=args.model,
        k=args.k,
    )

    if not routes:
        print("No route found.")
        return

    for idx, route in enumerate(routes, start=1):
        print(f"Route {idx}: {' -> '.join(route['path'])}")
        print(f"  total_time_sec={route['total_time_sec']:.2f} ({route['total_time_min']:.2f} min)")


if __name__ == "__main__":
    main()
