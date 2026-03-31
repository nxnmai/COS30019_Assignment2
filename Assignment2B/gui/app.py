"""
TBRGS — Traffic-Based Route Guidance System
Streamlit GUI: 4 tabs
  1. Route Planner    — find top-5 routes between SCATS sites
  2. Train Models     — train LSTM / GRU / BiLSTM / Transformer
  3. Evaluate         — compare model metrics & view charts
  4. Settings         — edit config.ini parameters
"""
from __future__ import annotations

import configparser
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import map_utils

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TBRGS — Traffic Route Guidance",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state initialization ──────────────────────────────────────────────
if "route_result" not in st.session_state:
    st.session_state.route_result = None
if "route_dt" not in st.session_state:
    st.session_state.route_dt = None
if "calibrated_coords" not in st.session_state:
    st.session_state.calibrated_coords = None

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #e6ddb0;
        --paper: #efe7c0;
        --olive: #8a9777;
        --olive-dark: #667154;
        --olive-soft: #9aa588;
        --tan: #c99a57;
        --tan-dark: #a87839;
        --text: #5a5037;
        --muted: #7e7357;
        --blue-accent: #5a7f91;
        --input-bg: #f3ecc7;
        --dash: #8c9776;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle, rgba(120,110,85,0.20) 1.5px, transparent 1.6px) 0 0 / 24px 24px,
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.3rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, .retro-title, .retro-nav-title {
        font-family: 'Teko', sans-serif !important;
        color: var(--text) !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    h3 {
        font-size: 2.15rem !important;
        line-height: 1;
        margin-bottom: 0.8rem;
    }

    /* Header */
    .retro-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 22px;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    .traffic-light {
        width: 58px;
        background: #252118;
        border-radius: 8px;
        padding: 9px 10px;
        box-shadow: inset 0 0 0 2px #5f5339;
        display: flex;
        flex-direction: column;
        gap: 8px;
        flex-shrink: 0;
    }

    .traffic-light span {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: block;
        margin: 0 auto;
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.12);
    }

    .light-red    { background: #d83d32; }
    .light-yellow { background: #e0a020; }
    .light-green  { background: #43b14a; }

    .retro-title {
        font-size: 3.15rem;
        line-height: 0.9;
        margin: 0;
    }

    .retro-subtitle {
        font-family: 'Teko', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-size: 0.95rem;
        color: var(--muted);
        margin-top: 0.2rem;
        text-align: center;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
        background: transparent;
        padding: 0;
        margin-bottom: 1.1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--tan);
        border-radius: 999px;
        padding: 8px 18px 7px;
        border: 2px solid transparent;
        color: #5f5035;
        font-family: 'Teko', sans-serif !important;
        font-size: 1.02rem;
        font-weight: 600;
        letter-spacing: 0.7px;
        min-height: auto;
        box-shadow: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--tan-dark) !important;
        color: #fff6dc !important;
        border-color: #86612c !important;
        box-shadow: none !important;
    }

    /* Left form panel marker */
    div[data-testid="stVerticalBlock"]:has(.planner-left-marker) {
        background: var(--olive);
        border-radius: 10px;
        padding: 18px 18px 14px;
        border: 1px solid #79856a;
    }

    /* Labels inside planner panel */
    div[data-testid="stVerticalBlock"]:has(.planner-left-marker) label {
        color: #f7f1d7 !important;
        font-family: 'Teko', sans-serif !important;
        text-transform: uppercase;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px;
    }

    /* Select / date / time inputs */
    div[data-baseweb="select"] > div {
        background: var(--input-bg);
        border: 2px solid #d8cf9c;
        border-radius: 14px;
        min-height: 40px;
        color: var(--text);
        box-shadow: none !important;
    }

    .stDateInput > div > div,
    .stTimeInput > div > div {
        background: var(--input-bg);
        border: 2px solid #d8cf9c;
        border-radius: 14px;
        box-shadow: none !important;
    }

    .stDateInput input,
    .stTimeInput input {
        color: var(--text) !important;
        background: transparent !important;
    }

    /* Slider */
    [data-testid="stSlider"] {
        padding-top: 0.1rem;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--tan) !important;
        border: 2px solid #bb8947 !important;
        box-shadow: none !important;
    }

    /* Buttons */
    .stButton > button {
        background: #f2e8bf;
        color: var(--blue-accent);
        border: none;
        border-radius: 999px;
        padding: 0.28rem 1.1rem;
        font-family: 'Teko', sans-serif !important;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.7px;
        box-shadow: none !important;
        width: 150px;
        display: block;
        margin: 0.6rem auto 0 auto;
    }

    .stButton > button:hover {
        background: #fbf2cd;
        color: #496d7f;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Placeholder / preview panel */
    .planner-preview {
        min-height: 340px;
        border: 2.5px dashed var(--dash);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: var(--olive-dark);
        font-family: 'Teko', sans-serif;
        font-size: 1.15rem;
        line-height: 1.05;
        background: rgba(255,255,255,0.08);
        padding: 1rem;
    }

    /* Route result cards */
    .route-card {
        background: rgba(255, 248, 223, 0.72);
        border: 2px solid #cdbd87;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: none;
    }

    .route-rank {
        font-family: 'Teko', sans-serif !important;
        font-size: 1.45rem;
        color: var(--text);
    }

    .route-time {
        font-family: 'Teko', sans-serif !important;
        font-size: 1.25rem;
        color: var(--blue-accent);
    }

    .route-path {
        font-size: 0.85rem;
        color: var(--muted);
        word-break: break-word;
        margin-top: 4px;
    }

    .metric-pill {
        display: inline-block;
        background: #d7bf88;
        color: #5d533d;
        border-radius: 999px;
        padding: 4px 12px;
        font-family: 'Teko', sans-serif;
        font-size: 0.92rem;
        letter-spacing: 0.4px;
        margin: 2px;
        border: none;
    }

    /* Formula card */
    .formula-card {
        background: rgba(255, 248, 223, 0.55);
        border: 2px solid #d4c38d;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 14px;
    }

    .formula-code {
        font-family: monospace;
        color: var(--olive-dark);
        font-size: 0.86rem;
        margin-top: 5px;
    }

    /* Dataframes and alerts */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Config helpers ──────────────────────────────────────────────────────────

CONFIG_PATH = PROJECT_ROOT / "config.ini"


def _load_cfg() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    return cfg


def _resolve(cfg: configparser.ConfigParser, key: str) -> Path:
    raw = cfg.get("paths", key)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ── Cached SCATS site list ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_site_list() -> pd.DataFrame:
    cfg = _load_cfg()
    clean_csv = _resolve(cfg, "clean_scats_csv")
    raw_xls = _resolve(cfg, "raw_scats_xls")

    if clean_csv.exists():
        # Force site IDs to be strings to preserve leading zeros
        df = pd.read_csv(clean_csv, dtype={"scats_number": str})
    elif raw_xls.exists():
        from preprocessing.data_loader import load_scats_data
        with st.spinner("Parsing raw SCATS data for the first time…"):
            df = load_scats_data(raw_xls)
            clean_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(clean_csv, index=False)
    else:
        return pd.DataFrame(columns=["scats_number", "location", "nb_latitude", "nb_longitude"])

    sites = (
        df[["scats_number", "location", "nb_latitude", "nb_longitude"]]
        .drop_duplicates(subset=["scats_number"])
        .sort_values("scats_number")
        .reset_index(drop=True)
    )
    return sites


def _site_options(sites: pd.DataFrame) -> List[str]:
    return [
        f"{str(row.scats_number).zfill(4)} — {row.location}"
        for _, row in sites.iterrows()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align:center; padding: 28px 0 18px;">
        <span style="font-size:2.8rem;">🚦</span>
        <h1 style="margin:0; font-size:2.2rem; font-weight:700;
                   background: linear-gradient(90deg,#6c63ff,#48c6ef);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Traffic-Based Route Guidance System
        </h1>
        <p style="color:#888; margin-top:6px; font-size:0.95rem;">
            COS30019 Assignment 2B · Swinburne University
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Route Planner", "🧠 Train Models", "📊 Evaluate", "⚙️ Settings", "🧪 Test Suite"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROUTE PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Find Top-5 Routes")
    col_l, col_r = st.columns([2, 3], gap="large")

    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        sites = _load_site_list()
        options = _site_options(sites)
        no_sites = len(options) == 0

        if no_sites:
            st.warning("⚠️ No SCATS sites loaded. Check that `data/raw/Scats Data October 2006.xls` exists.")

        origin_sel = st.selectbox(
            "🔵 Origin SCATS site",
            options if options else ["— no data —"],
            index=0,
            key="origin_sel",
        )
        dest_sel = st.selectbox(
            "🔴 Destination SCATS site",
            options if options else ["— no data —"],
            index=min(1, len(options) - 1),
            key="dest_sel",
        )

        travel_date = st.date_input(
            "📅 Travel date",
            value=date(2006, 10, 15),
            key="travel_date",
        )
        travel_time_val = st.time_input("🕒 Travel time", value=datetime.strptime("08:00", "%H:%M").time(), key="travel_time")

        cfg = _load_cfg()
        model_options = ["lstm", "gru", "bilstm", "transformer"]
        default_model = cfg.get("models", "default_model", fallback="lstm")
        model_idx = model_options.index(default_model) if default_model in model_options else 0
        selected_model = st.selectbox(
            "🤖 ML Model",
            model_options,
            index=model_idx,
            format_func=lambda x: x.upper(),
            key="route_model",
        )
        top_k = st.slider("🔢 Number of routes", min_value=1, max_value=10, value=5, key="top_k_slider")

        find_btn = st.button("🔍 Find Routes", key="find_routes_btn", disabled=no_sites)
        st.markdown("</div>", unsafe_allow_html=True)

        if find_btn and not no_sites:
            origin_id = origin_sel.split(" — ")[0].strip()
            dest_id = dest_sel.split(" — ")[0].strip()
            travel_dt = datetime.combine(travel_date, travel_time_val)

            if origin_id == dest_id:
                st.error("Origin and destination must be different.")
            else:
                with st.spinner("🔍 Calculating optimal routes based on predicted traffic..."):
                    try:
                        from main import find_routes
                        result = find_routes(
                            origin_id=origin_id,
                            dest_id=dest_id,
                            datetime=travel_dt,
                            model=selected_model,
                            k=top_k,
                            return_network=True,
                        )
                        # --- Persistence Store ---
                        st.session_state.route_result = result
                        st.session_state.route_dt = travel_dt
                    except Exception as e:
                        st.error(f"Routing error: {e}")
                        st.session_state.route_result = None

        if not no_sites:
            with st.expander("📊 Data Integrity Diagnostics"):
                try:
                    # Quick check on the current sites vs flow data
                    from main import _load_clean_flow_data, _normalize_node_id
                    fdf = _load_clean_flow_data(cfg)
                    total_records = len(fdf)
                    unique_sites = fdf["scats_number"].nunique()
                    
                    # Create normalized set for checking
                    norm_ids = fdf["scats_number"].astype(str).apply(_normalize_node_id).unique()
                    
                    st.write(f"**Total Records:** `{total_records:,}`")
                    st.write(f"**Unique SCATS IDs:** `{unique_sites}`")
                    
                    # Check for coordinates
                    coord_sites = fdf.dropna(subset=["nb_latitude", "nb_longitude"])["scats_number"].nunique()
                    st.write(f"**Sites with Coords:** `{coord_sites}` / `{unique_sites}`")
                    
                    # ID Format Check
                    sample_id = norm_ids[0] if len(norm_ids)>0 else "N/A"
                    st.write(f"**ID Format check:** `{sample_id}` (Should be 4 digits)")
                    
                    if coord_sites == 0:
                        st.error("🚨 CRITICAL: No coordinates found in data. Maps will be blank.")
                    elif coord_sites < unique_sites:
                        st.warning(f"⚠️ Some sites ({unique_sites - coord_sites}) are missing coordinates.")
                    else:
                        st.success("✅ All sites have valid map coordinates.")
                except Exception as ex:
                    st.error(f"Diag failed: {ex}")

            with st.expander("🌐 Current Road Network Connectivity"):
                st.markdown(
                    """
                    **Note:** The system uses a graph-based road network inferred from SCATS sensor locations. 
                    Connections are made primarily between sensors on the same road (up to 2km apart) or nearby 
                    intersections (within 0.5km). Use the table below to verify paths.
                    """
                )
                edges_path = _resolve(cfg, "graph_edges_csv")
                if edges_path.exists():
                    e_df = pd.read_csv(edges_path)
                    st.dataframe(e_df, height=300, width="stretch")
                else:
                    st.warning("Graph edges file not found. Rebuild it in Settings.")

    with col_r:
        # --- Persistence Render ---
        result = st.session_state.route_result
        travel_dt = st.session_state.route_dt

        if not result:
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:center;
                            height:320px; border:2px dashed rgba(108,99,255,0.3);
                            border-radius:16px; color:#555; flex-direction:column; gap:10px;">
                    <span style="font-size:2.5rem">🗺️</span>
                    <span>Select origin, destination, date/time and click <b>Find Routes</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            routes = result["routes"]
            if not routes:
                st.warning(
                    """
                    ⚠️ **No routes found between these sites.** 
                    
                    SCATS sites in this dataset represent fixed sensors in the Boroondara area. 
                    A route can only be found if a physical path exists between sensors in our road network graph.
                    
                    **Suggestions:**
                    - Check the **'Current Road Network Connectivity'** table below to see which sites are connected.
                    - Try selecting sites that are closer together along the same main road (e.g. High St).
                    - Use the **Settings** tab to **'Rebuild Graph Edges'** if you have recently changed connectivity parameters.
                    """
                )
            else:
                st.markdown(f"**{len(routes)} route(s) found** — {travel_dt.strftime('%A %d %b %Y, %H:%M')}")
                rank_colors = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

                for i, route in enumerate(routes):
                    mins = route["total_time_min"]
                    path_str = " → ".join(route["path"])
                    n_segs = len(route["segments"])
                    st.markdown(
                        f"""
                        <div class="route-card">
                            <span class="route-rank">{rank_colors[i]} Route {i+1}</span>
                            <span class="route-time" style="float:right">{mins:.1f} min</span>
                            <div class="route-path">{path_str}</div>
                            <div style="margin-top:8px">
                                <span class="metric-pill">⛓ {n_segs} segments</span>
                                <span class="metric-pill">🤖 {route['model_used'].upper()}</span>
                                <span class="metric-pill">⏱ {route['total_time_sec']:.0f} sec</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("🛠 Map Calibration (Research Initiative)"):
                    st.info("SCATS coordinates often have slight offsets. Use these sliders to calibrate.")
                    lat_offset = st.slider("Latitude Offset", -0.01, 0.01, 0.0, step=0.0001, format="%.4f")
                    lon_offset = st.slider("Longitude Offset", -0.01, 0.01, 0.0, step=0.0001, format="%.4f")

                # Apply offset to coordinates and normalize keys to string
                calibrated_coords = {
                    str(nid): (lat + lat_offset, lon + lon_offset)
                    for nid, (lat, lon) in result["node_coords"].items()
                }

                # Create and display Folium map with safe centering
                map_center = (-37.83, 145.07) # Boroondara default
                if calibrated_coords:
                    map_center = next(iter(calibrated_coords.values()))

                m = map_utils.create_focused_route_map(
                    routes=routes,
                    node_coords=calibrated_coords,
                    edge_flow=result["edge_flow"],
                    center=map_center
                )
                
                # Primary Map for Top 5 routes
                st_folium(m, width="100%", height=500, key="main_route_map")

                with st.expander("🔎 Segment detail — Route 1"):
                    seg_rows = []
                    for seg in routes[0]["segments"]:
                        seg_rows.append({
                            "From": seg["from"],
                            "To": seg["to"],
                            "Flow (veh/hr)": f"{seg['predicted_flow_veh_per_hour']:.0f}" if seg["predicted_flow_veh_per_hour"] is not None else "—",
                            "Time (sec)": f"{seg['travel_time_sec']:.1f}" if seg["travel_time_sec"] is not None else "—",
                        })
                    st.dataframe(pd.DataFrame(seg_rows), width="stretch")

                if not result.get("node_coords"):
                    st.warning("⚠️ **Network coordinates missing.** Your map is blank because the clean SCATS data doesn't contain site locations. Please remove your clean CSV and re-parse the raw Excel file.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Train a Traffic Flow Model")
    col_a, col_b = st.columns([1, 2], gap="large")

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cfg = _load_cfg()
        train_model = st.selectbox(
            "Model architecture",
            ["lstm", "gru", "bilstm", "transformer"],
            format_func=lambda x: {
                "lstm": "LSTM (Baseline)",
                "gru": "GRU (Efficient)",
                "bilstm": "BiLSTM (Model C)",
                "transformer": "Transformer (Bonus)",
            }[x],
            key="train_model_sel",
        )
        epochs = st.slider("Epochs", 5, 200, int(cfg.get("training", "epochs", fallback="100")), step=5, key="train_epochs")
        batch_size = st.select_slider("Batch size", [16, 32, 64, 128, 256], value=128, key="train_batch")
        lr = st.select_slider(
            "Learning rate",
            [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
            key="train_lr",
        )
        patience = st.slider("Early stop patience", 3, 30, int(cfg.get("training", "patience", fallback="12")), key="train_patience")
        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], key="train_device")

        # Data status
        dataset_path = _resolve(cfg, "dataset_npz")
        scaler_path = _resolve(cfg, "scaler_path")
        data_ok = dataset_path.exists() and scaler_path.exists()
        checkpoint_dir = _resolve(cfg, "checkpoints_dir")
        ckpt_path = checkpoint_dir / f"{train_model}.pt"

        st.markdown("---")
        st.markdown("**Data status**")
        if data_ok:
            st.markdown('<span class="badge-ok">✓ datasets.npz ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">⚠ datasets.npz missing</span>', unsafe_allow_html=True)

        if ckpt_path.exists():
            st.markdown(f'<span class="badge-ok">✓ {train_model}.pt exists</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-warn">○ No checkpoint yet</span>', unsafe_allow_html=True)

        preproc_btn = st.button("⚙️ Preprocess Data First", key="preproc_btn")
        train_btn = st.button("🚀 Start Training", key="train_btn")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        # --- Preprocess ---
        if preproc_btn:
            with st.spinner("Running preprocessing pipeline…"):
                try:
                    clean_csv = _resolve(cfg, "clean_scats_csv")
                    raw_xls = _resolve(cfg, "raw_scats_xls")

                    if clean_csv.exists():
                        df = pd.read_csv(clean_csv)
                    else:
                        from preprocessing.data_loader import load_scats_data
                        df = load_scats_data(raw_xls)
                        clean_csv.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(clean_csv, index=False)

                    from preprocessing.feature_engineering import prepare_datasets

                    lookback = cfg.getint("preprocessing", "lookback", fallback=12)
                    splits = prepare_datasets(
                        dataframe=df,
                        lookback=lookback,
                        train_days=cfg.getint("preprocessing", "train_days", fallback=21),
                        val_days=cfg.getint("preprocessing", "val_days", fallback=5),
                        test_days=cfg.getint("preprocessing", "test_days", fallback=5),
                        min_available_days=cfg.getint("preprocessing", "min_available_days", fallback=21),
                        scaler_output_path=scaler_path,
                        dataset_output_path=dataset_path,
                    )
                    st.success(
                        f"✅ Preprocessing complete!\n\n"
                        f"- Train: {splits.train_x.shape[0]:,} samples\n"
                        f"- Val:   {splits.val_x.shape[0]:,} samples\n"
                        f"- Test:  {splits.test_x.shape[0]:,} samples\n"
                        f"- Streams used: {splits.metadata['used_stream_count']}"
                    )
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

        # --- Train ---
        if train_btn:
            prog_placeholder = st.empty()
            log_placeholder = st.empty()
            chart_placeholder = st.empty()

            with prog_placeholder.container():
                st.info(f"Training **{train_model.upper()}** — this may take a while…")
                prog_bar = st.progress(0)

            # Run training in subprocess so the GUI doesn't freeze
            cmd = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "training" / "train.py"),
                "--model", train_model,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--patience", str(patience),
                "--device", device_choice,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )
                log_lines: List[str] = []
                train_losses, val_losses = [], []
                epoch_num = 0

                for line in proc.stdout:  # type: ignore[union-attr]
                    line = line.rstrip()
                    log_lines.append(line)

                    # Parse epoch progress
                    if line.startswith("Epoch "):
                        try:
                            parts = line.split()
                            ep_frac = parts[1]  # e.g. "003/100"
                            ep_done = int(ep_frac.split("/")[0])
                            ep_total = int(ep_frac.split("/")[1])
                            prog_bar.progress(ep_done / ep_total)
                            epoch_num = ep_done

                            # Extract losses
                            for part in parts:
                                if part.startswith("train_loss="):
                                    train_losses.append(float(part.split("=")[1]))
                                if part.startswith("val_loss="):
                                    val_losses.append(float(part.split("=")[1]))
                        except Exception:
                            pass

                        # Update live chart
                        if len(train_losses) > 1:
                            chart_df = pd.DataFrame(
                                {"Train Loss": train_losses, "Val Loss": val_losses}
                            )
                            with chart_placeholder.container():
                                st.markdown("**Training Loss Curve**")
                                st.line_chart(chart_df, width="stretch")

                    # Update log
                    log_placeholder.code("\n".join(log_lines[-30:]), language="bash")

                proc.wait()
                prog_bar.progress(1.0)

                if proc.returncode == 0:
                    prog_placeholder.success(f"✅ Training complete! Checkpoint saved to `{ckpt_path}`")
                    st.cache_data.clear()
                else:
                    prog_placeholder.error("❌ Training process exited with errors. See log above.")

            except Exception as e:
                prog_placeholder.error(f"Failed to launch training: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Model Evaluation & Comparison")
    cfg = _load_cfg()

    eval_models = st.multiselect(
        "Select models to evaluate",
        ["lstm", "gru", "bilstm", "transformer"],
        default=["lstm", "gru", "bilstm"],
        format_func=lambda x: x.upper(),
        key="eval_models",
    )
    eval_btn = st.button("▶ Run Evaluation", key="eval_btn")

    eval_dir = _resolve(cfg, "evaluation_dir")
    metrics_csv = eval_dir / "metrics_table.csv"

    if eval_btn:
        checkpoint_dir = _resolve(cfg, "checkpoints_dir")
        dataset_path = _resolve(cfg, "dataset_npz")
        scaler_path = _resolve(cfg, "scaler_path")

        if not dataset_path.exists():
            st.error("datasets.npz not found. Run preprocessing first (Training tab).")
        else:
            with st.spinner("Evaluating models…"):
                try:
                    from training.evaluate import evaluate_models, DEFAULT_MODEL_SPECS

                    # Add bilstm spec dynamically if not present
                    if "bilstm" not in DEFAULT_MODEL_SPECS:
                        DEFAULT_MODEL_SPECS["bilstm"] = {
                            "module": "models.bilstm_model",
                            "class_candidates": ["BiLSTMRegressor"],
                            "checkpoint": str(checkpoint_dir / "bilstm.pt"),
                        }
                    else:
                        DEFAULT_MODEL_SPECS["bilstm"]["checkpoint"] = str(checkpoint_dir / "bilstm.pt")

                    overrides = {
                        m: str(checkpoint_dir / f"{m}.pt")
                        for m in eval_models
                        if (checkpoint_dir / f"{m}.pt").exists()
                    }

                    metrics_df = evaluate_models(
                        dataset_path=dataset_path,
                        scaler_path=scaler_path,
                        output_dir=eval_dir,
                        model_names=eval_models,
                        checkpoint_overrides=overrides,
                        device=cfg.get("inference", "device", fallback="auto"),
                    )
                    st.success("Evaluation complete!")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    # Display results if available
    if metrics_csv.exists():
        st.markdown("#### 📋 Metrics Comparison")
        metrics_df = pd.read_csv(metrics_csv)
        metrics_df.columns = [c.replace("_percent", " (%)").upper() for c in metrics_df.columns]

        # Style the table
        styled = (
            metrics_df.style
            .format({c: "{:.4f}" for c in metrics_df.columns if c != "MODEL"})
            .highlight_min(
                subset=[c for c in metrics_df.columns if c != "MODEL"],
                color="rgba(74,222,128,0.2)",
            )
        )
        st.dataframe(styled, width="stretch")

        # Bar chart
        st.markdown("#### 📊 MAE / RMSE / MAPE Comparison")
        if "MAE" in metrics_df.columns and "MODEL" in metrics_df.columns:
            bar_data = metrics_df.set_index("MODEL")[["MAE", "RMSE"]].rename_axis("Model")
            st.bar_chart(bar_data, width="stretch")

    # Predicted vs Actual plots
    pred_plot = eval_dir / "predicted_vs_actual.png"
    if pred_plot.exists():
        st.markdown("#### 📈 Predicted vs Actual (Test Set)")
        st.image(str(pred_plot), width="stretch")

    # Per-model plots
    for model_name in ["lstm", "gru", "bilstm", "transformer"]:
        per_plot = eval_dir / f"pred_vs_actual_{model_name}.png"
        if per_plot.exists():
            with st.expander(f"📉 {model_name.upper()} — Detailed plot"):
                st.image(str(per_plot), width="stretch")

    if not metrics_csv.exists() and not eval_btn:
        st.markdown(
            """
            <div style="text-align:center; padding:60px; color:#555;
                        border:2px dashed rgba(255,255,255,0.1); border-radius:16px;">
                <span style="font-size:2rem">📊</span><br>
                Train at least one model then click <b>Run Evaluation</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Settings — `config.ini`")
    cfg = _load_cfg()
    col_s1, col_s2 = st.columns(2, gap="large")

    with col_s1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🚗 Travel Time**")
        speed_limit = st.number_input(
            "Speed limit (km/h)", 20, 120,
            int(float(cfg.get("travel_time", "speed_limit_kmh", fallback="60"))),
            key="cfg_speed",
        )
        intersection_delay = st.number_input(
            "Intersection delay (sec)", 5, 120,
            int(float(cfg.get("travel_time", "intersection_delay_sec", fallback="30"))),
            key="cfg_delay",
        )
        free_flow_threshold = st.number_input(
            "Free-flow threshold (veh/hr)", 100, 1000,
            int(float(cfg.get("travel_time", "free_flow_threshold", fallback="351"))),
            key="cfg_fft",
        )
        min_speed = st.number_input(
            "Minimum speed (km/h)", 1, 30,
            int(float(cfg.get("travel_time", "min_speed_kmh", fallback="5"))),
            key="cfg_min_speed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🔀 Routing**")
        default_k = st.slider(
            "Default top-k routes", 1, 10,
            int(cfg.get("routing", "default_k", fallback="5")),
            key="cfg_k",
        )
        max_depth = st.slider(
            "Max path depth", 5, 30,
            int(cfg.get("routing", "max_path_depth", fallback="14")),
            key="cfg_depth",
        )
        undirected = st.checkbox(
            "Undirected graph (bidirectional edges)",
            value=cfg.getboolean("routing", "undirected", fallback=False),
            key="cfg_undirected",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_s2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🧠 Training Defaults**")
        def_model_opts = ["lstm", "gru", "bilstm", "transformer"]
        def_model_val = cfg.get("models", "default_model", fallback="lstm")
        def_model_idx = def_model_opts.index(def_model_val) if def_model_val in def_model_opts else 0
        default_model_setting = st.selectbox(
            "Default model",
            def_model_opts,
            index=def_model_idx,
            format_func=lambda x: x.upper(),
            key="cfg_default_model",
        )
        cfg_epochs = st.slider("Default epochs", 10, 300, int(cfg.get("training", "epochs", fallback="100")), key="cfg_epochs")
        cfg_batch = st.select_slider("Default batch size", [16, 32, 64, 128, 256], value=128, key="cfg_batch")
        cfg_lookback = st.slider("Lookback window (time steps)", 4, 48, int(cfg.get("preprocessing", "lookback", fallback="12")), key="cfg_lookback")
        cfg_patience = st.slider("Early stop patience", 3, 30, int(cfg.get("training", "patience", fallback="12")), key="cfg_patience")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🖥️ Inference**")
        device_opts = ["auto", "cpu", "cuda"]
        cur_device = cfg.get("inference", "device", fallback="auto")
        device_setting = st.selectbox(
            "Compute device",
            device_opts,
            index=device_opts.index(cur_device) if cur_device in device_opts else 0,
            key="cfg_device",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    save_btn = st.button("💾 Save Settings", key="save_settings_btn")
    if save_btn:
        try:
            cfg.set("travel_time", "speed_limit_kmh", str(speed_limit))
            cfg.set("travel_time", "intersection_delay_sec", str(intersection_delay))
            cfg.set("travel_time", "free_flow_threshold", str(free_flow_threshold))
            cfg.set("travel_time", "min_speed_kmh", str(min_speed))
            cfg.set("routing", "default_k", str(default_k))
            cfg.set("routing", "max_path_depth", str(max_depth))
            cfg.set("routing", "undirected", str(undirected).lower())
            cfg.set("models", "default_model", default_model_setting)
            cfg.set("training", "epochs", str(cfg_epochs))
            cfg.set("training", "batch_size", str(cfg_batch))
            cfg.set("training", "patience", str(cfg_patience))
            cfg.set("preprocessing", "lookback", str(cfg_lookback))
            cfg.set("inference", "device", device_setting)

            with open(CONFIG_PATH, "w") as f:
                cfg.write(f)
            st.success("✅ Settings saved to config.ini")
        except Exception as e:
            st.error(f"Failed to save settings: {e}")

    st.markdown("---")
    st.markdown("**🛠 Utility Actions**")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    with col_u1:
        if st.button("🔁 Rebuild Graph Edges", key="rebuild_graph_btn"):
            with st.spinner("Building graph adjacency…"):
                try:
                    clean_csv = _resolve(cfg, "clean_scats_csv")
                    edges_csv = _resolve(cfg, "graph_edges_csv")
                    from preprocessing.graph_builder import build_graph_edges
                    # Enforce string loading for graph building too
                    df = pd.read_csv(clean_csv, dtype={"scats_number": str})
                    edge_df = build_graph_edges(df)
                    edge_df.to_csv(edges_csv, index=False)
                    st.success(f"✅ Built {len(edge_df)} edges → {edges_csv.name}")
                except Exception as e:
                    st.error(f"Graph build failed: {e}")
    with col_u2:
        if st.button("🔄 Total System Sync", key="reparse_data_btn", help="Clears data, graph, and cache to restore map connectivity."):
            with st.spinner("🔄 Deep Sync: Clearing memory & rebuilding graph..."):
                try:
                    import importlib
                    import main
                    importlib.reload(main) # Force reload of coordinate logic
                    
                    clean_csv = _resolve(cfg, "clean_scats_csv")
                    edges_csv = _resolve(cfg, "graph_edges_csv")
                    raw_xls = _resolve(cfg, "raw_scats_xls")
                    
                    # 1. Delete old caches
                    if clean_csv.exists(): clean_csv.unlink()
                    if edges_csv.exists(): edges_csv.unlink()
                    st.cache_data.clear()
                    
                    # 2. Re-parse SCATS data from XLS
                    from preprocessing.data_loader import load_scats_data
                    fdf = load_scats_data(raw_xls)
                    fdf.to_csv(clean_csv, index=False)
                    
                    # 3. Automatically Rebuild Graph Edges
                    from preprocessing.graph_builder import build_graph_edges
                    edf = build_graph_edges(fdf)
                    edf.to_csv(edges_csv, index=False)
                    
                    st.session_state.route_result = None
                    st.session_state.route_dt = None
                    
                    st.success(f"✅ Deep Sync Complete! ({len(edf)} edges). Map link restored.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync failed: {e}")
    with col_u3:
        if st.button("🗑 Clear Cache", key="clear_cache_btn"):
            st.cache_data.clear()
            st.success("Cache cleared.")
    with col_u4:
        edges_csv = _resolve(cfg, "graph_edges_csv")
        if edges_csv.exists():
            edge_df_display = pd.read_csv(edges_csv)
            st.markdown(
                f'<span class="badge-ok">✓ Graph: {len(edge_df_display)} edges</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="badge-warn">○ graph_edges.csv missing</span>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 🧪 Automated Test Suite")
    st.info("Run internal unit tests to verify system components (models, graph logic, etc.). Each test can be run against multiple model architectures.")
    
    test_dir = PROJECT_ROOT / "testing"
    if not test_dir.exists():
        st.error(f"Test directory not found: {test_dir}")
    else:
        # Natural sorting for test files (1, 2, ..., 10)
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', s)]
        
        test_files = sorted([f.name for f in test_dir.glob("testcase*.py")], key=natural_sort_key)
        
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Test Configuration**")
            to_run = st.multiselect("Select test cases", test_files, default=test_files[:3] if test_files else [])
            models_to_test = st.multiselect("Models to verify", ["lstm", "gru", "bilstm", "transformer"], default=["lstm"])
            run_all = st.button("🚀 Run Tests", key="run_tests_btn")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_t2:
            if run_all and to_run and models_to_test:
                results = []
                import re
                
                # Global progress
                total_steps = len(to_run) * len(models_to_test)
                progress_bar = st.progress(0)
                step_count = 0

                for model_name in models_to_test:
                    st.markdown(f"#### 🧠 Testing Model: `{model_name.upper()}`")
                    for tf in to_run:
                        step_count += 1
                        progress_bar.progress(step_count / total_steps)
                        
                        with st.status(f"[{model_name.upper()}] Running {tf}...", expanded=False) as status:
                            try:
                                import subprocess
                                # Run test case with --model argument
                                res = subprocess.run(
                                    [sys.executable, str(test_dir / tf), "--model", model_name],
                                    capture_output=True,
                                    text=True,
                                    timeout=60
                                )
                                
                                # Extract meaningful result using regex
                                result_match = re.search(r"\[TC\d+\] Result:\s*(.*)", res.stdout)
                                meaningful_result = result_match.group(1).strip() if result_match else "No explicit result found"
                                
                                # Determine pass/fail based on return code AND meaningful result
                                if res.returncode == 0 and "FAILED" not in meaningful_result.upper():
                                    status_val = "✅ PASSED"
                                    st_state = "complete"
                                else:
                                    status_val = "❌ FAILED"
                                    st_state = "error"

                                results.append({
                                    "Model": model_name.upper(),
                                    "Test": tf,
                                    "Status": status_val,
                                    "Meaningful Result": meaningful_result
                                })
                                
                                if res.stdout:
                                    st.code(res.stdout, language="bash")
                                if res.stderr:
                                    st.error(f"Error output:\n{res.stderr}")
                                
                                status.update(label=f"[{model_name.upper()}] {tf} - {status_val}", state=st_state)
                            except Exception as e:
                                st.error(f"Failed to run {tf}: {e}")
                                results.append({"Model": model_name.upper(), "Test": tf, "Status": "❌ ERROR", "Meaningful Result": str(e)})
                
                st.markdown("#### 📊 Execution Summary")
                st.dataframe(pd.DataFrame(results), width="stretch")
            else:
                st.markdown(
                    """
                    <div style="text-align:center; padding:100px; color:#555;
                                border:2px dashed rgba(255,255,255,0.1); border-radius:16px;">
                        Configure test cases and models then click <b>Run Tests</b>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
