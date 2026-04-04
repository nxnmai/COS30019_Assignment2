# Traffic-Based Route Guidance System (TBRGS) — Assignment 2B

A comprehensive system for traffic flow prediction and optimal route guidance using deep learning (LSTM, GRU, BiLSTM, Transformer) and graph-based routing algorithms.

## 📁 Project Structure

```text
Assignment2B/
├── main.py              # Central routing engine & system entry point
├── config.ini           # System configuration (paths, model hyperparams)
├── requirements.txt     # Python dependencies
├── data/
│   ├── raw/             # Original SCATS .xls files
│   └── processed/       # Cleaned CSVs and generated road graphs
├── gui/
│   ├── app.py           # Streamlit dashboard (Tabs: Planner, Train, Eval, Settings)
│   └── map_utils.py     # High-fidelity Folium map rendering logic
├── models/              # Neural network architectures (PyTorch)
├── prediction/          # Travel time calculation and flow estimation
├── preprocessing/       # SCATS data parsing and Graph Building logic
├── routing/             # Yen's K-Shortest Paths & Dijkstra implementation
├── training/            # Model training & checkpoint management
└── testing/             # Unit tests and performance benchmarks
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Synchronization & Setup
Launch the GUI and use the **Settings** tab to perform a **Total System Sync**. This will:
- Parse the raw SCATS Excel data.
- Build the road network adjacency graph.
- Normalize site IDs (e.g., 0970).

```bash
streamlit run gui/app.py
```

### 3. Command Line Training (Optional)
You can also train models directly from the terminal:
```bash
python training/train.py --model lstm --epochs 100
python training/train.py --model transformer --epochs 100
```

## 🗺️ Key Features

- **Focused Mapping**: Interactive Folium maps showing the Top 5 routes with traffic-aware color coding (Green to Red).
- **Deep Learning Suite**: Support for LSTM, GRU, BiLSTM, and Transformer architectures for time-series flow prediction.
- **Micro-Calibration**: Manual coordinate offset adjustment (defaulted to 0.0013) for precise sensor alignment.
- **Shadow Normalization**: Robust ID matching system ensuring parity between raw traffic data and graph nodes.
