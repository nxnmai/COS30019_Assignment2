# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the road network graph (one-time)
cd Assignment2B
python preprocessing/graph_builder.py

# 3. Launch the GUI — use the Preprocess + Train buttons inside
streamlit run gui/app.py

# OR train from CLI:
python training/train.py --model lstm --epochs 100
python training/train.py --model gru --epochs 100
python training/train.py --model bilstm --epochs 100
