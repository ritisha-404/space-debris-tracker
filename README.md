# Space Debris Tracker

An LSTM-based web app for tracking space debris trajectories and flagging potential
close approaches ("conjunctions") with active satellites.

This repo is a **starter scaffold**, not a finished product. It gives you a working
end-to-end skeleton — synthetic data, a trainable LSTM, a Flask API, and a simple
dashboard frontend — so you can swap in real orbital data (e.g. TLEs from Space-Track
or CelesTrak) and iterate.


## Quickstart

```bash
# 1. create a virtualenv (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. train the (synthetic-data) starter model
python backend/train.py

# 4. run the web app
python backend/app.py
```

Then open http://localhost:5000 in your browser.

## Project structure

```
space-debris-tracker/
├── backend/
│   ├── app.py           # Flask API + static file serving
│   ├── model.py         # LSTM model definition
│   ├── train.py         # Training script
│   └── data_utils.py    # Synthetic data generation + windowing
├── frontend/
│   ├── index.html       # Dashboard UI
│   ├── style.css
│   └── script.js        # Fetches API, renders charts
├── data/
│   └── sample_debris_data.csv   # Small static sample dataset
├── models/
│   └── .gitkeep          # trained checkpoints land here (git-ignored)
├── requirements.txt
└── .gitignore
```

