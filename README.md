# Space Debris Tracker (IN DEVELOPMENT)

An LSTM-based web app for tracking space debris trajectories and flagging potential
close approaches ("conjunctions") with active satellites.


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

## PREVIEW
<img width="963" height="960" alt="sdt" src="https://github.com/user-attachments/assets/fc6d2749-38e2-4eb6-ba8d-2e8740920512" />


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

