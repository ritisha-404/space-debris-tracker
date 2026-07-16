# Space Debris Tracker

An LSTM-based web app for tracking space debris trajectories and flagging potential
close approaches ("conjunctions") with active satellites.

This repo is a **starter scaffold**, not a finished product. It gives you a working
end-to-end skeleton — synthetic data, a trainable LSTM, a Flask API, and a simple
dashboard frontend — so you can swap in real orbital data (e.g. TLEs from Space-Track
or CelesTrak) and iterate.

## How it works (current state)

1. `backend/data_utils.py` generates synthetic orbital trajectories (simple elliptical
   orbits + noise) and slices them into `(sequence_in -> next_position)` training
   windows.
2. `backend/model.py` defines a small PyTorch LSTM that takes a window of past
   `(x, y, z)` positions and predicts the next position.
3. `backend/train.py` trains the model on synthetic data and saves a checkpoint to
   `models/lstm_debris.pt`.
4. `backend/app.py` is a Flask API that:
   - serves the frontend
   - `GET /api/debris` — returns sample debris objects + their recent trajectories
   - `POST /api/predict` — takes a sequence of positions, returns the model's
     predicted next position
   - `GET /api/risk` — naive pairwise distance check between predicted positions to
     flag potential close approaches
5. `frontend/` is a plain HTML/JS/Chart.js dashboard that calls the API and plots
   trajectories + predictions.

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

## Where to take this next

- **Real data**: swap `data_utils.py`'s synthetic generator for real TLE data pulled
  from [CelesTrak](https://celestrak.org/) or [Space-Track.org](https://www.space-track.org/),
  propagated with `sgp4` or `skyfield` into position vectors.
- **Better features**: feed the LSTM velocity, orbital elements, or atmospheric drag
  estimates, not just raw position.
- **Sequence-to-sequence prediction**: predict N future steps instead of one, for
  longer-horizon conjunction screening.
- **Real risk metric**: replace the naive Euclidean-distance check in `/api/risk`
  with a proper probability-of-collision calculation (e.g. using covariance/uncertainty
  from an ensemble or a probabilistic LSTM).
- **Persistence**: add a database (Postgres/SQLite) instead of in-memory sample data.
- **Auth + deployment**: containerize with Docker, add a production WSGI server
  (gunicorn), deploy behind HTTPS.

## License

Use whatever you like — this is a starter scaffold with no license restrictions attached.
Add your own `LICENSE` file once you decide.
