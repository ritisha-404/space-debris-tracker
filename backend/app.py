"""
app.py

flask API + static frontend server for the space debris tracker.

endpoints:
    GET  /                  -> dashboard (frontend/index.html)
    GET  /api/debris        -> sample debris objects with recent trajectories
    POST /api/predict       -> {"sequence": [[x,y,z], ...]} -> predicted next position
    GET  /api/risk          -> naive pairwise close-approach check across sample debris

run with:
    python backend/app.py
"""

import os

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from data_utils import generate_synthetic_debris_set
from model import DebrisLSTM

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "models", "lstm_debris.pt")
NORM_PATH = CHECKPOINT_PATH.replace(".pt", "_norm.npz")
SEQ_LEN = 10
CLOSE_APPROACH_THRESHOLD_KM = 25.0  # naive flag threshold, tune as needed

app = Flask(__name__, static_folder=None)

# --- Load model (if trained) -------------------------------------------------
_model = None
_mean = np.zeros(3, dtype=np.float32)
_std = np.ones(3, dtype=np.float32)

if os.path.exists(CHECKPOINT_PATH):
    _model = DebrisLSTM()
    _model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    _model.eval()
    if os.path.exists(NORM_PATH):
        stats = np.load(NORM_PATH)
        _mean, _std = stats["mean"], stats["std"]
    print("Loaded trained model from", CHECKPOINT_PATH)
else:
    print("No trained checkpoint found. Run `python backend/train.py` first. "
          "Endpoints that need the model will return an error until then.")

_sample_dataset = generate_synthetic_debris_set(num_objects=8, num_steps=60)


def _predict_next(sequence):
    """sequence: list of [x, y, z] (length SEQ_LEN) -> predicted [x, y, z]"""
    if _model is None:
        raise RuntimeError("Model not loaded. Train it first with backend/train.py")

    arr = np.array(sequence, dtype=np.float32)
    norm = (arr - _mean) / _std
    x = torch.from_numpy(norm).unsqueeze(0)  # (1, seq_len, 3)

    with torch.no_grad():
        pred_norm = _model(x).squeeze(0).numpy()

    pred = pred_norm * _std + _mean
    return pred.tolist()



@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/debris")
def api_debris():
    """Return sample debris objects with their recent trajectory."""
    result = []
    for obj_id, traj in _sample_dataset.items():
        result.append({
            "id": obj_id,
            "trajectory": traj[-SEQ_LEN:].tolist(),
        })
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    sequence = body.get("sequence")

    if not sequence or len(sequence) != SEQ_LEN:
        return jsonify({
            "error": f"'sequence' must be a list of {SEQ_LEN} [x, y, z] points"
        }), 400

    try:
        prediction = _predict_next(sequence)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({"predicted_position": prediction})


@app.route("/api/risk")
def api_risk():
    """Naive pairwise close-approach screening across the sample dataset.

    For each object, predicts its next position, then flags any pair of
    predicted positions closer than CLOSE_APPROACH_THRESHOLD_KM.

    This is a placeholder heuristic.
    """
    if _model is None:
        return jsonify({"error": "Model not loaded. Run backend/train.py first."}), 503

    predictions = {}
    for obj_id, traj in _sample_dataset.items():
        seq = traj[-SEQ_LEN:].tolist()
        predictions[obj_id] = _predict_next(seq)

    flagged = []
    ids = list(predictions.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = np.array(predictions[ids[i]]), np.array(predictions[ids[j]])
            dist = float(np.linalg.norm(a - b))
            if dist < CLOSE_APPROACH_THRESHOLD_KM:
                flagged.append({
                    "object_a": ids[i],
                    "object_b": ids[j],
                    "predicted_distance_km": round(dist, 3),
                })

    return jsonify({
        "predictions": predictions,
        "close_approaches": flagged,
        "threshold_km": CLOSE_APPROACH_THRESHOLD_KM,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
