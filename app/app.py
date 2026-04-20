from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import RetailChurnPredictor

app = Flask(__name__)

# Load artifacts once at startup to avoid reloading on every request.
PREDICTOR = RetailChurnPredictor()


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    try:
        prediction = PREDICTOR.predict(payload)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    return jsonify({"prediction": prediction}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
