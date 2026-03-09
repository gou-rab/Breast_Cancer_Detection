import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=BASE_DIR, static_folder=STATIC_DIR)

MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave points_se",
    "symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
    "perimeter_worst","area_worst","smoothness_worst","compactness_worst",
    "concavity_worst","concave points_worst","symmetry_worst",
    "fractal_dimension_worst",
]

# ── Load model lazily (after train.py has run during build) ──────────────────
model  = None
scaler = None

def load_artifacts():
    global model, scaler
    if model is None:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_artifacts()
        data   = request.get_json(force=True)
        values = [float(data.get(f, 0)) for f in FEATURES]
        arr    = np.array(values).reshape(1, -1)
        arr_s  = scaler.transform(arr)

        pred  = int(model.predict(arr_s)[0])
        proba = model.predict_proba(arr_s)[0].tolist()

        return jsonify({
            "prediction":     pred,
            "label":          "Malignant" if pred == 1 else "Benign",
            "confidence":     round(max(proba) * 100, 2),
            "prob_benign":    round(proba[0] * 100, 2),
            "prob_malignant": round(proba[1] * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
