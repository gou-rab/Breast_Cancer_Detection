import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder=".", static_folder="static")

MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        values = [float(data.get(f, 0)) for f in FEATURES]
        arr    = np.array(values).reshape(1, -1)
        arr_s  = scaler.transform(arr)

        pred  = int(model.predict(arr_s)[0])
        proba = model.predict_proba(arr_s)[0].tolist()

        return jsonify({
            "prediction":   pred,
            "label":        "Malignant" if pred == 1 else "Benign",
            "confidence":   round(max(proba) * 100, 2),
            "prob_benign":  round(proba[0] * 100, 2),
            "prob_malignant": round(proba[1] * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
