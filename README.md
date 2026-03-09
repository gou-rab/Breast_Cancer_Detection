# 🎗️ Breast Cancer Detection — AI Web App

An AI-powered breast cancer detection web application built with Flask and Random Forest classification, trained on the Wisconsin Breast Cancer Dataset.


> **AI-powered breast cancer classification web app** trained on the Wisconsin Breast Cancer Dataset.  
> Predicts **Malignant** or **Benign** tumors with **96.49% accuracy** using Random Forest.



![Accuracy](https://img.shields.io/badge/Accuracy-96.49%25-brightgreen?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-~0.99-blue?style=flat-square)
![Samples](https://img.shields.io/badge/Training_Samples-569-orange?style=flat-square)
![Features](https://img.shields.io/badge/Features-30-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

</div>

---

## 📸 App Preview


---

## 🧠 About The Project

Breast cancer is the **most common cancer** among women worldwide. Early detection dramatically improves survival rates. This project leverages **machine learning** to assist in identifying whether a tumor is benign or malignant based on 30 digitized features computed from a fine needle aspirate (FNA) of a breast mass.

The model analyzes properties like:
- **Radius**, **Texture**, **Perimeter**, **Area** of the tumor
- **Smoothness**, **Compactness**, **Concavity** measurements
- **Mean**, **Standard Error**, and **Worst** values for each feature

> ⚠️ **This is a research/educational tool — not a clinical diagnostic system.**

---

## 📊 Model Performance

<div align="center">

| Metric | Benign | Malignant | Overall |
|:------:|:------:|:---------:|:-------:|
| **Precision** | 95% | 100% | 97% |
| **Recall** | 100% | 90% | 96% |
| **F1-Score** | 97% | 95% | 96% |
| **Support** | 72 | 42 | 114 |

| 🎯 Accuracy | 📈 AUC Score | 🔁 CV Score | 🌲 Estimators |
|:-----------:|:------------:|:-----------:|:-------------:|
| **96.49%** | **~0.99** | **95.82%** | **200 trees** |

</div>

---

## 📁 Project Structure

```
🗂️ breast-cancer-detection/
│
├── 📄 app.py                     ← Flask web server & prediction API
├── 🧠 train.py                   ← Model training + chart generation
├── 🌐 index.html                 ← Responsive frontend (mobile + desktop)
├── 📦 model.pkl                  ← Trained Random Forest model
├── ⚖️  scaler.pkl                 ← StandardScaler for feature normalization
├── 📊 data.csv                   ← Wisconsin Breast Cancer Dataset (569 rows)
├── 📋 requirements.txt           ← Python dependencies
├── 🚀 render.yaml                ← Render.com deployment configuration
├── 🚫 .gitignore
├── 📖 README.md
│
└── 📂 static/
    └── 📂 visualizations/
        ├── 🖼️ confusion_matrix.png
        ├── 🖼️ roc_curve.png
        ├── 🖼️ feature_importance.png
        ├── 🖼️ correlation_heatmap.png
        └── 🖼️ class_distribution.png
```

---

## ✨ Features

- 🔬 **Real-time AI Prediction** — instant Malignant / Benign result with confidence %
- 📊 **5 Model Visualizations** — Confusion Matrix, ROC Curve, Feature Importance, Correlation Heatmap, Class Distribution
- 🧪 **Sample Data Buttons** — pre-fill with a real Benign or Malignant case instantly
- 📱 **Fully Responsive** — works perfectly on mobile phones, tablets, and laptops
- 🌙 **Dark Medical UI** — clean dark theme with color-coded result cards
- ⚡ **Fast Inference** — sub-second prediction via optimized Flask API
- 🛡️ **Input Validation** — graceful error handling with toast notifications

---

## 🔧 Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:------|:----------:|:--------|
| 🎨 **Frontend** | HTML5, CSS3, Vanilla JS | Responsive UI, form handling, result display |
| ⚙️ **Backend** | Python 3.11, Flask 3.0 | REST API, routing, model serving |
| 🤖 **ML Model** | Scikit-learn, Random Forest | Breast cancer classification |
| 📐 **Preprocessing** | StandardScaler, Pandas | Feature normalization & data cleaning |
| 📊 **Visualization** | Matplotlib, Seaborn | Training charts & model diagnostics |
| 🚀 **Server** | Gunicorn | Production WSGI server |
| ☁️ **Hosting** | Render.com | Cloud deployment via render.yaml |

</div>

---

## 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/breast-cancer-detection.git
cd breast-cancer-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model & generate visualizations
python train.py

# 4. Start the Flask app
python app.py
```

🌐 Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 📡 API Reference

### `POST /predict`

Send 30 feature values, receive a diagnosis.

**Request Body** (`application/json`):
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "...": "..."
}
```

**Response**:
```json
{
  "prediction": 1,
  "label": "Malignant",
  "confidence": 98.5,
  "prob_benign": 1.5,
  "prob_malignant": 98.5
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## 📈 Visualizations

| Chart | Description |
|:------|:------------|
| 🟦 **Confusion Matrix** | True vs predicted labels across test set |
| 📉 **ROC Curve** | Trade-off between sensitivity and specificity (AUC ~0.99) |
| 🌲 **Feature Importance** | Top 15 most influential features from Random Forest |
| 🔥 **Correlation Heatmap** | Feature-to-feature correlations for top 12 features |
| 📊 **Class Distribution** | Benign (357) vs Malignant (212) sample counts |

---

## 🗃️ Dataset

**Wisconsin Breast Cancer Dataset (WBCD)**

- 📦 **569** patient samples
- 🔢 **30** numeric features per sample
- 🎯 **2** classes: Benign (B) → `0`, Malignant (M) → `1`
- 📐 Features computed from digitized images of FNA of breast mass
- 🏛️ Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

## ⚠️ Medical Disclaimer

> This application is built **strictly for educational and research purposes**.  
> It is **NOT** intended to be used as a substitute for professional medical advice, diagnosis, or treatment.  
> Always seek the guidance of a **qualified healthcare provider** with any questions regarding a medical condition.

---

<div align="center">

Made with ❤️ using Python, Flask & Scikit-learn

⭐ **Star this repo if you found it helpful!**

</div>
