import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score
)

DATA_PATH   = "data.csv"
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
VIZ_DIR     = "static/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

BENIGN_COLOR    = "#4ECDC4"
MALIGNANT_COLOR = "#FF6B6B"
BG_COLOR        = "#0D1117"
PANEL_COLOR     = "#161B22"
TEXT_COLOR      = "#E6EDF3"
GRID_COLOR      = "#21262D"
ACCENT          = "#58A6FF"

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    PANEL_COLOR,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "text.color":        TEXT_COLOR,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "figure.dpi":        130,
    })

print("📂  Loading data …")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c or c == "id"], errors="ignore")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df = df.dropna()

FEATURES = [c for c in df.columns if c != "diagnosis"]
X = df[FEATURES]
y = df["diagnosis"]

print(f"   Samples : {len(df)}  |  Features : {len(FEATURES)}")
print(f"   Malignant : {y.sum()}  |  Benign : {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\n🤖  Training models …")
models = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}
best_name, best_model, best_acc = None, None, 0

for name, clf in models.items():
    clf.fit(X_train_s, y_train)
    cv  = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="accuracy").mean()
    acc = accuracy_score(y_test, clf.predict(X_test_s))
    print(f"   {name:<25}  CV={cv:.4f}  Test={acc:.4f}")
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, clf

print(f"\n✅  Best model → {best_name}  (accuracy={best_acc:.4f})")

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler,     SCALER_PATH)
print(f"💾  Saved  {MODEL_PATH}  &  {SCALER_PATH}")

set_dark_style()
y_pred      = best_model.predict(X_test_s)
y_prob      = best_model.predict_proba(X_test_s)[:, 1]

# 5-a  Confusion Matrix
print("\n📊  Generating visualizations …")
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    linewidths=1, linecolor=GRID_COLOR,
    annot_kws={"size": 18, "weight": "bold"},
    ax=ax
)
ax.set_xlabel("Predicted Label", labelpad=10)
ax.set_ylabel("True Label", labelpad=10)
ax.set_title("Confusion Matrix", pad=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{VIZ_DIR}/confusion_matrix.png", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

# 5-b  ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC = {roc_auc:.4f}")
ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight="bold", pad=14)
ax.legend(loc="lower right", framealpha=0.3)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{VIZ_DIR}/roc_curve.png", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

feat_df = (
    pd.DataFrame({"feature": FEATURES, "importance": importances})
    .sort_values("importance", ascending=True)
    .tail(15)
)
fig, ax = plt.subplots(figsize=(8, 6))
colors = [MALIGNANT_COLOR if v > feat_df["importance"].median() else BENIGN_COLOR
          for v in feat_df["importance"]]
bars = ax.barh(feat_df["feature"], feat_df["importance"], color=colors, height=0.65)
for bar, val in zip(bars, feat_df["importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8, color=TEXT_COLOR)
ax.set_xlabel("Importance Score")
ax.set_title("Top 15 Feature Importances", fontweight="bold", pad=14)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{VIZ_DIR}/feature_importance.png", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

top12 = (
    pd.DataFrame({"feature": FEATURES, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(12)["feature"].tolist()
)
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[top12 + ["diagnosis"]].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5,
    linecolor=GRID_COLOR, annot_kws={"size": 7},
    ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlation Heatmap (Top 12 Features)", fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(f"{VIZ_DIR}/correlation_heatmap.png", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

# 5-e  Class Distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts()
bars = ax.bar(
    ["Benign (0)", "Malignant (1)"],
    [counts[0], counts[1]],
    color=[BENIGN_COLOR, MALIGNANT_COLOR],
    width=0.5, edgecolor=GRID_COLOR
)
for bar, val in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
            str(val), ha="center", fontweight="bold", fontsize=13, color=TEXT_COLOR)
ax.set_ylabel("Count")
ax.set_title("Class Distribution", fontweight="bold", pad=14)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{VIZ_DIR}/class_distribution.png", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

print("   ✔  confusion_matrix.png")
print("   ✔  roc_curve.png")
print("   ✔  feature_importance.png")
print("   ✔  correlation_heatmap.png")
print("   ✔  class_distribution.png")

print("\n📋  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("🎉  Training complete!")
