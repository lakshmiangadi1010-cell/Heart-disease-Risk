from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart_disease_risk_dataset_earlymed-selected-columns.csv"
MODEL_PATH = BASE_DIR / "heart_risk_model.joblib"

FEATURES = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "High_BP",
    "High_Cholesterol",
]

LABELS = {
    "Chest_Pain": "Chest pain or pressure",
    "Shortness_of_Breath": "Shortness of breath",
    "Fatigue": "Unusual fatigue",
    "Palpitations": "Heart palpitations",
    "Dizziness": "Dizziness or lightheadedness",
    "Swelling": "Swelling in legs or ankles",
    "Pain_Arms_Jaw_Back": "Pain in arm, jaw, or back",
    "Cold_Sweats_Nausea": "Cold sweats or nausea",
    "High_BP": "High blood pressure",
    "High_Cholesterol": "High cholesterol",
}

app = Flask(__name__)


def train_model():
    data = pd.read_csv(DATA_PATH)
    missing = [feature for feature in FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected features: {', '.join(missing)}")

    features = data[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1)
    has_real_target = "Heart_Risk" in data.columns
    if has_real_target:
        target = pd.to_numeric(data["Heart_Risk"], errors="coerce").fillna(0).astype(int)
        model_source = "trained target labels"
    else:
        # The supplied selected-columns file has no target. This proxy keeps the
        # demo runnable, but must be replaced with clinical labels for real use.
        weights = np.array([2.2, 1.7, 1.0, 1.2, 0.8, 1.0, 2.0, 1.8, 1.4, 1.1])
        proxy_score = features.to_numpy() @ weights
        target = pd.Series((proxy_score >= 5.0).astype(int), index=features.index)
        model_source = "proxy labels (no Heart_Risk column supplied)"

    if target.nunique() < 2:
        raise ValueError("Training data must contain both low-risk and high-risk labels")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    test_predictions = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
        "precision": round(float(precision_score(y_test, test_predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, test_predictions, zero_division=0)), 4),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "has_real_target": has_real_target,
    }
    model.fit(features, target)
    joblib.dump(model, MODEL_PATH)
    return model, model_source, metrics


MODEL, MODEL_SOURCE, MODEL_METRICS = train_model()


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES, labels=LABELS)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_source": MODEL_SOURCE,
            "features": len(FEATURES),
            "training": MODEL_METRICS,
            "production_ready": MODEL_METRICS["has_real_target"],
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = []
        for feature in FEATURES:
            value = payload.get(feature, False)
            if isinstance(value, bool):
                values.append(float(value))
            elif value in (0, 1, 0.0, 1.0):
                values.append(float(value))
            else:
                raise ValueError(f"{feature} must be a boolean or 0/1")
        row = pd.DataFrame([values], columns=FEATURES)
        probability = float(MODEL.predict_proba(row)[0][1])
    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid assessment data: {error}"}), 400

    elevated = probability >= 0.5
    return jsonify(
        {
            "risk": "elevated" if elevated else "lower",
            "probability": round(probability * 100, 1),
            "model_source": MODEL_SOURCE,
            "message": (
                "Several indicators suggest elevated risk. Seek prompt medical advice."
                if elevated
                else "Fewer indicators were detected, but this does not rule out heart disease."
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)