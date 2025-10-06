ffrom flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURE_NAMES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

VALID_RANGES = {
    'fixed_acidity': (4.6, 15.9),
    'volatile_acidity': (0.12, 1.58),
    'citric_acid': (0.0, 1.0),
    'residual_sugar': (0.9, 15.5),
    'chlorides': (0.012, 0.611),
    'free_sulfur_dioxide': (1.0, 72.0),
    'total_sulfur_dioxide': (6.0, 289.0),
    'density': (0.990, 1.004),
    'pH': (2.74, 4.01),
    'sulphates': (0.33, 2.0),
    'alcohol': (8.0, 14.9)
}

model = None
scaler = None
load_errors = []

def _safe_load(path, label):
    try:
        obj = joblib.load(path)
        return obj, None
    except Exception as e:
        return None, f"{label}: {e}"

scaler, err_s = _safe_load(SCALER_PATH, "scaler")
if err_s:
    load_errors.append(err_s)

model, err_m = _safe_load(MODEL_PATH, "model")
if err_m:
    load_errors.append(err_m)

@app.get("/")
def home():
    return jsonify({
        "message": "Wine Quality Prediction API",
        "version": "1.0",
        "endpoints": {
            "/": "Información de la API (GET)",
            "/health": "Health check (GET)",
            "/example": "Ejemplo de datos (GET)",
            "/predict": "Predicción de calidad (POST, JSON)",
            "/stats": "Estadísticas del modelo (GET)",
            "/form": "Formulario web (GET)"
        },
        "author": "GUILLERMO_CANAS"
    })

@app.get("/health")
def health():
    ok = (model is not None) and (scaler is not None) and (not load_errors)
    return jsonify({
        "status": "healthy" if ok else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "errors": load_errors,
        "cwd": os.getcwd(),
        "base_dir": BASE_DIR
    }), (200 if ok else 500)

@app.get("/example")
def example():
    return jsonify({
        "example_input": {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        },
        "valid_ranges": VALID_RANGES
    })

@app.route("/stats")
def stats():
    return jsonify({
        "model_type": "Random Forest Classifier",
        "n_features": len(FEATURE_NAMES),
        "accuracy": 0.85,
        "training_samples": 1279,
        "test_samples": 320,
        "features": FEATURE_NAMES
    })

@app.get("/form")
def form_page():
    return render_template("index.html")

@app.post("/predict")
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                "error": "Modelo o scaler no cargados",
                "details": load_errors
            }), 500

        data = request.get_json(silent=True) or {}
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({
                "error": "Missing fields",
                "missing": missing,
                "required_order": FEATURE_NAMES,
                "valid_ranges": VALID_RANGES
            }), 400

        try:
            features = [float(data[f]) for f in FEATURE_NAMES]
        except Exception as conv_err:
            return jsonify({
                "error": "Type conversion error",
                "details": str(conv_err)
            }), 400

        X = np.array(features, dtype=float).reshape(1, -1)

        try:
            Xs = scaler.transform(X)
        except Exception as scale_err:
            return jsonify({
                "error": "Scaler transform failed",
                "details": str(scale_err)
            }), 500

        try:
            pred = model.predict(Xs)[0]
        except Exception as pred_err:
            return jsonify({
                "error": "Model predict failed",
                "details": str(pred_err)
            }), 500

        prob_low = prob_high = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xs)[0]
                if len(proba) >= 2:
                    prob_low = float(proba[0])
                    prob_high = float(proba[1])
        except Exception:
            pass

        quality = "high" if int(pred) == 1 else "low"
        confidence = None
        if prob_low is not None and prob_high is not None:
            confidence = float(max(prob_low, prob_high))

        return jsonify({
            "quality": quality,
            "probability_low": prob_low,
            "probability_high": prob_high,
            "confidence": confidence,
            "input_features": {k: data[k] for k in FEATURE_NAMES}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
