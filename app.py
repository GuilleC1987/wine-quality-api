from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Nombres de las características
FEATURE_NAMES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

@app.route('/')
def home():
    """Endpoint principal con información de la API"""
    return jsonify({
        'message': 'Wine Quality Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'Información de la API',
            '/health': 'Health check',
            '/predict': 'Predicción de calidad (POST)',
            '/example': 'Ejemplo de datos de entrada'
        },
        'author': 'GUILLERMO_CANAS'  # ⚠️ CAMBIA ESTO
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/example')
def example():
    """Retorna un ejemplo de datos de entrada"""
    return jsonify({
        'example_input': {
            'fixed_acidity': 7.4,
            'volatile_acidity': 0.7,
            'citric_acid': 0.0,
            'residual_sugar': 1.9,
            'chlorides': 0.076,
            'free_sulfur_dioxide': 11.0,
            'total_sulfur_dioxide': 34.0,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4
        },
        'expected_output': {
            'quality': 'low',
            'probability_low': 0.85,
            'probability_high': 0.15
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para hacer predicciones
    
    Request body (JSON):
    {
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
    }
    """
    try:
        # Obtener datos del request
        data = request.get_json()
        
        # Validar que todos los campos estén presentes
        missing_fields = [field for field in FEATURE_NAMES if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing fields',
                'missing': missing_fields
            }), 400
        
        # Extraer features en el orden correcto
        features = [float(data[field]) for field in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        
        # Escalar features
        features_scaled = scaler.transform(features_array)
        
        # Hacer predicción
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Preparar respuesta
        quality = 'high' if prediction == 1 else 'low'
        
        return jsonify({
            'quality': quality,
            'probability_low': float(probabilities[0]),
            'probability_high': float(probabilities[1]),
            'confidence': float(max(probabilities)),
            'input_features': data
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Para desarrollo local
    app.run(debug=True, host='0.0.0.0', port=5000)