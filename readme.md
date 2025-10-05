
# Wine Quality Prediction API

API para predecir la calidad de vinos basándose en características químicas.

## Autor
**Jorge Guillermo Canas Magana**

## Dataset
Wine Quality Dataset - UCI Machine Learning Repository

## Modelo
- Algoritmo: Random Forest Classifier
- Accuracy: ~85%
- Features: 11 características químicas

## Endpoints

### GET /
Información general de la API

### GET /health
Health check del servicio

### GET /example
Ejemplo de datos de entrada

### POST /predict
Realiza una predicción

**Request:**
```json
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

#Response

{
  "quality": "low",
  "probability_low": 0.85,
  "probability_high": 0.15,
  "confidence": 0.85
}

Uso Local

pip install -r requirements.txt
python app.py

URL de Producción
https://TU-USUARIO.pythonanywhere.com

Fecha de Deployment: 05/10/2025

