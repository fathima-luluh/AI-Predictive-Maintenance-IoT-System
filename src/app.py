from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("models/predictive_maintenance_model.pkl")

# Home route (test if API is running)
@app.route('/')
def home():
    return "🚀 Predictive Maintenance API is Running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract sensor values
        temperature = data['temperature']
        vibration = data['vibration']
        current = data['current']

        # Convert to model input format
        features = np.array([[temperature, vibration, current]])

        # Prediction
        prediction = model.predict(features)[0]

        # Convert result
        result = "FAILURE 🔴" if prediction == 1 else "NORMAL 🟢"

        return jsonify({
            "status": "success",
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Run server
if __name__ == "__main__":
    app.run(debug=True)