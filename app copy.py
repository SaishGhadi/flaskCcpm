from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(name)
CORS(app)  # Enable CORS for Laravel access

model = joblib.load('cc_price_model.pkl')

@app.route('/')
def home():
    return "Carbon Credit Price Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)
        return jsonify({'predicted_price': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if name == 'main':
    app.run(debug=True)