import joblib
from flask import Flask, request, jsonify
import numpy as np

try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado.")
    model = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado.'}), 500
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)