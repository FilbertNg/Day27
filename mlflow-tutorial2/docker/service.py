from flask import Flask, request, jsonify
import mlflow.pyfunc
import numpy as np

app = Flask(__name__)

# Set the MLflow tracking URI to point to your MLflow server
mlflow.set_tracking_uri('http://mlflow-web:5000')

# Load the model from the MLflow model registry
model_uri = "models:/Diabetes Model/1"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Diabetes Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert the 'features' list to a 2D NumPy array (expected shape: [1, 10] for a single sample)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
