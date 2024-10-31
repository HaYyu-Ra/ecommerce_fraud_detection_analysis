from flask import Flask, request, jsonify, render_template
import logging
import pandas as pd
import joblib
import os

# Set up logging
logging.basicConfig(
    filename='fraud_detection.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__, 
            template_folder="C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/ecommerce_fraud_detection_system/fraud_detection_api/templates")

# Load pre-trained model (ensure the correct path)
model_path = os.path.join(
    "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/ecommerce_fraud_detection_system/fraud_detection_api/data",
    "fraud_detection_model.pkl"
)

# Load the model in a try-except block
model = None
try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Model file not found at {model_path}. Please check the path.")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': "Model not loaded. Please check the model path."}), 500

    try:
        # Gather input data from the JSON request
        data = request.json
        logging.info(f"Received data: {data}")  # Log the received data

        # Convert input data to DataFrame
        df = pd.DataFrame([data])

        # Check if all necessary columns are present
        expected_columns = ['user_id', 'amount', 'transaction_type']
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return jsonify({'error': f"Missing columns: {missing_cols}"}), 400

        # Make prediction
        prediction = model.predict(df)[0]
        logging.info(f"Prediction: {prediction} for data: {data}")

        return jsonify({'prediction': int(prediction)})

    except ValueError as ve:
        logging.error(f"ValueError in prediction: {str(ve)}")
        return jsonify({'error': f"Input value error: {str(ve)}"}), 400
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app in debug mode for better error messages during development
    app.run(host='0.0.0.0', port=5000, debug=True)

