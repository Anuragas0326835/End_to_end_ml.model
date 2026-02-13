from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Configuration
MODEL_FILE = "../gym_rf_model.pkl"
SCALER_FILE = "../scaler.pkl"
ENCODERS_FILE = "../encoders.pkl"

# Global variables to hold artifacts
model = None
scaler = None
encoders = None

def load_artifacts():
    """Loads model and preprocessors into global variables."""
    global model, scaler, encoders
    
    # Load Model
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    
    # Load Scaler
    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")

    # Load Encoders
    if os.path.exists(ENCODERS_FILE):
        with open(ENCODERS_FILE, 'rb') as f:
            encoders = pickle.load(f)
        print("Encoders loaded successfully.")

# Load artifacts immediately
load_artifacts()

@app.route('/', methods=['GET'])
def home():
    return "Gym Attendance Prediction API is running! Send a POST request to /predict to use the model."

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # 1. Get JSON data
        data = request.get_json()
        
        # 2. Create DataFrame
        input_data = {
            'Date': [data['Date']],
            'Sleep_Hours': [data['Sleep_Hours']],
            'Mood': [data['Mood']],
            'Work_Load': [data['Work_Load']],
            'Weather': [data['Weather']]
        }
        df = pd.DataFrame(input_data)

        # 3. Feature Engineering (Must match train.py logic)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day_Of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Is_Weekend'] = df['Day_Of_Week'].apply(lambda x: 1 if x >= 5 else 0)
        df.drop('Date', axis=1, inplace=True)

        # 4. Preprocessing (Apply saved Encoders and Scaler)
        if encoders:
            for col, le in encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])
        
        if scaler:
            numerical_cols = ['Sleep_Hours']
            df[numerical_cols] = scaler.transform(df[numerical_cols])

        # 5. Prediction
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df).max()
        
        result = "Go to Gym" if prediction == 1 else "Skip Gym"
        return jsonify({"prediction": result, "probability": float(prob), "raw_output": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)