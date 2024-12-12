import pandas as pd
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load models
with open("app/model/rf_model.pkl", "rb") as rf_file:
    rf_model = pickle.load(rf_file)

with open("app/model/xgb_model.pkl", "rb") as xgb_file:
    xgb_model = pickle.load(xgb_file)

app = Flask(__name__)

# Assuming `X_train.columns` has feature names
FEATURE_NAMES = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'DiffWalk', 'Sex', 'Normalized_BMI', 'Normalized_Age', 'BMI_Category', 'Log_MentHlth', 'Log_PhysHlth', 'HighBP_HeartDisease', 'PhysActivity_BMI', 'Socioeconomics_NoDoc']

#curl -X POST -H "Content-Type: application/json" \
#-d '{"features": [1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 0, 1, 0.5, 1.2, 3, 0, 0, 0, 0, 0]}' \
#http://localhost:5050/predict

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Convert to DataFrame with feature names
        input_df = pd.DataFrame(features, columns=FEATURE_NAMES)

        # Get predictions
        rf_pred = rf_model.predict_proba(input_df)[:, 1]
        xgb_pred = xgb_model.predict_proba(input_df)[:, 1]

        # Ensemble prediction
        rf_weight = 0.3
        xgb_weight = 0.7
        ensemble_pred = (rf_pred * rf_weight) + (xgb_pred * xgb_weight)

        # Convert predictions to Python-native types
        response = {
            "rf_prediction": float(rf_pred[0]),
            "xgb_prediction": float(xgb_pred[0]),
            "ensemble_prediction": float(ensemble_pred[0]),
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
