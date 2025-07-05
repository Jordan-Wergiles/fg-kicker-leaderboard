from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load preprocessing artifacts
scaler = joblib.load("../../models/nn_scaler.pkl")
input_columns = joblib.load("../../models/nn_input_columns.pkl")
calibrator = joblib.load("../../models/nn_iso_calibrator.pkl")

class FGOEModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize and load the model
model = FGOEModel(input_dim=len(input_columns))
model.load_state_dict(torch.load("../../models/model_nn.pt"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input JSON to feature vector
    input_df = {col: [data.get(col, 0)] for col in input_columns}
    input_array = np.array([input_df[col][0] for col in input_columns]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Predict and calibrate
    with torch.no_grad():
        raw_prob = model(input_tensor).item()
        fgoe_prob = float(calibrator.predict([raw_prob])[0])

    return jsonify({
        "fgoe_probability": round(fgoe_prob, 4),
        "raw_model_output": round(raw_prob, 4),
        "inputs": data
    })

if __name__ == "__main__":
    app.run(debug=True)
