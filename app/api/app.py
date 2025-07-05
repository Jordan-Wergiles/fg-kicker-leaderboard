from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("best_model.pkl")
leaderboard = pd.read_csv("fg_leaderboard.csv")
scored = pd.read_csv("fgoe_per_attempt.csv")

@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    min_kicks = int(request.args.get("min_kicks", 0))
    top_n = int(request.args.get("top_n", 10))
    filtered = leaderboard[leaderboard["kicks"] >= min_kicks]
    sorted_lb = filtered.sort_values("fgoe_per_kick", ascending=False).head(top_n)
    return sorted_lb.to_dict(orient="records")

@app.route("/api/predict", methods=["POST"])
def predict_fgoe():
    data = request.json
    input_df = pd.DataFrame([data])
    pred_prob = model.predict_proba(input_df)[0][1]
    actual = int(data.get("make", 1))
    fgoe = actual - pred_prob
    return jsonify({
        "predicted_probability": round(pred_prob, 4),
        "fgoe": round(fgoe, 4)
    })

@app.route("/api/fgoe_trend/<player_id>", methods=["GET"])
def fgoe_trend(player_id):
    df = scored[scored["player_id"] == player_id]
    df = df.groupby("game_date").agg({
        "score": "mean"
    }).reset_index()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
