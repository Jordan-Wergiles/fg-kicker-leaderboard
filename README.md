# 🏈 NFL Field Goal Kicker Leaderboard

This project represents a full-stack Dockerized application for evaluating NFL field goal kickers using a neural network-based model. The backend computes Field Goals Over Expected (FGOE), and the frontend provides an interactive dashboard for predictions and leaderboard exploration.

---

## Project Structure

fg-kicker-leaderboard/
├── app/ # Backend Flask API and model files
│ ├── api/
│ │ └── app.py # Flask server with model serving
│ ├── models/ # Trained neural net, scaler, calibrator, input column names
│ └── leaderboard_notebook.ipynb # Jupyter analysis notebook which is how I determined the model I ultimately went with
├── client/ # React frontend dashboard
├── notebooks/ # (Optional) Mounted directory for Jupyter notebook access
├── Dockerfile.api # Dockerfile for Flask API backend
├── Dockerfile.frontend # Dockerfile for React frontend
├── docker-compose.yml # Coordinates all containers
├── requirements.txt # Python dependencies

## Quick Start (Docker)

### 1. Clone the repository
git clone https://github.com/yourusername/fg-kicker-leaderboard.git
cd fg-kicker-leaderboard

### 2. Build and start all services

docker compose up --build

This will spin up:
backend (Flask API on port 5000)
frontend (React app on port 3000)
notebook (Jupyter notebook on port 8888)

### 3. Access the application
Frontend app: http://localhost:3000

Backend API: http://localhost:5000

Jupyter Notebook: http://localhost:8888

No token is required to access Jupyter since I disabled token authentication in the Docker config.

## Web App Breakdown:

### FGOE Predictor Tab
Use the web app to enter kick data information.

Click "Predict FGOE" to receive a calibrated probability that the kick is made.

The model uses a trained neural network calibrated with isotonic regression.

### Leaderboard Tab
Use the web app to access the kicker leaderboard based on the calculated metric to determine the best kickers based on the regular season kick data provided.

Additional functionality has been built in the tab to search for players and sort by each respective column.

### Player Analysis Tab
Use the web app to access the raw fgoe data for each kick to dive deeper into a specific kicker with filters and compare to another kicker.

Additional functionality has been built in to aggregate the data and create simple visualizations.

## Skills Demonstrated

Docker & containerized ML workflows

Flask model deployment

Real-time neural network inference

React dashboards with dynamic charts

Jupyter notebook-based machine learning research and model deployment

Modular project structure

## Local Development (Optional)
If you prefer to run outside of Docker:

# Backend (from fg-kicker-leaderboard/app/api)
pip install -r ../../requirements.txt
python app.py

# Frontend (from fg-kicker-leaderboard/client)
npm install
npm start

If you have any questions please don't hesitate to reach out at jwergs18@gmail.com
