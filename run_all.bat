
@echo off
echo 🔁 Starting FGOE Project...

call .venv\Scripts\activate.bat

start "Flask API" cmd /k "cd app\api && set FLASK_APP=app.py && flask run"
start "React Frontend" cmd /k "cd client && npm start"

echo 🚀 Flask running on http://localhost:5000
echo 🌐 React app running on http://localhost:3000
