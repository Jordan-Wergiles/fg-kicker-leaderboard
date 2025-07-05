#!/bin/bash
docker build -t fg-kicker-leaderboard .
docker run --rm -v $(pwd)/app:/app fg-kicker-leaderboard
FLASK_APP=app/api/app.py flask run --host=0.0.0.0 --port=5000

