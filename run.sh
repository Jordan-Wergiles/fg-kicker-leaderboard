#!/bin/bash
docker build -t fg-kicker-leaderboard .
docker run --rm -v $(pwd)/app:/app fg-kicker-leaderboard
