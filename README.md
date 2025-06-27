# NFL Field Goal Kicker Leaderboard

A Dockerized Python pipeline that ranks NFL field goal kickers using a logistic regression model to compute Field Goals Over Expected (FGOE).

## 🔧 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fg-kicker-leaderboard.git
cd fg-kicker-leaderboard
```

2. Build and run with Docker:
```bash
./run.sh
```

3. Output:
- `leaderboard.csv` will be created in the `/app` directory.

## 🧠 Methodology

- Filters data through Week 6 of 2018 Regular Season
- Predicts make probability from `attempt_yards` using logistic regression
- Computes `FGOE = Actual - Expected`
- Aggregates per kicker and ranks by total FGOE

## 💡 Skills Demonstrated

- Docker & containerized pipelines
- Model-based performance evaluation
- Python scripting with CLI support
- Reproducible research and modular design

## 📁 Files

- `app/leaderboard.py`: Main logic
- `app/field_goal_attempts.csv` / `app/players.csv`: Input data
- `leaderboard.csv`: Output leaderboard
- `Dockerfile`: Container definition
- `run.sh`: Local build & run script
