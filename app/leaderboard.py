import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse

def load_data(fg_path, players_path):
    fg = pd.read_csv(fg_path)
    players = pd.read_csv(players_path)

    # Filter to regular season and weeks <= 6
    fg = fg[(fg['season_type'] == 'Reg') & (fg['week'] <= 6)]
    fg['make'] = fg['field_goal_result'].map({'Made': 1, 'Missed': 0})

    return fg, players

def fit_model(fg):
    model = LogisticRegression()
    model.fit(fg[['attempt_yards']], fg['make'])
    fg['expected_make_prob'] = model.predict_proba(fg[['attempt_yards']])[:, 1]
    fg['fgoe'] = fg['make'] - fg['expected_make_prob']
    return fg, model

def create_leaderboard(fg, players, min_attempts=10):
    kicker_stats = fg.groupby('player_id').agg(
        attempts=('make', 'count'),
        fgoe_total=('fgoe', 'sum'),
        fgoe_per_attempt=('fgoe', 'mean')
    ).reset_index()

    kicker_stats = kicker_stats[kicker_stats['attempts'] >= min_attempts].copy()
    kicker_stats['rating'] = kicker_stats['fgoe_total']
    kicker_stats['rank'] = kicker_stats['rating'].rank(method='min', ascending=False).astype(int)

    leaderboard = kicker_stats.merge(players[['player_id', 'player_name']], on='player_id')
    return leaderboard[['player_id', 'player_name', 'rating', 'rank']].sort_values(by='rank')

def main():
    parser = argparse.ArgumentParser(description="Generate field goal kicker leaderboard.")
    parser.add_argument('--fg_file', type=str, default='field_goal_attempts.csv')
    parser.add_argument('--players_file', type=str, default='kickers.csv')
    parser.add_argument('--output', type=str, default='leaderboard.csv')
    parser.add_argument('--min_attempts', type=int, default=10)
    args = parser.parse_args()

    fg, players = load_data(args.fg_file, args.players_file)
    fg, model = fit_model(fg)
    leaderboard = create_leaderboard(fg, players, min_attempts=args.min_attempts)
    leaderboard.to_csv(args.output, index=False)
    print(f"Leaderboard saved to {args.output}")

if __name__ == '__main__':
    main()
