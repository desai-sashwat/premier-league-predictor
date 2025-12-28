"""
Utility functions for Premier League Predictor
"""

from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_title_race(predictions: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create visualization of title race probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 6 win probability
    top6 = predictions.head(6)

    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top6)))
    bars = ax1.barh(top6['team'], top6['prob_win_league'], color=colors)
    ax1.set_xlabel('Win Probability (%)')
    ax1.set_title('Title Race - Win Probability')
    ax1.invert_yaxis()

    for bar, prob in zip(bars, top6['prob_win_league']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{prob:.1f}%', va='center', fontsize=10)

    # Projected points with confidence intervals
    ax2 = axes[1]
    y_pos = np.arange(len(top6))

    ax2.barh(y_pos, top6['predicted_points'], color='steelblue', alpha=0.7, label='Predicted')

    if 'points_90_ci_low' in top6.columns:
        xerr = [
            top6['predicted_points'] - top6['points_90_ci_low'],
            top6['points_90_ci_high'] - top6['predicted_points']
        ]
        ax2.errorbar(top6['predicted_points'], y_pos, xerr=xerr,
                     fmt='none', color='black', capsize=3, label='90% CI')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top6['team'])
    ax2.set_xlabel('Projected Points')
    ax2.set_title('Projected Final Points (with 90% CI)')
    ax2.invert_yaxis()
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_prediction_history(history: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot how predictions have changed over gameweeks.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    teams = history['predicted_winner'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(teams)))
    color_map = dict(zip(teams, colors))

    for team in teams:
        team_data = history[history['predicted_winner'] == team]
        ax.scatter(team_data['gameweek'], [team] * len(team_data),
                   s=100, c=[color_map[team]], label=team, marker='o')

    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Predicted Winner')
    ax.set_title('Title Race Predictions Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_feature_importance(importance: pd.Series, top_n: int = 15,
                            save_path: Optional[str] = None):
    """
    Plot feature importance from the model.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = importance.head(top_n)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    bars = ax.barh(range(len(top_features)), top_features.values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def calculate_elo_ratings(fixtures: pd.DataFrame, k_factor: float = 32,
                          initial_rating: float = 1500) -> Dict[str, float]:
    """
    Calculate Elo ratings for teams based on match results.
    Can be used as an additional feature.
    """
    ratings = {}

    for _, match in fixtures.iterrows():
        home = match.get('home_team')
        away = match.get('away_team')
        score = match.get('score', '')

        if not home or not away or not score:
            continue

        # Initialize ratings
        if home not in ratings:
            ratings[home] = initial_rating
        if away not in ratings:
            ratings[away] = initial_rating

        try:
            sep = '–' if '–' in score else '-'
            home_goals, away_goals = map(int, score.split(sep))
        except:
            continue

        # Calculate expected scores
        exp_home = 1 / (1 + 10 ** ((ratings[away] - ratings[home]) / 400))
        exp_away = 1 - exp_home

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if home_goals > away_goals:
            actual_home, actual_away = 1, 0
        elif home_goals < away_goals:
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5

        # Update ratings
        ratings[home] += k_factor * (actual_home - exp_home)
        ratings[away] += k_factor * (actual_away - exp_away)

    return ratings


def generate_report(predictions: pd.DataFrame, gameweek: int,
                    save_path: Optional[str] = None) -> str:
    """
    Generate a text report of predictions.
    """
    report = []
    report.append("=" * 60)
    report.append(f"PREMIER LEAGUE 2024/25 - GAMEWEEK {gameweek} PREDICTIONS")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 60)

    # Title prediction
    winner = predictions.iloc[0]
    report.append(f"\n🏆 PREDICTED CHAMPION: {winner['team']}")
    report.append(f"   Projected Points: {winner['predicted_points']:.1f}")
    if 'prob_win_league' in winner:
        report.append(f"   Win Probability: {winner['prob_win_league']:.1f}%")

    # Top 4
    report.append("\n📊 PREDICTED TOP 4:")
    for i, row in predictions.head(4).iterrows():
        prob = f"({row['prob_top_4']:.1f}%)" if 'prob_top_4' in row else ""
        report.append(f"   {i + 1}. {row['team']} - {row['predicted_points']:.1f} pts {prob}")

    # Champions League places (5-6)
    report.append("\n⚽ EUROPA LEAGUE:")
    for i, row in predictions.iloc[4:6].iterrows():
        report.append(f"   {i + 1}. {row['team']} - {row['predicted_points']:.1f} pts")

    # Relegation battle
    if 'prob_relegation' in predictions.columns:
        report.append("\n⚠️ RELEGATION BATTLE:")
        relegation = predictions.nlargest(5, 'prob_relegation')
        for _, row in relegation.iterrows():
            report.append(f"   {row['team']} - {row['prob_relegation']:.1f}% risk")

    report.append("\n" + "=" * 60)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")

    return report_text


def validate_predictions(predictions: pd.DataFrame, actual_standings: pd.DataFrame) -> Dict:
    """
    Validate predictions against actual standings (for backtesting).
    """
    # Merge on team name
    merged = predictions.merge(
        actual_standings[['team', 'points']],
        on='team',
        suffixes=('_pred', '_actual')
    )

    # Calculate metrics
    mae = np.abs(merged['predicted_points'] - merged['points_actual']).mean()
    rmse = np.sqrt(((merged['predicted_points'] - merged['points_actual']) ** 2).mean())

    # Position accuracy
    pred_order = predictions['team'].tolist()
    actual_order = actual_standings.sort_values('points', ascending=False)['team'].tolist()

    top4_correct = len(set(pred_order[:4]) & set(actual_order[:4]))
    winner_correct = pred_order[0] == actual_order[0]

    return {
        'mae': mae,
        'rmse': rmse,
        'top4_accuracy': top4_correct / 4,
        'winner_correct': winner_correct,
        'kendall_tau': calculate_kendall_tau(pred_order, actual_order)
    }


def calculate_kendall_tau(pred_order: List[str], actual_order: List[str]) -> float:
    """Calculate Kendall's Tau correlation for rankings."""
    from scipy.stats import kendalltau

    # Convert to ranks
    pred_ranks = {team: i for i, team in enumerate(pred_order)}
    actual_ranks = {team: i for i, team in enumerate(actual_order)}

    common_teams = set(pred_ranks.keys()) & set(actual_ranks.keys())

    pred_r = [pred_ranks[t] for t in common_teams]
    actual_r = [actual_ranks[t] for t in common_teams]

    tau, _ = kendalltau(pred_r, actual_r)
    return tau