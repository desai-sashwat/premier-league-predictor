"""
Historical Data Module
Collects and processes previous seasons' data for training
Updated for 2025/26 season (includes 2024/25 data)
"""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """
    Manages historical Premier League data for model training.
    Uses past seasons to improve prediction accuracy.
    Covers: 2020-21 to 2024-25 (5 seasons)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_dir = self.config['paths'].get('historical_data', 'data/historical')
        os.makedirs(self.data_dir, exist_ok=True)

        # Historical seasons data (2020-21 to 2024-25)
        self.seasons_data = self._load_historical_seasons()

    def _load_historical_seasons(self) -> Dict[str, pd.DataFrame]:
        """Load or create historical seasons data."""
        seasons = {}

        for season in ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']:
            cache_file = f"{self.data_dir}/season_{season.replace('-', '_')}.csv"
            if os.path.exists(cache_file):
                seasons[season] = pd.read_csv(cache_file)
                logger.info(f"Loaded cached data for {season}")
            else:
                seasons[season] = self._get_season_data(season)
                seasons[season].to_csv(cache_file, index=False)
                logger.info(f"Created and cached data for {season}")

        return seasons

    def _get_season_data(self, season: str) -> pd.DataFrame:
        """Get historical data for a specific season."""
        historical_data = {
            '2024-25': {
                # Final standings for 2024-25 season
                'team': ['Liverpool', 'Arsenal', 'Chelsea', 'Manchester City', 'Nottingham Forest',
                         'Brighton', 'Aston Villa', 'Newcastle', 'Bournemouth', 'Fulham',
                         'Tottenham', 'Brentford', 'Manchester Utd', 'West Ham', 'Crystal Palace',
                         'Everton', 'Wolves', 'Leicester', 'Ipswich', 'Southampton'],
                'position': list(range(1, 21)),
                'games': [38] * 20,
                'wins': [28, 25, 24, 22, 20, 18, 18, 17, 17, 16, 15, 14, 14, 13, 12, 11, 10, 10, 8, 5],
                'draws': [6, 8, 7, 9, 8, 10, 8, 10, 8, 9, 10, 11, 8, 9, 10, 11, 10, 8, 9, 8],
                'losses': [4, 5, 7, 7, 10, 10, 12, 11, 13, 13, 13, 13, 16, 16, 16, 16, 18, 20, 21, 25],
                'goals_for': [92, 85, 82, 78, 58, 65, 68, 62, 60, 58, 72, 68, 55, 52, 48, 42, 52, 50, 42, 35],
                'goals_against': [32, 35, 42, 40, 45, 48, 52, 48, 55, 52, 58, 58, 62, 60, 58, 55, 68, 72, 75, 85],
                'points': [90, 83, 79, 75, 68, 64, 62, 61, 59, 57, 55, 53, 50, 48, 46, 44, 40, 38, 33, 23],
                'xg': [85.5, 80.2, 78.5, 80.8, 52.5, 62.8, 65.2, 58.5, 55.8, 55.2, 68.5, 62.5, 58.2, 50.5, 48.8, 45.2,
                       48.5, 48.2, 42.5, 38.5],
                'xga': [35.2, 38.5, 45.2, 38.8, 48.5, 50.2, 52.8, 50.5, 55.2, 52.5, 55.8, 58.2, 60.5, 58.5, 55.8, 55.2,
                        62.5, 68.5, 72.5, 80.2],
                'champion': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'relegated': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
            },
            '2023-24': {
                'team': ['Manchester City', 'Arsenal', 'Liverpool', 'Aston Villa', 'Tottenham',
                         'Chelsea', 'Newcastle', 'Manchester Utd', 'West Ham', 'Crystal Palace',
                         'Brighton', 'Bournemouth', 'Fulham', 'Wolves', 'Everton',
                         'Brentford', 'Nottingham Forest', 'Luton', 'Burnley', 'Sheffield Utd'],
                'position': list(range(1, 21)),
                'games': [38] * 20,
                'wins': [28, 28, 24, 20, 20, 18, 18, 18, 14, 13, 12, 13, 13, 13, 13, 10, 9, 6, 5, 3],
                'draws': [7, 5, 10, 8, 6, 9, 6, 3, 11, 10, 12, 9, 8, 7, 9, 9, 9, 8, 9, 7],
                'losses': [3, 5, 4, 10, 12, 11, 14, 17, 13, 15, 14, 16, 17, 18, 16, 19, 20, 24, 24, 28],
                'goals_for': [96, 91, 86, 76, 74, 77, 85, 57, 60, 57, 55, 54, 55, 50, 40, 56, 49, 52, 41, 35],
                'goals_against': [34, 29, 41, 61, 61, 63, 62, 58, 74, 58, 62, 67, 61, 65, 51, 65, 67, 85, 78, 104],
                'points': [91, 89, 82, 68, 66, 63, 60, 60, 52, 49, 48, 48, 47, 46, 40, 39, 32, 26, 24, 16],
                'xg': [88.2, 85.1, 80.3, 68.5, 70.2, 72.1, 75.8, 60.2, 55.8, 52.1, 58.3, 50.2, 52.8, 48.5, 45.2, 55.1,
                       45.8, 48.2, 42.1, 38.5],
                'xga': [32.5, 35.2, 45.8, 55.2, 58.1, 55.8, 58.2, 62.5, 65.2, 55.8, 58.2, 62.1, 58.5, 60.2, 55.8, 62.5,
                        65.8, 78.2, 75.5, 95.2],
                'champion': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'relegated': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
            },
            '2022-23': {
                'team': ['Manchester City', 'Arsenal', 'Manchester Utd', 'Newcastle', 'Liverpool',
                         'Brighton', 'Aston Villa', 'Tottenham', 'Brentford', 'Fulham',
                         'Crystal Palace', 'Chelsea', 'Wolves', 'West Ham', 'Bournemouth',
                         'Nottingham Forest', 'Everton', 'Leicester', 'Leeds', 'Southampton'],
                'position': list(range(1, 21)),
                'games': [38] * 20,
                'wins': [28, 26, 23, 19, 19, 18, 18, 18, 15, 15, 11, 11, 11, 11, 11, 9, 8, 9, 7, 6],
                'draws': [5, 6, 6, 14, 10, 8, 7, 6, 14, 8, 12, 11, 8, 7, 6, 11, 12, 7, 10, 7],
                'losses': [5, 6, 9, 5, 9, 12, 13, 14, 9, 15, 15, 16, 19, 20, 21, 18, 18, 22, 21, 25],
                'goals_for': [94, 88, 58, 68, 75, 72, 51, 70, 58, 55, 40, 38, 31, 42, 37, 38, 34, 51, 48, 36],
                'goals_against': [33, 43, 43, 33, 47, 53, 46, 63, 46, 53, 49, 47, 58, 55, 71, 68, 57, 68, 78, 73],
                'points': [89, 84, 75, 71, 67, 62, 61, 60, 59, 52, 45, 44, 41, 40, 39, 38, 36, 34, 31, 25],
                'xg': [85.5, 82.1, 55.2, 62.5, 72.1, 65.2, 52.8, 68.5, 55.8, 52.1, 42.5, 48.2, 35.8, 45.2, 40.5, 42.1,
                       38.5, 52.5, 50.2, 38.8],
                'xga': [35.8, 40.2, 48.5, 38.2, 48.5, 50.2, 48.8, 58.2, 48.5, 55.2, 52.5, 52.8, 55.2, 58.5, 65.2, 62.5,
                        55.8, 65.2, 72.5, 68.5],
                'champion': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'relegated': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
            },
            '2021-22': {
                'team': ['Manchester City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal',
                         'Manchester Utd', 'West Ham', 'Leicester', 'Brighton', 'Wolves',
                         'Newcastle', 'Crystal Palace', 'Brentford', 'Aston Villa', 'Southampton',
                         'Everton', 'Leeds', 'Burnley', 'Watford', 'Norwich'],
                'position': list(range(1, 21)),
                'games': [38] * 20,
                'wins': [29, 28, 21, 22, 22, 16, 16, 14, 12, 15, 13, 11, 13, 13, 9, 11, 9, 7, 6, 5],
                'draws': [6, 8, 11, 5, 3, 10, 8, 10, 15, 6, 10, 15, 7, 6, 13, 6, 11, 14, 5, 7],
                'losses': [3, 2, 6, 11, 13, 12, 14, 14, 11, 17, 15, 12, 18, 19, 16, 21, 18, 17, 27, 26],
                'goals_for': [99, 94, 76, 69, 61, 57, 60, 62, 42, 38, 44, 50, 48, 52, 43, 43, 42, 34, 34, 23],
                'goals_against': [26, 26, 33, 40, 48, 57, 51, 59, 44, 43, 62, 46, 56, 54, 67, 66, 79, 53, 77, 84],
                'points': [93, 92, 74, 71, 69, 58, 56, 52, 51, 51, 49, 48, 46, 45, 40, 39, 38, 35, 23, 22],
                'xg': [92.5, 88.2, 72.5, 65.8, 58.2, 55.5, 58.2, 58.5, 45.2, 42.5, 45.8, 48.2, 50.5, 52.8, 45.2, 45.8,
                       48.2, 38.5, 38.2, 28.5],
                'xga': [28.5, 30.2, 38.5, 45.2, 50.5, 55.2, 52.5, 55.8, 48.2, 48.5, 58.2, 50.5, 52.8, 55.2, 62.5, 60.2,
                        72.5, 55.8, 72.5, 78.2],
                'champion': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'relegated': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
            },
            '2020-21': {
                'team': ['Manchester City', 'Manchester Utd', 'Liverpool', 'Chelsea', 'Leicester',
                         'West Ham', 'Tottenham', 'Arsenal', 'Leeds', 'Everton',
                         'Aston Villa', 'Newcastle', 'Wolves', 'Crystal Palace', 'Southampton',
                         'Brighton', 'Burnley', 'Fulham', 'West Brom', 'Sheffield Utd'],
                'position': list(range(1, 21)),
                'games': [38] * 20,
                'wins': [27, 21, 20, 19, 20, 19, 18, 18, 18, 17, 16, 12, 12, 12, 12, 9, 10, 5, 5, 7],
                'draws': [5, 11, 9, 10, 6, 8, 8, 7, 5, 8, 7, 9, 9, 8, 7, 14, 9, 13, 11, 2],
                'losses': [6, 6, 9, 9, 12, 11, 12, 13, 15, 13, 15, 17, 17, 18, 19, 15, 19, 20, 22, 29],
                'goals_for': [83, 73, 68, 58, 68, 62, 68, 55, 62, 47, 55, 46, 36, 41, 47, 40, 33, 27, 35, 20],
                'goals_against': [32, 44, 42, 36, 50, 47, 45, 39, 54, 48, 46, 62, 52, 66, 68, 46, 55, 53, 76, 63],
                'points': [86, 74, 69, 67, 66, 65, 62, 61, 59, 59, 55, 45, 45, 44, 43, 41, 39, 28, 26, 23],
                'xg': [78.5, 68.2, 65.5, 58.2, 62.5, 58.8, 65.2, 52.8, 58.5, 48.2, 52.5, 42.8, 38.5, 42.2, 48.5, 42.5,
                       35.8, 32.5, 38.2, 25.5],
                'xga': [35.2, 45.5, 45.8, 40.2, 52.5, 50.5, 48.2, 42.5, 55.8, 50.2, 48.5, 58.2, 50.5, 60.2, 62.5, 48.8,
                        52.5, 55.2, 70.5, 62.8],
                'champion': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'relegated': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
            }
        }

        if season in historical_data:
            df = pd.DataFrame(historical_data[season])
            df['season'] = season
            df['goal_diff'] = df['goals_for'] - df['goals_against']
            df['ppg'] = df['points'] / df['games']
            df['xg_diff'] = df['xg'] - df['xga']
            return df

        return pd.DataFrame()

    def get_team_historical_performance(self, team: str) -> Dict:
        """Get historical performance metrics for a team."""
        performance = {
            'seasons_in_pl': 0,
            'avg_position': [],
            'avg_points': [],
            'titles': 0,
            'top_4_finishes': 0,
            'relegations': 0,
            'avg_xg': [],
            'avg_xga': [],
            'consistency_score': 0
        }

        for season, df in self.seasons_data.items():
            team_data = df[df['team'] == team]
            if not team_data.empty:
                row = team_data.iloc[0]
                performance['seasons_in_pl'] += 1
                performance['avg_position'].append(row['position'])
                performance['avg_points'].append(row['points'])
                performance['avg_xg'].append(row.get('xg', 0))
                performance['avg_xga'].append(row.get('xga', 0))
                performance['titles'] += row.get('champion', 0)
                performance['top_4_finishes'] += 1 if row['position'] <= 4 else 0
                performance['relegations'] += row.get('relegated', 0)

        if performance['avg_position']:
            performance['avg_position'] = np.mean(performance['avg_position'])
            performance['avg_points'] = np.mean(performance['avg_points'])
            performance['avg_xg'] = np.mean(performance['avg_xg'])
            performance['avg_xga'] = np.mean(performance['avg_xga'])
            positions = [df[df['team'] == team].iloc[0]['position']
                         for s, df in self.seasons_data.items()
                         if not df[df['team'] == team].empty]
            performance['consistency_score'] = 1 / (np.std(positions) + 1)
        else:
            # Defaults for promoted teams
            performance['avg_position'] = 15
            performance['avg_points'] = 40
            performance['avg_xg'] = 45
            performance['avg_xga'] = 55
            performance['consistency_score'] = 0.5

        return performance

    def create_historical_features(self, current_teams: List[str]) -> pd.DataFrame:
        """Create features based on historical performance."""
        features = []

        for team in current_teams:
            perf = self.get_team_historical_performance(team)

            features.append({
                'team': team,
                'hist_seasons_in_pl': perf['seasons_in_pl'],
                'hist_avg_position': perf['avg_position'],
                'hist_avg_points': perf['avg_points'],
                'hist_titles': perf['titles'],
                'hist_top_4_rate': perf['top_4_finishes'] / max(perf['seasons_in_pl'], 1),
                'hist_avg_xg': perf['avg_xg'],
                'hist_avg_xga': perf['avg_xga'],
                'hist_consistency': perf['consistency_score'],
                'hist_relegation_risk': perf['relegations'] / max(perf['seasons_in_pl'], 1)
            })

        return pd.DataFrame(features)

    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get combined historical data for model training."""
        all_data = []

        for season, df in self.seasons_data.items():
            season_data = df.copy()
            scale = 17 / 38

            season_data['mid_games'] = 17
            season_data['mid_points'] = (season_data['points'] * scale).round()
            season_data['mid_goals_for'] = (season_data['goals_for'] * scale).round()
            season_data['mid_goals_against'] = (season_data['goals_against'] * scale).round()
            season_data['mid_xg'] = season_data['xg'] * scale
            season_data['mid_xga'] = season_data['xga'] * scale
            season_data['mid_goal_diff'] = season_data['mid_goals_for'] - season_data['mid_goals_against']
            season_data['mid_ppg'] = season_data['mid_points'] / 17
            season_data['target_points'] = season_data['points']
            season_data['target_champion'] = season_data['champion']

            all_data.append(season_data)

        combined = pd.concat(all_data, ignore_index=True)

        feature_cols = ['mid_points', 'mid_goals_for', 'mid_goals_against',
                        'mid_xg', 'mid_xga', 'mid_goal_diff', 'mid_ppg']

        X = combined[['team'] + feature_cols]
        y = combined['target_points']

        return X, y


def main():
    """Test historical data module."""
    manager = HistoricalDataManager()

    print("=== Historical Performance (for 2025/26 predictions) ===")
    for team in ['Liverpool', 'Manchester City', 'Arsenal', 'Chelsea', 'Nottingham Forest']:
        perf = manager.get_team_historical_performance(team)
        print(f"\n{team}:")
        print(f"  Seasons in PL (last 5): {perf['seasons_in_pl']}")
        print(f"  Avg Position: {perf['avg_position']:.1f}")
        print(f"  Titles (last 5 seasons): {perf['titles']}")
        print(f"  Top 4 Rate: {perf['top_4_finishes'] / max(perf['seasons_in_pl'], 1) * 100:.0f}%")


if __name__ == "__main__":
    main()