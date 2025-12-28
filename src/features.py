"""
Feature Engineering for Premier League Predictor
Integrates: Base stats, Historical data, Fixture difficulty
Season: 2025/26
"""

import logging
import os
from typing import Dict, Tuple

import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

from fixture_difficulty import FixtureDifficultyAnalyzer
# Import feature modules
from historical_data import HistoricalDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates ML-ready features from multiple data sources:
    1. Current season statistics (standings, xG, etc.)
    2. Historical performance (past 5 seasons)
    3. Fixture difficulty (remaining schedule analysis)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.weights = self.config['feature_weights']
        self.scaler = MinMaxScaler()
        self.feature_names = []

        # Initialize feature managers
        self.historical_manager = HistoricalDataManager(config_path)
        self.fixture_analyzer = FixtureDifficultyAnalyzer(config_path)

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load the latest raw data files."""
        data = {}
        raw_dir = self.config['paths']['raw_data']

        files = ['standings', 'standard', 'shooting', 'passing',
                 'defense', 'possession', 'misc', 'fixtures']

        for file in files:
            filepath = f"{raw_dir}/{file}_latest.csv"
            if os.path.exists(filepath):
                data[file] = pd.read_csv(filepath)
                logger.info(f"Loaded {file}: {len(data[file])} rows")
            else:
                logger.warning(f"File not found: {filepath}")
                data[file] = pd.DataFrame()

        return data

    def merge_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all data sources into a single DataFrame."""
        if data['standings'].empty:
            raise ValueError("No standings data available")

        merged = data['standings'].copy()
        if 'squad' in merged.columns and 'team' not in merged.columns:
            merged = merged.rename(columns={'squad': 'team'})

        for stat_name in ['standard', 'shooting', 'passing', 'defense', 'possession', 'misc']:
            if not data[stat_name].empty:
                df = data[stat_name].copy()
                if 'squad' in df.columns:
                    df = df.rename(columns={'squad': 'team'})

                cols_to_merge = [c for c in df.columns if c != 'team']
                df_subset = df[['team'] + cols_to_merge]

                merged = merged.merge(
                    df_subset, on='team', how='left',
                    suffixes=('', f'_{stat_name}')
                )

        logger.info(f"Merged data: {merged.shape}")
        return merged

    def create_base_features(self, df: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Create base statistical features from current season data."""
        features = pd.DataFrame()
        features['team'] = df['team']

        games = df.get('games', df.get('mp', 17))

        # === OFFENSIVE FEATURES ===
        features['goals_scored'] = df.get('goals_for', df.get('gls', 0))
        features['xg'] = df.get('xg_for', df.get('xg', 0))
        features['xg_performance'] = features['goals_scored'] - features['xg']
        features['goals_per_game'] = features['goals_scored'] / games.replace(0, 1)

        # === DEFENSIVE FEATURES ===
        features['goals_conceded'] = df.get('goals_against', df.get('gls_against', 0))
        features['xga'] = df.get('xg_against', df.get('xga', 0))
        features['xga_performance'] = features['xga'] - features['goals_conceded']
        features['goals_conceded_per_game'] = features['goals_conceded'] / games.replace(0, 1)

        # === POINTS & FORM ===
        points = df.get('points', df.get('pts', 0))
        features['points'] = points
        features['ppg'] = points / games.replace(0, 1)
        features['goal_diff'] = df.get('goal_diff', df.get('gd', 0))
        features['goal_diff_per_game'] = features['goal_diff'] / games.replace(0, 1)

        # === DERIVED METRICS ===
        features['xg_diff'] = features['xg'] - features['xga']
        features['xg_diff_per_game'] = features['xg_diff'] / games.replace(0, 1)

        # Win/Draw/Loss rates
        wins = df.get('wins', df.get('w', 0))
        draws = df.get('draws', df.get('d', 0))
        losses = df.get('losses', df.get('l', 0))

        features['win_rate'] = wins / games.replace(0, 1)
        features['draw_rate'] = draws / games.replace(0, 1)
        features['loss_rate'] = losses / games.replace(0, 1)

        # Recent form from fixtures
        for team in features['team']:
            form = self._calculate_form(fixtures, team, n_games=5)
            features.loc[features['team'] == team, 'form_last5'] = form['points']
            features.loc[features['team'] == team, 'form_goals'] = form['goals']

        # Fill NaN with sensible defaults
        for col in features.columns:
            if col != 'team':
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

        return features

    def _calculate_form(self, fixtures: pd.DataFrame, team: str, n_games: int = 5) -> Dict:
        """Calculate recent form for a team."""
        if fixtures.empty:
            return {'points': 0, 'goals': 0, 'conceded': 0}

        team_matches = fixtures[
            ((fixtures.get('home_team', pd.Series()) == team) |
             (fixtures.get('away_team', pd.Series()) == team)) &
            (fixtures.get('score', pd.Series()).notna())
        ].copy()

        if team_matches.empty:
            return {'points': 0, 'goals': 0, 'conceded': 0}

        team_matches = team_matches.tail(n_games)

        points, goals, conceded = 0, 0, 0
        for _, match in team_matches.iterrows():
            try:
                score = str(match.get('score', ''))
                if not score or ('–' not in score and '-' not in score):
                    continue

                sep = '–' if '–' in score else '-'
                home_goals, away_goals = map(int, score.split(sep))

                is_home = match.get('home_team') == team

                if is_home:
                    goals += home_goals
                    conceded += away_goals
                    if home_goals > away_goals:
                        points += 3
                    elif home_goals == away_goals:
                        points += 1
                else:
                    goals += away_goals
                    conceded += home_goals
                    if away_goals > home_goals:
                        points += 3
                    elif away_goals == home_goals:
                        points += 1
            except:
                continue

        return {'points': points, 'goals': goals, 'conceded': conceded}

    def create_all_features(self, data: Dict[str, pd.DataFrame], gameweek: int = 17) -> pd.DataFrame:
        """
        Create comprehensive feature set from all sources.
        """
        merged = self.merge_data(data)
        teams = merged['team'].tolist()

        logger.info("Creating base features...")
        base_features = self.create_base_features(merged, data.get('fixtures', pd.DataFrame()))

        logger.info("Creating historical features...")
        historical_features = self.historical_manager.create_historical_features(teams)

        logger.info("Creating fixture difficulty features...")
        # Update fixture analyzer with current standings
        self.fixture_analyzer.update_ratings_from_standings(merged)
        fixture_features = self.fixture_analyzer.create_fixture_features(teams)

        # Merge all features
        all_features = base_features.merge(historical_features, on='team', how='left')
        all_features = all_features.merge(fixture_features, on='team', how='left')

        # Fill any NaN values
        for col in all_features.columns:
            if col != 'team':
                all_features[col] = pd.to_numeric(all_features[col], errors='coerce').fillna(0)

        logger.info(f"Total features created: {len(all_features.columns) - 1}")

        return all_features

    def apply_feature_weights(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply category weights to features.
        Categories: offensive, defensive, form, historical, fixture
        """
        teams = features['team'].values
        weighted = pd.DataFrame({'team': teams})

        # Define feature categories
        categories = {
            'offensive': ['goals_scored', 'xg', 'xg_performance', 'goals_per_game'],
            'defensive': ['goals_conceded', 'xga', 'xga_performance', 'goals_conceded_per_game'],
            'form': ['points', 'ppg', 'win_rate', 'form_last5', 'goal_diff_per_game'],
            'historical': ['hist_avg_position', 'hist_avg_points', 'hist_titles',
                          'hist_top_4_rate', 'hist_consistency'],
            'fixture': ['fixture_avg_difficulty', 'fixture_next5_difficulty',
                       'fixture_advantage_score', 'fixture_easy_games']
        }

        # Category weights (must sum to 1.0)
        category_weights = {
            'offensive': 0.22,
            'defensive': 0.22,
            'form': 0.28,
            'historical': 0.13,
            'fixture': 0.15
        }

        # Apply weights
        for category, cols in categories.items():
            weight = category_weights.get(category, 0.1)
            available_cols = [c for c in cols if c in features.columns]

            if available_cols:
                cat_data = features[available_cols].copy()

                # Handle inverse features (lower is better)
                inverse_cols = ['goals_conceded', 'xga', 'goals_conceded_per_game',
                               'fixture_avg_difficulty', 'fixture_next5_difficulty',
                               'hist_avg_position']

                for col in available_cols:
                    if col in inverse_cols:
                        cat_data[col] = cat_data[col].max() - cat_data[col]

                # Scale to 0-1
                cat_normalized = pd.DataFrame(
                    self.scaler.fit_transform(cat_data),
                    columns=available_cols
                )

                # Apply category weight
                for col in available_cols:
                    weighted[f"w_{category}_{col}"] = cat_normalized[col] * weight

        return weighted

    def create_target_variable(self, df: pd.DataFrame, gameweek: int = 17) -> pd.Series:
        """
        Create target variable: projected final points.
        """
        games = df.get('games', df.get('mp', gameweek))
        points = df.get('points', df.get('pts', 0))
        total_games = 38

        ppg = points / games.replace(0, 1)
        base_projection = ppg * total_games

        return base_projection

    def engineer_features(self, gameweek: int = 17) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main method to create all features.
        Returns (X, y) where X is feature matrix and y is projected points.
        """
        logger.info(f"Starting feature engineering for gameweek {gameweek}...")

        data = self.load_raw_data()
        all_features = self.create_all_features(data, gameweek)
        weighted_features = self.apply_feature_weights(all_features)

        X = weighted_features.copy()

        # Add raw features
        raw_cols = ['ppg', 'xg_diff_per_game', 'goal_diff', 'points']
        for col in raw_cols:
            if col in all_features.columns:
                X[col] = all_features[col]

        merged = self.merge_data(data)
        y = self.create_target_variable(merged, gameweek)

        self.feature_names = [c for c in X.columns if c != 'team']

        save_dir = self.config['paths']['processed_data']
        os.makedirs(save_dir, exist_ok=True)

        X.to_csv(f"{save_dir}/features_gw{gameweek}.csv", index=False)
        all_features.to_csv(f"{save_dir}/all_features_gw{gameweek}.csv", index=False)

        standings = merged[['team', 'points', 'games', 'goal_diff']].copy()
        standings['projected_points'] = y
        standings.to_csv(f"{save_dir}/standings_with_projection_gw{gameweek}.csv", index=False)

        logger.info(f"Created {len(self.feature_names)} weighted features for {len(X)} teams")

        return X, y


def main():
    """Test feature engineering."""
    engineer = FeatureEngineer()
    X, y = engineer.engineer_features(gameweek=17)

    print("\n=== Feature Summary ===")
    print(f"Total features: {len(engineer.feature_names)}")
    print(f"Teams: {len(X)}")

    print("\n=== Projected Points (2025/26 Season) ===")
    results = pd.DataFrame({
        'team': X['team'],
        'projected_points': y
    }).sort_values('projected_points', ascending=False)
    print(results.head(10))


if __name__ == "__main__":
    main()