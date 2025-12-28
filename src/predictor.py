"""
Main Predictor Module - Orchestrates the entire prediction pipeline
"""

import logging
import os
from datetime import datetime
from typing import Dict

import pandas as pd
import yaml
from fixture_difficulty import FixtureDifficultyAnalyzer

from features import FeatureEngineer
from historical_data import HistoricalDataManager
from model import PLPredictor
from scraper import FBRefScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PremierLeaguePredictor:
    """
    Main class that orchestrates:
    1. Data scraping from FBRef
    2. Feature engineering
    3. Model training
    4. Predictions and simulations
    5. Tracking predictions over gameweeks
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.scraper = FBRefScraper(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.model = PLPredictor(config_path)

        # Feature managers (no player features)
        self.historical_manager = HistoricalDataManager(config_path)
        self.fixture_analyzer = FixtureDifficultyAnalyzer(config_path)

        self.predictions_history = []
        self.current_gameweek = self.config.get('current_gameweek', 17)

    def update_data(self, gameweek: int = None) -> Dict[str, pd.DataFrame]:
        """Scrape latest data from FBRef."""
        logger.info(f"Updating data for gameweek {gameweek or self.current_gameweek}...")

        data = self.scraper.scrape_all_stats()
        self.scraper.save_data(data, gameweek=gameweek)

        return data

    def prepare_features(self, gameweek: int = None) -> tuple:
        """Engineer features from raw data."""
        logger.info("Engineering features...")

        X, y = self.feature_engineer.engineer_features(gameweek=gameweek)
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series, gameweek: int = None) -> Dict:
        """Train the prediction model."""
        logger.info("Training model...")

        metrics = self.model.train(X, y, gameweek=gameweek)
        self.model.save_model(gameweek=gameweek)

        return metrics

    def get_predictions(self, X: pd.DataFrame, current_standings: pd.DataFrame = None,
                        remaining_games: int = None) -> pd.DataFrame:
        """Get predictions with confidence intervals."""

        # Basic predictions
        predictions = self.model.predict(X)

        # Monte Carlo simulation for probabilities
        if current_standings is not None and remaining_games is not None:
            current_points = current_standings['points']
            simulations = self.model.simulate_season(
                X, current_points, remaining_games,
                n_simulations=self.config['prediction']['monte_carlo_simulations']
            )

            # Merge results
            predictions = predictions.merge(
                simulations[['team', 'prob_win_league', 'prob_top_4',
                            'prob_relegation', 'points_90_ci_low', 'points_90_ci_high']],
                on='team'
            )

        return predictions

    def run_full_pipeline(self, gameweek: int = None, scrape: bool = True,
                          train: bool = True) -> pd.DataFrame:
        """
        Run the complete prediction pipeline.

        Args:
            gameweek: Current gameweek number
            scrape: Whether to scrape fresh data
            train: Whether to retrain the model

        Returns:
            DataFrame with predictions and probabilities
        """
        gw = gameweek or self.current_gameweek
        logger.info(f"=" * 60)
        logger.info(f"Running prediction pipeline for Gameweek {gw}")
        logger.info(f"=" * 60)

        # Step 1: Scrape data
        if scrape:
            self.update_data(gameweek=gw)

        # Step 2: Engineer features
        X, y = self.prepare_features(gameweek=gw)

        # Step 3: Train model
        if train:
            metrics = self.train_model(X, y, gameweek=gw)
            self._print_training_summary(metrics)
        else:
            self.model.load_model()

        # Step 4: Get current standings for simulation
        standings = self._load_current_standings()
        remaining_games = 38 - gw

        # Step 5: Generate predictions
        predictions = self.get_predictions(X, standings, remaining_games)

        # Step 6: Add current standings info
        if standings is not None:
            predictions = predictions.merge(
                standings[['team', 'points', 'games', 'goal_diff']],
                on='team',
                how='left'
            )

        # Step 7: Save predictions
        self._save_predictions(predictions, gw)

        # Step 8: Track history
        self._track_prediction(predictions, gw)

        return predictions

    def _load_current_standings(self) -> pd.DataFrame:
        """Load current standings from processed data."""
        filepath = f"{self.config['paths']['processed_data']}/standings_with_projection_latest.csv"
        alt_path = f"{self.config['paths']['raw_data']}/standings_latest.csv"

        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        elif os.path.exists(alt_path):
            return pd.read_csv(alt_path)
        return None

    def _print_training_summary(self, metrics: Dict):
        """Print training metrics summary."""
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)

        for model_name, m in metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  MAE: {m['mae']:.2f} points")
            print(f"  RMSE: {m['rmse']:.2f} points")
            print(f"  R²: {m['r2']:.3f}")
            print(f"  CV MAE: {m['cv_mae']:.2f} ± {m['cv_std']:.2f}")

    def _save_predictions(self, predictions: pd.DataFrame, gameweek: int):
        """Save predictions to disk."""
        save_dir = self.config['paths']['predictions']
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save with timestamp
        predictions.to_csv(f"{save_dir}/predictions_gw{gameweek}_{timestamp}.csv", index=False)

        # Save latest
        predictions.to_csv(f"{save_dir}/predictions_latest.csv", index=False)

        logger.info(f"Predictions saved to {save_dir}")

    def _track_prediction(self, predictions: pd.DataFrame, gameweek: int):
        """Track predictions over time for analysis."""
        top_prediction = predictions.iloc[0]

        self.predictions_history.append({
            'gameweek': gameweek,
            'timestamp': datetime.now().isoformat(),
            'predicted_winner': top_prediction['team'],
            'winner_prob': top_prediction.get('prob_win_league', 0),
            'winner_points': top_prediction['predicted_points'],
            'top_4': predictions.head(4)['team'].tolist()
        })

    def compare_predictions(self) -> pd.DataFrame:
        """Compare predictions across gameweeks."""
        if not self.predictions_history:
            return pd.DataFrame()

        return pd.DataFrame(self.predictions_history)

    def print_predictions(self, predictions: pd.DataFrame, top_n: int = 10):
        """Pretty print predictions with enhanced insights."""
        print("\n" + "=" * 90)
        print("PREMIER LEAGUE 2025/26 - TITLE RACE PREDICTIONS (ENHANCED MODEL)")
        print("=" * 90)

        print(f"\n{'Rank':<5} {'Team':<22} {'Pts':<8} {'Win %':<8} {'Top 4 %':<9} {'Fixture':<10}")
        print("-" * 70)

        for i, row in predictions.head(top_n).iterrows():
            rank = i + 1
            team = row['team'][:20]
            pts = row['predicted_points']
            win_prob = row.get('prob_win_league', 0)
            top4_prob = row.get('prob_top_4', 0)

            # Get fixture difficulty
            fixture_diff = self.fixture_analyzer.calculate_fixture_difficulty(row['team'])
            fixture_str = f"{fixture_diff['avg_difficulty']:.1f}"

            print(f"{rank:<5} {team:<22} {pts:<8.1f} {win_prob:<8.1f} {top4_prob:<9.1f} {fixture_str:<10}")

        print("\n" + "-" * 70)

        # Fixture comparison for top 4
        print("\n📅 FIXTURE DIFFICULTY (Remaining Games):")
        for team in predictions.head(4)['team']:
            diff = self.fixture_analyzer.calculate_fixture_difficulty(team)
            next5 = self.fixture_analyzer.calculate_next5_difficulty(team)
            print(f"  {team}: Avg={diff['avg_difficulty']:.1f}, Next 5={next5['avg_difficulty']:.1f}, "
                  f"Hard={diff['hard_games']}, Easy={diff['easy_games']}")

        # Historical context
        print("\n📊 HISTORICAL CONTEXT (Last 5 Seasons):")
        for team in predictions.head(4)['team']:
            hist = self.historical_manager.get_team_historical_performance(team)
            print(f"  {team}: {hist['titles']} titles, Avg pos: {hist['avg_position']:.1f}, "
                  f"Top 4 rate: {hist['top_4_finishes']/max(hist['seasons_in_pl'],1)*100:.0f}%")

        # Relegation battle
        if 'prob_relegation' in predictions.columns:
            print("\n⚠️ RELEGATION BATTLE:")
            print("-" * 40)
            relegation = predictions.nlargest(5, 'prob_relegation')[
                ['team', 'predicted_points', 'prob_relegation']
            ]
            for _, row in relegation.iterrows():
                print(f"  {row['team']:<20} {row['prob_relegation']:.1f}% relegation risk")

        print("\n" + "=" * 90)

    def get_prediction_confidence(self, predictions: pd.DataFrame) -> str:
        """Analyze prediction confidence based on remaining games."""
        remaining = 38 - self.current_gameweek

        if remaining > 25:
            return "LOW - Season just started, predictions are highly uncertain"
        elif remaining > 15:
            return "MODERATE - Patterns emerging but much can change"
        elif remaining > 8:
            return "GOOD - Clear picture forming, reasonable confidence"
        elif remaining > 3:
            return "HIGH - Strong confidence in predictions"
        else:
            return "VERY HIGH - Title race nearly decided"


def main():
    """Run the predictor."""
    predictor = PremierLeaguePredictor()

    # Run full pipeline
    predictions = predictor.run_full_pipeline(
        gameweek=17,
        scrape=True,
        train=True
    )

    # Print results
    predictor.print_predictions(predictions)

    # Show confidence level
    confidence = predictor.get_prediction_confidence(predictions)
    print(f"\nPrediction Confidence: {confidence}")

    # Feature importance
    print("\n=== TOP FEATURES ===")
    importance = predictor.model.get_feature_importance()
    print(importance.head(10))


if __name__ == "__main__":
    main()