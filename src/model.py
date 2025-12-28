"""
Machine Learning Models for Premier League Prediction
Ensemble approach with iterative improvement
Works with or without XGBoost/LightGBM
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional libraries
HAS_XGBOOST = False
HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
    logger.info("XGBoost available")
except ImportError:
    logger.info("XGBoost not available - using GradientBoosting instead")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
    logger.info("LightGBM available")
except ImportError:
    logger.info("LightGBM not available - using GradientBoosting instead")


class PLPredictor:
    """
    Ensemble model for Premier League winner prediction.
    Combines multiple models for robust predictions.
    Works with scikit-learn only if XGBoost/LightGBM unavailable.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models = {}
        self.ensemble_weights = {}
        self.is_trained = False
        self.training_history = []
        self.feature_names = []

        self._initialize_models()

    def _initialize_models(self):
        """Initialize base models based on available libraries."""
        model_config = self.config['model']

        # Model 1: XGBoost or GradientBoosting
        if HAS_XGBOOST:
            self.models['xgboost'] = XGBRegressor(
                n_estimators=model_config['xgboost']['n_estimators'],
                max_depth=model_config['xgboost']['max_depth'],
                learning_rate=model_config['xgboost']['learning_rate'],
                subsample=model_config['xgboost'].get('subsample', 0.8),
                colsample_bytree=model_config['xgboost'].get('colsample_bytree', 0.8),
                random_state=42,
                verbosity=0
            )
            self.ensemble_weights['xgboost'] = 0.35
        else:
            self.models['gradient_boost_1'] = GradientBoostingRegressor(
                n_estimators=model_config['xgboost']['n_estimators'],
                max_depth=model_config['xgboost']['max_depth'],
                learning_rate=model_config['xgboost']['learning_rate'],
                subsample=model_config['xgboost'].get('subsample', 0.8),
                random_state=42
            )
            self.ensemble_weights['gradient_boost_1'] = 0.35

        # Model 2: LightGBM or GradientBoosting variant
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = LGBMRegressor(
                n_estimators=model_config['lightgbm']['n_estimators'],
                max_depth=model_config['lightgbm']['max_depth'],
                learning_rate=model_config['lightgbm']['learning_rate'],
                num_leaves=model_config['lightgbm'].get('num_leaves', 31),
                random_state=42,
                verbose=-1
            )
            self.ensemble_weights['lightgbm'] = 0.30
        else:
            self.models['gradient_boost_2'] = GradientBoostingRegressor(
                n_estimators=model_config['lightgbm']['n_estimators'],
                max_depth=model_config['lightgbm']['max_depth'],
                learning_rate=model_config['lightgbm']['learning_rate'],
                subsample=0.7,
                random_state=43
            )
            self.ensemble_weights['gradient_boost_2'] = 0.30

        # Model 3: Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=model_config['random_forest']['n_estimators'],
            max_depth=model_config['random_forest']['max_depth'],
            min_samples_split=model_config['random_forest'].get('min_samples_split', 5),
            random_state=42,
            n_jobs=-1
        )
        self.ensemble_weights['random_forest'] = 0.20

        # Model 4: AdaBoost for diversity
        self.models['adaboost'] = AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=44
        )
        self.ensemble_weights['adaboost'] = 0.10

        # Model 5: Ridge Regression as baseline
        self.models['ridge'] = Ridge(alpha=1.0)
        self.ensemble_weights['ridge'] = 0.05

        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")

    def train(self, X: pd.DataFrame, y: pd.Series, gameweek: int = None) -> Dict:
        """
        Train all models in the ensemble.
        Returns training metrics.
        """
        logger.info(f"Training ensemble models (gameweek {gameweek})...")

        # Prepare features (exclude team name)
        feature_cols = [c for c in X.columns if c != 'team']
        X_train = X[feature_cols].values
        y_train = y.values

        # Handle any NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=y_train[~np.isnan(y_train)].mean() if np.any(~np.isnan(y_train)) else 0)

        metrics = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            try:
                # Fit model
                model.fit(X_train, y_train)

                # Cross-validation score (use fewer folds for small dataset)
                n_splits = min(5, max(2, len(X_train) // 4))
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=n_splits,
                    scoring='neg_mean_absolute_error'
                )

                # Training metrics
                y_pred = model.predict(X_train)

                metrics[name] = {
                    'mae': mean_absolute_error(y_train, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
                    'r2': r2_score(y_train, y_pred),
                    'cv_mae': -cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }

                logger.info(f"  {name} - MAE: {metrics[name]['mae']:.2f}, "
                           f"CV MAE: {metrics[name]['cv_mae']:.2f}")

            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                metrics[name] = {'mae': np.inf, 'rmse': np.inf, 'r2': 0, 'cv_mae': np.inf, 'cv_std': 0}

        self.is_trained = True
        self.feature_names = feature_cols

        # Save training record
        self.training_history.append({
            'gameweek': gameweek,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'n_samples': len(X_train)
        })

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make ensemble predictions.
        Returns DataFrame with team names and predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        teams = X['team'].values
        feature_cols = [c for c in X.columns if c != 'team']
        X_pred = X[feature_cols].values

        # Handle NaN values
        X_pred = np.nan_to_num(X_pred, nan=0.0)

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X_pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        # Weighted ensemble
        ensemble_pred = np.zeros(len(X_pred))
        total_weight = 0

        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 0)
            ensemble_pred += weight * pred
            total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        # Build results DataFrame
        results = pd.DataFrame({
            'team': teams,
            'predicted_points': ensemble_pred
        })

        # Add individual model predictions
        for name, pred in predictions.items():
            results[f'{name}_pred'] = pred

        return results.sort_values('predicted_points', ascending=False).reset_index(drop=True)

    def simulate_season(self, X: pd.DataFrame, current_points: pd.Series,
                        remaining_games: int, n_simulations: int = 10000) -> pd.DataFrame:
        """
        Monte Carlo simulation of remaining season.
        Returns probability distribution for each team's final position.
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")

        predictions = self.predict(X)
        teams = predictions['team'].values
        predicted_final = predictions['predicted_points'].values

        n_teams = len(teams)

        # Simulation results
        position_counts = {team: {i: 0 for i in range(1, 21)} for team in teams}
        final_points_dist = {team: [] for team in teams}

        for sim in range(n_simulations):
            # Add noise based on remaining uncertainty
            uncertainty = remaining_games / 38 * 8  # More games left = more uncertainty
            noise = np.random.normal(0, uncertainty, n_teams)

            simulated_final = predicted_final + noise

            # Store points
            for i, team in enumerate(teams):
                final_points_dist[team].append(simulated_final[i])

            # Rank teams
            rankings = np.argsort(-simulated_final) + 1
            for i, team in enumerate(teams):
                pos = rankings[i]
                if pos <= 20:  # Safety check
                    position_counts[team][pos] += 1

        # Calculate probabilities
        results = []
        for team in teams:
            prob_1st = position_counts[team][1] / n_simulations * 100
            prob_top4 = sum(position_counts[team].get(i, 0) for i in range(1, 5)) / n_simulations * 100
            prob_relegation = sum(position_counts[team].get(i, 0) for i in range(18, 21)) / n_simulations * 100

            points_list = final_points_dist[team]
            mean_points = np.mean(points_list)
            std_points = np.std(points_list)

            results.append({
                'team': team,
                'prob_win_league': prob_1st,
                'prob_top_4': prob_top4,
                'prob_relegation': prob_relegation,
                'mean_projected_points': mean_points,
                'std_points': std_points,
                'points_90_ci_low': np.percentile(points_list, 5),
                'points_90_ci_high': np.percentile(points_list, 95)
            })

        return pd.DataFrame(results).sort_values('prob_win_league', ascending=False).reset_index(drop=True)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from all models."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        importance_dict = {feat: 0 for feat in self.feature_names}
        count = 0

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(self.feature_names):
                    importance_dict[feat] += model.feature_importances_[i]
                count += 1

        if count > 0:
            importance_dict = {k: v / count for k, v in importance_dict.items()}

        return pd.Series(importance_dict).sort_values(ascending=False)

    def save_model(self, gameweek: int = None):
        """Save trained models to disk."""
        save_dir = self.config['paths'].get('models', 'models')
        os.makedirs(save_dir, exist_ok=True)

        gw_suffix = f"_gw{gameweek}" if gameweek else ""

        # Save each model
        for name, model in self.models.items():
            filepath = f"{save_dir}/{name}{gw_suffix}.joblib"
            joblib.dump(model, filepath)
            logger.info(f"Saved {name} to {filepath}")

        # Save training history
        history_path = f"{save_dir}/training_history.joblib"
        joblib.dump(self.training_history, history_path)

        # Save feature names and weights
        meta = {
            'feature_names': self.feature_names,
            'ensemble_weights': self.ensemble_weights,
            'last_gameweek': gameweek
        }
        joblib.dump(meta, f"{save_dir}/model_meta.joblib")

    def load_model(self, gameweek: int = None):
        """Load trained models from disk."""
        save_dir = self.config['paths'].get('models', 'models')
        gw_suffix = f"_gw{gameweek}" if gameweek else ""

        # Load each model
        for name in list(self.models.keys()):
            filepath = f"{save_dir}/{name}{gw_suffix}.joblib"
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                logger.info(f"Loaded {name} from {filepath}")

        # Load metadata
        meta_path = f"{save_dir}/model_meta.joblib"
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.feature_names = meta.get('feature_names', [])
            self.ensemble_weights = meta.get('ensemble_weights', self.ensemble_weights)

        self.is_trained = True


def main():
    """Test the model."""
    from features import FeatureEngineer

    # Create features
    engineer = FeatureEngineer()
    X, y = engineer.engineer_features(gameweek=17)

    # Train model
    predictor = PLPredictor()
    metrics = predictor.train(X, y, gameweek=17)

    print("\n=== Training Metrics ===")
    for model, m in metrics.items():
        print(f"{model}: MAE={m['mae']:.2f}, R²={m['r2']:.3f}")

    # Predictions
    predictions = predictor.predict(X)
    print("\n=== Predictions ===")
    print(predictions[['team', 'predicted_points']].head(10))

    # Feature importance
    importance = predictor.get_feature_importance()
    print("\n=== Top 10 Features ===")
    print(importance.head(10))

    # Save model
    predictor.save_model(gameweek=17)


if __name__ == "__main__":
    main()