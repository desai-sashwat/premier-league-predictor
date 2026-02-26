# Premier League Winner Predictor Using Machine Learning Ensemble

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![API](https://img.shields.io/badge/API-Live-green.svg)](https://premier-league-predictor-yh0q.onrender.com/docs)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

## Table of Contents
- [Overview](#overview)
- [Live API](#live-api)
- [Author](#author)
- [Machine Learning Approach](#machine-learning-approach)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Data Pipeline](#data-pipeline)
  - [Data Collection](#data-collection)
  - [Feature Engineering](#feature-engineering)
  - [Feature Categories](#feature-categories)
- [Model Architecture](#model-architecture)
  - [Ensemble Configuration](#ensemble-configuration)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [License](#license)

## Overview
This project implements a machine learning system to predict the Premier League winner using an ensemble of regression models. The system scrapes real-time data from FBRef, engineers comprehensive features from multiple data sources (current season statistics, historical performance, and fixture difficulty), and uses Monte Carlo simulation to estimate win probabilities. The predictor updates iteratively throughout the season, providing gameweek-by-gameweek projections with confidence intervals.

The project is deployed as a REST API using FastAPI and Docker, with automated weekly model retraining via GitHub Actions.

## Live API

The predictor is deployed and accessible at:

**Base URL**: https://premier-league-predictor-yh0q.onrender.com

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check and model status |
| `/predict?gameweek=N` | GET | Get predictions for gameweek N (1-38) |
| `/train?gameweek=N&scrape=true` | POST | Trigger model retraining |
| `/compare` | GET | Compare predictions across gameweeks |
| `/docs` | GET | Interactive Swagger API documentation |

**Example Request**:
```bash
curl https://premier-league-predictor-yh0q.onrender.com/predict?gameweek=27
```

**Example Response**:
```json
{
  "gameweek": 27,
  "generated_at": "2026-02-26T00:23:06.971220",
  "predictions": [
    {
      "team": "Arsenal",
      "predicted_points": 82.5,
      "prob_win_league": 5.21,
      "prob_top_4": 20.02,
      "prob_relegation": 0.0,
      "points_90_ci_low": 75.3,
      "points_90_ci_high": 89.7,
      "points": 39,
      "games": 17,
      "goal_diff": 21
    }
  ]
}
```

> **Note**: The API is hosted on Render's free tier and may take ~30 seconds to cold start after periods of inactivity.

## Author
- Sashwat Desai (desai.sas@northeastern.edu)
  - MS, Applied Mathematics
  - Northeastern University, Boston

## Machine Learning Approach
The implementation uses a weighted ensemble of regression models to predict final season points:
- **Ensemble Architecture**: Combines 5 diverse models (XGBoost, LightGBM, Random Forest, AdaBoost, Ridge) with optimized weights
- **Feature Engineering**: 30+ engineered features across offensive, defensive, form, historical, and fixture difficulty categories
- **Probabilistic Prediction**: Monte Carlo simulation (10,000 iterations) for win probability estimation
- **Iterative Updates**: Weekly model retraining as new gameweek data becomes available
- **Graceful Fallbacks**: Automatic fallback to scikit-learn models if XGBoost/LightGBM unavailable

## Key Features
- Real-time data scraping from FBRef using Selenium WebDriver
- Comprehensive feature engineering from multiple data sources
- Weighted ensemble model with cross-validation
- Monte Carlo simulation for probabilistic predictions
- Historical performance analysis (past 5 seasons)
- Fixture difficulty rating system
- REST API with FastAPI for real-time prediction serving
- Dockerized deployment for portability
- Automated weekly retraining via GitHub Actions
- Prediction tracking and comparison across gameweeks

## Repository Structure
- **.github/workflows/** - CI/CD pipelines
  - **weekly-update.yml** - Automated weekly scrape, retrain, and deploy
- **config/** - Configuration files
  - **config.yaml** - Model hyperparameters, feature weights, and scraping settings
- **data/** - Data storage directories
  - **historical/** - Historical season data (past 5 seasons)
  - **predictions/** - Model prediction outputs with timestamps
  - **processed/** - Engineered features and processed data
  - **raw/** - Raw scraped data from FBRef
- **models/** - Saved trained model weights (.joblib files)
- **notebooks/** - Jupyter notebooks for analysis and experimentation
- **src/** - Source code modules
  - **features.py** - Feature engineering pipeline
  - **fixture_difficulty.py** - Fixture difficulty rating system
  - **historical_data.py** - Historical data management
  - **model.py** - Ensemble model implementation
  - **predictor.py** - Main prediction orchestration
  - **scraper.py** - FBRef web scraper with Selenium
  - **utils.py** - Utility functions
- **app.py** - FastAPI application for REST API serving
- **Dockerfile** - Multi-stage Docker build for production
- **docker-compose.yml** - Docker Compose configuration
- **main.py** - Command-line entry point
- **requirements.txt** - Python dependencies

## Data Pipeline

### Data Collection
The system scrapes comprehensive statistics from FBRef including:
- **League Standings**: Points, wins, draws, losses, goal difference
- **Squad Statistics**: Standard performance metrics
- **Shooting Stats**: Goals, shots, shots on target, xG
- **Passing Stats**: Pass completion, progressive passes
- **Defensive Stats**: Tackles, interceptions, blocks, xGA
- **Possession Stats**: Possession percentage, touches
- **Fixtures**: Match results and upcoming schedule

### Feature Engineering
Features are created from three primary sources:

| Source | Description | Weight |
|--------|-------------|--------|
| Current Season | Live statistics from ongoing season | 40% |
| Historical Data | Past 5 seasons performance | 15% |
| Fixture Difficulty | Remaining schedule analysis | 15% |
| Form Metrics | Recent 5-game performance | 30% |

### Feature Categories

| Category | Weight | Key Metrics |
|----------|--------|-------------|
| Offensive | 22% | Goals scored, xG, xG performance, goals per game |
| Defensive | 22% | Goals conceded, xGA, xGA performance, goals conceded per game |
| Form | 28% | Points, PPG, win rate, last 5 games, goal difference trend |
| Historical | 13% | Average position, average points, titles, top-4 rate, consistency |
| Fixture | 15% | Average difficulty, next 5 difficulty, advantage score, easy games count |

## Model Architecture

### Ensemble Configuration
The predictor uses a weighted ensemble of five models:

| Model | Weight | Configuration |
|-------|--------|---------------|
| XGBoost | 35% | 200 estimators, max_depth=6, lr=0.05 |
| LightGBM | 30% | 200 estimators, max_depth=6, lr=0.05 |
| Random Forest | 20% | 300 estimators, max_depth=10 |
| AdaBoost | 10% | 100 estimators, lr=0.1 |
| Ridge Regression | 5% | alpha=1.0 (baseline) |

**Fallback Behavior**: If XGBoost or LightGBM are unavailable, the system automatically substitutes GradientBoostingRegressor with equivalent hyperparameters.

### Monte Carlo Simulation
Win probability estimation using 10,000 simulations:
- Adds Gaussian noise proportional to remaining season uncertainty
- Calculates probability distributions for each finishing position
- Outputs 90% confidence intervals for projected points
- Tracks probability of winning league, top-4 finish, and relegation

## Results
The system outputs comprehensive predictions including:

| Metric | Description |
|--------|-------------|
| Predicted Points | Ensemble-weighted final points projection |
| Win Probability | Monte Carlo estimated probability of winning the league |
| Top-4 Probability | Probability of finishing in Champions League spots |
| Points 90% CI | 90% confidence interval for final points |
| Prediction Confidence | Model confidence based on CV scores and remaining games |

### Performance Metrics
- **Cross-Validation**: 5-fold CV with MAE scoring
- **Ensemble Diversity**: Multiple model types for robust predictions
- **Adaptive Uncertainty**: Wider confidence intervals early in season

## Setup and Installation

### Prerequisites
```bash
# Clone this repository
git clone https://github.com/desai-sashwat/premier-league-predictor.git
cd premier-league-predictor

# Install required packages
pip install -r requirements.txt
```

### Chrome WebDriver
The scraper uses Selenium with Chrome. The `webdriver-manager` package automatically handles driver installation.

## Usage

### Command Line Interface
```bash
# Initialize project structure
python main.py --init

# Full pipeline: scrape data, train model, predict (for gameweek 17)
python main.py --scrape --train -g 17

# Update with new gameweek data
python main.py --update -g 18

# Predict using existing trained model
python main.py --predict

# Compare predictions across gameweeks
python main.py --compare

# Run automated weekly updates (scheduled for Tuesdays)
python main.py --schedule
```

### Python API
```python
from src.predictor import PremierLeaguePredictor

# Initialize predictor
predictor = PremierLeaguePredictor()

# Run full pipeline
predictions = predictor.run_full_pipeline(
    gameweek=17,
    scrape=True,
    train=True
)

# Display results
predictor.print_predictions(predictions)

# Get prediction confidence
confidence = predictor.get_prediction_confidence(predictions)
print(f"Confidence: {confidence}")
```

### REST API
```bash
# Run locally
uvicorn app:app --reload --port 8000

# Run with Docker
docker compose up --build

# API is available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Output Files
- **data/predictions/predictions_gwX_TIMESTAMP.csv**: Full predictions with probabilities
- **data/processed/features_gwX.csv**: Engineered features for each gameweek
- **models/MODEL_NAME_gwX.joblib**: Saved model weights

## Deployment

### Docker
```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# Stop
docker compose down
```

### Cloud Deployment (Render)
The application is deployed on Render with automatic deployments on push to `main`. The Dockerfile is auto-detected and built by Render's pipeline.

### Automated Weekly Updates
A GitHub Actions workflow (`.github/workflows/weekly-update.yml`) runs every Tuesday at 08:00 UTC to:
1. Scrape the latest match data from FBRef
2. Retrain all ensemble models
3. Generate updated predictions
4. Commit new model weights and predictions to the repository
5. Trigger automatic redeployment on Render

## Future Work
- **Player-Level Features**: Incorporate individual player statistics and injury data
- **Transfer Window Impact**: Model the effect of January transfers on predictions
- **Bayesian Approach**: Implement Bayesian ensemble for better uncertainty quantification
- **Deep Learning**: Explore LSTM/Transformer models for sequential match prediction
- **Multi-League Support**: Extend to other major European leagues
- **Web Dashboard**: Create interactive visualization dashboard for predictions
- **Betting Odds Integration**: Compare predictions against bookmaker odds for calibration

## License
This project is licensed under the MIT License - see the LICENSE file for details.
