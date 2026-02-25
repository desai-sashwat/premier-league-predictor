"""
FastAPI application for Premier League Winner Predictor.
Wraps the existing PremierLeaguePredictor for REST API serving.
"""

import os
import glob
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predictor import PremierLeaguePredictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global predictor instance (loaded once at startup)
# ---------------------------------------------------------------------------
predictor: Optional[PremierLeaguePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the predictor and trained models on startup."""
    global predictor
    logger.info("Initializing PremierLeaguePredictor...")
    predictor = PremierLeaguePredictor()

    # Attempt to load the most recent trained model weights
    model_files = sorted(glob.glob("models/*.joblib"))
    if model_files:
        logger.info(f"Found {len(model_files)} saved model file(s).")
    else:
        logger.warning(
            "No saved models found in models/. "
            "Run `python main.py --scrape --train -g <GW>` first, "
            "or call POST /train after deployment."
        )
    yield
    logger.info("Shutting down predictor.")


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Premier League Predictor API",
    description="Predict the Premier League winner using an ML ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool


class TeamPrediction(BaseModel):
    team: str
    current_points: Optional[float] = None
    predicted_points: Optional[float] = None
    win_probability: Optional[float] = None
    top4_probability: Optional[float] = None
    points_ci_lower: Optional[float] = None
    points_ci_upper: Optional[float] = None
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    gameweek: int
    generated_at: str
    predictions: list[TeamPrediction]


class TrainResponse(BaseModel):
    message: str
    gameweek: int
    scrape: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and whether models are loaded."""
    models_exist = len(glob.glob("models/*.joblib")) > 0
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=models_exist,
    )


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    gameweek: int = Query(
        ..., ge=1, le=38, description="Current gameweek (1-38)"
    ),
):
    """
    Return predictions for the given gameweek using the trained ensemble.
    Requires that models have already been trained (POST /train or CLI).
    """
    try:
        predictions = predictor.run_full_pipeline(
            gameweek=gameweek,
            scrape=False,  # use existing data
            train=False,   # use existing model
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _format_predictions(predictions, gameweek)


@app.post("/train", response_model=TrainResponse)
async def train(
    gameweek: int = Query(
        ..., ge=1, le=38, description="Current gameweek (1-38)"
    ),
    scrape: bool = Query(
        False, description="Scrape fresh data from FBRef before training"
    ),
):
    """
    Trigger model training (and optionally scraping) for a given gameweek.
    """
    try:
        predictor.run_full_pipeline(
            gameweek=gameweek,
            scrape=scrape,
            train=True,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return TrainResponse(
        message="Training complete",
        gameweek=gameweek,
        scrape=scrape,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/compare")
async def compare_predictions():
    """
    Compare predictions across saved gameweeks.
    Returns a list of prediction CSVs with timestamps.
    """
    pred_files = sorted(glob.glob("data/predictions/predictions_gw*.csv"))
    if not pred_files:
        raise HTTPException(status_code=404, detail="No prediction history found.")

    comparisons = []
    for f in pred_files:
        try:
            df = pd.read_csv(f)
            comparisons.append(
                {
                    "file": os.path.basename(f),
                    "teams": len(df),
                    "top_team": df.iloc[0].to_dict() if len(df) > 0 else None,
                }
            )
        except Exception:
            continue

    return {"comparisons": comparisons}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_predictions(predictions, gameweek: int) -> PredictionResponse:
    """Convert raw predictor output into the API response schema."""
    if isinstance(predictions, pd.DataFrame):
        team_preds = []
        for _, row in predictions.iterrows():
            team_preds.append(
                TeamPrediction(
                    team=row.get("team", row.get("Squad", "Unknown")),
                    current_points=row.get("current_points"),
                    predicted_points=row.get("predicted_points"),
                    win_probability=row.get("win_probability"),
                    top4_probability=row.get("top4_probability"),
                    points_ci_lower=row.get("points_ci_lower"),
                    points_ci_upper=row.get("points_ci_upper"),
                    confidence=row.get("confidence"),
                )
            )
        return PredictionResponse(
            gameweek=gameweek,
            generated_at=datetime.utcnow().isoformat(),
            predictions=team_preds,
        )

    # Fallback: if predictions come back as a dict or other structure
    raise HTTPException(
        status_code=500,
        detail="Unexpected prediction format. Check predictor output.",
    )
