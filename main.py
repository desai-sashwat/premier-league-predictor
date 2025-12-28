#!/usr/bin/env python3
"""
Premier League Winner Predictor - Main Entry Point

Usage:
    python main.py --scrape --train              # Full pipeline: scrape, train, predict
    python main.py --update --gameweek 18        # Update with new gameweek data
    python main.py --predict                      # Just predict (use existing model)
    python main.py --compare                      # Compare predictions across gameweeks
    python main.py --schedule                     # Run automated weekly updates
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import schedule

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predictor import PremierLeaguePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_prediction(gameweek: int, scrape: bool = True, train: bool = True):
    """Run the prediction pipeline."""
    logger.info(f"Starting prediction for gameweek {gameweek}")

    try:
        predictor = PremierLeaguePredictor()
        predictor.current_gameweek = gameweek

        predictions = predictor.run_full_pipeline(
            gameweek=gameweek,
            scrape=scrape,
            train=train
        )

        # Display results
        predictor.print_predictions(predictions)

        confidence = predictor.get_prediction_confidence(predictions)
        print(f"\n📊 Prediction Confidence: {confidence}")

        # Show winner
        winner = predictions.iloc[0]
        print(f"\n🏆 PREDICTED WINNER: {winner['team']}")
        print(f"   Projected Points: {winner['predicted_points']:.1f}")
        if 'prob_win_league' in winner:
            print(f"   Win Probability: {winner['prob_win_league']:.1f}%")

        return predictions

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def compare_gameweeks():
    """Compare predictions across multiple gameweeks."""
    predictor = PremierLeaguePredictor()

    predictions_dir = predictor.config['paths']['predictions']

    if not os.path.exists(predictions_dir):
        print("No historical predictions found.")
        return

    # Load all prediction files
    import glob
    files = glob.glob(f"{predictions_dir}/predictions_gw*.csv")

    if not files:
        print("No prediction files found.")
        return

    import pandas as pd

    print("\n" + "=" * 70)
    print("PREDICTION HISTORY - TRACKING WINNER PREDICTIONS")
    print("=" * 70)

    history = []
    for f in sorted(files):
        df = pd.read_csv(f)
        gw = int(f.split('_gw')[1].split('_')[0])
        winner = df.iloc[0]

        history.append({
            'gameweek': gw,
            'predicted_winner': winner['team'],
            'points': winner['predicted_points'],
            'win_prob': winner.get('prob_win_league', 'N/A')
        })

    history_df = pd.DataFrame(history).sort_values('gameweek')

    print(f"\n{'GW':<5} {'Predicted Winner':<25} {'Points':<10} {'Win Prob':<10}")
    print("-" * 50)

    for _, row in history_df.iterrows():
        prob = f"{row['win_prob']:.1f}%" if isinstance(row['win_prob'], float) else row['win_prob']
        print(f"{row['gameweek']:<5} {row['predicted_winner']:<25} {row['points']:<10.1f} {prob:<10}")

    # Show consistency
    winners = history_df['predicted_winner'].value_counts()
    print(f"\n📈 Prediction Consistency:")
    for team, count in winners.items():
        pct = count / len(history_df) * 100
        print(f"   {team}: Predicted as winner {count}/{len(history_df)} times ({pct:.0f}%)")


def run_scheduled():
    """Run scheduled updates (for automation)."""
    logger.info("Starting scheduled predictor...")

    def weekly_update():
        """Run weekly prediction update."""
        # Determine current gameweek (simplified - you'd want proper logic here)
        # This assumes roughly 1 gameweek per week
        start_date = datetime(2024, 8, 17)  # Season start
        weeks = (datetime.now() - start_date).days // 7
        gameweek = min(weeks, 38)

        logger.info(f"Running scheduled update for estimated gameweek {gameweek}")
        run_prediction(gameweek, scrape=True, train=True)

    # Schedule for Tuesday (after most gameweeks complete)
    schedule.every().tuesday.at("10:00").do(weekly_update)

    print("Scheduler started. Press Ctrl+C to stop.")
    print("Next run:", schedule.next_run())

    while True:
        schedule.run_pending()
        time.sleep(60)


def create_project_structure():
    """Create necessary directories."""
    dirs = [
        'config',
        'data/raw',
        'data/processed',
        'data/predictions',
        'models',
        'src',
        'notebooks'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")

    # Create __init__.py
    with open('src/__init__.py', 'w') as f:
        f.write('# Premier League Predictor\n')

    print("\n✓ Project structure created!")
    print("\nNext steps:")
    print("1. Copy config.yaml to config/")
    print("2. Copy Python files to src/")
    print("3. Run: python main.py --scrape --train --gameweek 17")


def main():
    parser = argparse.ArgumentParser(
        description='Premier League Winner Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --init                    # Setup project structure
  python main.py --scrape --train -g 17    # Scrape data, train model for GW17
  python main.py --update -g 18            # Update for new gameweek
  python main.py --predict                 # Predict using existing model
  python main.py --compare                 # Compare historical predictions
        """
    )

    parser.add_argument('--init', action='store_true',
                        help='Initialize project structure')
    parser.add_argument('--scrape', action='store_true',
                        help='Scrape fresh data from FBRef')
    parser.add_argument('--train', action='store_true',
                        help='Train/retrain the model')
    parser.add_argument('--predict', action='store_true',
                        help='Make predictions (uses existing model if not training)')
    parser.add_argument('--update', action='store_true',
                        help='Update with new gameweek (scrape + train)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare predictions across gameweeks')
    parser.add_argument('--schedule', action='store_true',
                        help='Run automated weekly updates')
    parser.add_argument('-g', '--gameweek', type=int, default=17,
                        help='Current gameweek number (default: 17)')

    args = parser.parse_args()

    # Handle init
    if args.init:
        create_project_structure()
        return

    # Handle compare
    if args.compare:
        compare_gameweeks()
        return

    # Handle schedule
    if args.schedule:
        run_scheduled()
        return

    # Handle update (shorthand for scrape + train)
    if args.update:
        args.scrape = True
        args.train = True

    # Default: if no action specified, run full pipeline
    if not any([args.scrape, args.train, args.predict]):
        print("No action specified. Running full pipeline...")
        args.scrape = True
        args.train = True
        args.predict = True

    # Run prediction
    run_prediction(
        gameweek=args.gameweek,
        scrape=args.scrape,
        train=args.train
    )


if __name__ == "__main__":
    main()