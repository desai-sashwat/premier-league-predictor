"""
Fixture Difficulty Module
Analyzes remaining fixtures and calculates difficulty ratings
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixtureDifficultyAnalyzer:
    """
    Analyzes fixture difficulty based on:
    - Opponent strength (current form, historical performance)
    - Home/Away advantage
    - Big match factor
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Team strength ratings (1-10 scale, based on 2025/26 season form)
        self.team_ratings = self._calculate_team_ratings()

        # Home advantage factor
        self.home_advantage = 0.15

        # Remaining fixtures
        self.remaining_fixtures = self._load_remaining_fixtures()

    def _calculate_team_ratings(self) -> Dict[str, float]:
        """
        Calculate team strength ratings based on current season.
        Scale: 1-10 (10 = strongest)
        Dynamically calculated based on standings if available.
        """
        # Default ratings (will be updated with real data)
        ratings = {
            'Liverpool': 9.2,
            'Arsenal': 9.0,
            'Manchester City': 9.0,
            'Chelsea': 8.5,
            'Newcastle': 8.0,
            'Nottingham Forest': 7.8,
            'Tottenham': 7.8,
            'Aston Villa': 7.5,
            'Manchester Utd': 7.5,
            'Brighton': 7.5,
            'Bournemouth': 7.2,
            'Fulham': 7.0,
            'West Ham': 7.0,
            'Brentford': 6.8,
            'Crystal Palace': 6.5,
            'Everton': 6.2,
            'Wolves': 6.0,
            'Leicester': 5.8,
            'Ipswich': 5.5,
            'Southampton': 4.5,
            # Backup for any team not listed
            'Leeds': 6.0,
            'Burnley': 5.8,
            'Sheffield Utd': 5.5,
        }
        return ratings

    def update_ratings_from_standings(self, standings: pd.DataFrame):
        """Update team ratings based on current standings."""
        if standings.empty:
            return

        # Calculate ratings based on points per game
        if 'ppg' not in standings.columns:
            games = standings.get('games', standings.get('mp', 17))
            points = standings.get('points', standings.get('pts', 0))
            standings = standings.copy()
            standings['ppg'] = points / games.replace(0, 1)

        # Normalize PPG to 1-10 scale
        max_ppg = standings['ppg'].max()
        min_ppg = standings['ppg'].min()

        for _, row in standings.iterrows():
            team = row.get('team', row.get('squad', ''))
            ppg = row.get('ppg', 1.5)

            if team and max_ppg > min_ppg:
                # Scale to 4-10 range (no team below 4)
                rating = 4 + 6 * (ppg - min_ppg) / (max_ppg - min_ppg)
                self.team_ratings[team] = round(rating, 1)

    def _load_remaining_fixtures(self) -> Dict[str, List[Dict]]:
        """
        Load remaining Premier League fixtures for 2025/26 season.
        Format: (opponent, is_home, gameweek)
        """
        # Sample fixtures for key teams (GW18 onwards)
        remaining = {
            'Liverpool': [
                ('Leeds', True, 18), ('West Ham', False, 19), ('Manchester Utd', True, 20),
                ('Nottingham Forest', False, 21), ('Brentford', True, 22), ('Bournemouth', False, 23),
                ('Sheffield Utd', True, 24), ('Everton', False, 25), ('Manchester City', True, 26),
                ('Newcastle', False, 27), ('Burnley', True, 28), ('Chelsea', False, 29),
                ('Brighton', True, 30), ('Fulham', False, 31), ('Wolves', True, 32),
                ('Aston Villa', False, 33), ('Tottenham', True, 34), ('West Ham', True, 35),
                ('Arsenal', False, 36), ('Crystal Palace', True, 37), ('Brighton', False, 38),
            ],
            'Arsenal': [
                ('Sheffield Utd', False, 18), ('Brentford', True, 19), ('Brighton', False, 20),
                ('Wolves', True, 21), ('Tottenham', False, 22), ('Aston Villa', True, 23),
                ('Manchester City', False, 24), ('Leeds', True, 25), ('West Ham', False, 26),
                ('Nottingham Forest', True, 27), ('Everton', False, 28), ('Burnley', True, 29),
                ('Bournemouth', False, 30), ('Newcastle', True, 31), ('Crystal Palace', False, 32),
                ('Fulham', True, 33), ('Manchester Utd', True, 34), ('Chelsea', False, 35),
                ('Liverpool', True, 36), ('Leeds', False, 37), ('Brighton', True, 38),
            ],
            'Manchester City': [
                ('Everton', False, 18), ('Leeds', True, 19), ('West Ham', False, 20),
                ('Brentford', True, 21), ('Sheffield Utd', False, 22), ('Chelsea', False, 23),
                ('Arsenal', True, 24), ('Newcastle', False, 25), ('Liverpool', False, 26),
                ('Tottenham', True, 27), ('Bournemouth', True, 28), ('Nottingham Forest', False, 29),
                ('Crystal Palace', True, 30), ('Aston Villa', False, 31), ('Wolves', True, 32),
                ('Burnley', False, 33), ('Brighton', True, 34), ('Everton', True, 35),
                ('Fulham', False, 36), ('Manchester Utd', True, 37), ('West Ham', True, 38),
            ],
            'Chelsea': [
                ('Fulham', True, 18), ('Sheffield Utd', False, 19), ('Bournemouth', True, 20),
                ('Crystal Palace', True, 21), ('Wolves', False, 22), ('Manchester City', True, 23),
                ('West Ham', False, 24), ('Everton', True, 25), ('Aston Villa', False, 26),
                ('Burnley', True, 27), ('Brentford', False, 28), ('Leeds', True, 29),
                ('Liverpool', True, 30), ('Newcastle', False, 31), ('Manchester Utd', True, 32),
                ('Tottenham', False, 33), ('Brighton', True, 34), ('Nottingham Forest', False, 35),
                ('Arsenal', True, 36), ('Fulham', False, 37), ('Wolves', True, 38),
            ],
            'Newcastle': [
                ('Burnley', True, 18), ('Fulham', False, 19), ('Nottingham Forest', False, 20),
                ('Sheffield Utd', True, 21), ('West Ham', False, 22), ('Tottenham', True, 23),
                ('Bournemouth', False, 24), ('Manchester City', True, 25), ('Crystal Palace', False, 26),
                ('Liverpool', True, 27), ('Aston Villa', False, 28), ('Everton', True, 29),
                ('Brentford', False, 30), ('Arsenal', False, 31), ('Brighton', True, 32),
                ('Manchester Utd', False, 33), ('Leeds', True, 34), ('Chelsea', True, 35),
                ('Wolves', False, 36), ('Tottenham', True, 37), ('Burnley', False, 38),
            ],
        }

        # Generate for remaining teams
        all_teams = list(self.team_ratings.keys())
        for team in all_teams:
            if team not in remaining:
                remaining[team] = self._generate_fixtures_for_team(team, all_teams)

        return remaining

    def _generate_fixtures_for_team(self, team: str, all_teams: List[str]) -> List[Tuple]:
        """Generate placeholder fixtures for a team."""
        opponents = [t for t in all_teams if t != team]
        fixtures = []

        np.random.seed(hash(team) % 2**32)
        np.random.shuffle(opponents)

        for i, opp in enumerate(opponents[:21]):
            is_home = i % 2 == 0
            gw = 18 + i
            fixtures.append((opp, is_home, gw))

        return fixtures

    def calculate_fixture_difficulty(self, team: str, n_fixtures: int = None) -> Dict:
        """Calculate fixture difficulty for a team."""
        fixtures = self.remaining_fixtures.get(team, [])

        if n_fixtures:
            fixtures = fixtures[:n_fixtures]

        if not fixtures:
            return {
                'team': team,
                'avg_difficulty': 5.0,
                'total_difficulty': 0,
                'max_difficulty': 5.0,
                'min_difficulty': 5.0,
                'std_difficulty': 0,
                'hard_games': 0,
                'easy_games': 0,
                'home_games': 0,
                'away_games': 0,
                'fixture_count': 0
            }

        difficulties = []
        hard_count, easy_count = 0, 0
        home_count, away_count = 0, 0

        for fixture in fixtures:
            opponent, is_home, gw = fixture
            opp_rating = self.team_ratings.get(opponent, 5.0)

            if is_home:
                difficulty = opp_rating * (1 - self.home_advantage)
                home_count += 1
            else:
                difficulty = opp_rating * (1 + self.home_advantage)
                away_count += 1

            difficulties.append(difficulty)

            if difficulty >= 8.0:
                hard_count += 1
            elif difficulty <= 5.5:
                easy_count += 1

        return {
            'team': team,
            'avg_difficulty': np.mean(difficulties) if difficulties else 5.0,
            'total_difficulty': np.sum(difficulties) if difficulties else 0,
            'max_difficulty': np.max(difficulties) if difficulties else 5.0,
            'min_difficulty': np.min(difficulties) if difficulties else 5.0,
            'std_difficulty': np.std(difficulties) if len(difficulties) > 1 else 0,
            'hard_games': hard_count,
            'easy_games': easy_count,
            'home_games': home_count,
            'away_games': away_count,
            'fixture_count': len(fixtures)
        }

    def calculate_next5_difficulty(self, team: str) -> Dict:
        """Calculate difficulty of next 5 fixtures."""
        return self.calculate_fixture_difficulty(team, n_fixtures=5)

    def calculate_run_in_difficulty(self, team: str, last_n: int = 10) -> Dict:
        """Calculate difficulty of final fixtures."""
        fixtures = self.remaining_fixtures.get(team, [])

        if len(fixtures) > last_n:
            fixtures = fixtures[-last_n:]

        difficulties = []
        for fixture in fixtures:
            opponent, is_home, gw = fixture
            opp_rating = self.team_ratings.get(opponent, 5.0)

            if is_home:
                difficulty = opp_rating * (1 - self.home_advantage)
            else:
                difficulty = opp_rating * (1 + self.home_advantage)

            difficulties.append(difficulty)

        return {
            'team': team,
            'run_in_avg_difficulty': np.mean(difficulties) if difficulties else 5.0,
            'run_in_total': np.sum(difficulties) if difficulties else 0,
        }

    def create_fixture_features(self, teams: List[str]) -> pd.DataFrame:
        """Create fixture-based features for all teams."""
        features = []

        for team in teams:
            overall = self.calculate_fixture_difficulty(team)
            next_5 = self.calculate_next5_difficulty(team)
            run_in = self.calculate_run_in_difficulty(team, last_n=10)

            # Big 6 games remaining
            big_6 = ['Liverpool', 'Arsenal', 'Chelsea', 'Manchester City', 'Tottenham', 'Manchester Utd']
            fixtures = self.remaining_fixtures.get(team, [])
            big_6_games = sum(1 for f in fixtures if f[0] in big_6)
            big_6_home = sum(1 for f in fixtures if f[0] in big_6 and f[1])

            features.append({
                'team': team,
                'fixture_avg_difficulty': overall['avg_difficulty'],
                'fixture_total_difficulty': overall['total_difficulty'],
                'fixture_std_difficulty': overall['std_difficulty'],
                'fixture_hard_games': overall['hard_games'],
                'fixture_easy_games': overall['easy_games'],
                'fixture_home_remaining': overall['home_games'],
                'fixture_away_remaining': overall['away_games'],
                'fixture_next5_difficulty': next_5['avg_difficulty'],
                'fixture_next5_hard': next_5['hard_games'],
                'fixture_next5_easy': next_5['easy_games'],
                'fixture_runin_difficulty': run_in['run_in_avg_difficulty'],
                'fixture_big6_remaining': big_6_games,
                'fixture_big6_home': big_6_home,
                'fixture_advantage_score': 10 - overall['avg_difficulty'],
            })

        return pd.DataFrame(features)

    def get_difficulty_comparison(self, teams: List[str] = None) -> pd.DataFrame:
        """Compare fixture difficulty across teams."""
        if teams is None:
            teams = list(self.team_ratings.keys())

        comparison = []
        for team in teams:
            diff = self.calculate_fixture_difficulty(team)
            comparison.append({
                'team': team,
                'avg_difficulty': diff['avg_difficulty'],
                'hard_games': diff['hard_games'],
                'easy_games': diff['easy_games'],
                'home_advantage': diff['home_games'] - diff['away_games'],
            })

        df = pd.DataFrame(comparison)
        df['difficulty_rank'] = df['avg_difficulty'].rank(ascending=False)
        df['fixture_advantage'] = df['avg_difficulty'].max() - df['avg_difficulty']

        return df.sort_values('avg_difficulty')

    def predict_points_from_fixtures(self, team: str, current_ppg: float) -> Dict:
        """Estimate points from remaining fixtures."""
        fixtures = self.remaining_fixtures.get(team, [])
        expected_points = 0
        fixture_predictions = []

        for fixture in fixtures:
            opponent, is_home, gw = fixture
            opp_rating = self.team_ratings.get(opponent, 5.0)
            team_rating = self.team_ratings.get(team, 5.0)

            rating_diff = team_rating - opp_rating
            if is_home:
                rating_diff += 1.0

            win_prob = 0.5 + (rating_diff * 0.05)
            win_prob = max(0.1, min(0.9, win_prob))

            draw_prob = 0.25 - abs(rating_diff) * 0.02
            draw_prob = max(0.1, min(0.35, draw_prob))

            expected = win_prob * 3 + draw_prob * 1
            expected_points += expected

            fixture_predictions.append({
                'opponent': opponent,
                'home': is_home,
                'gameweek': gw,
                'expected_points': expected,
                'win_prob': win_prob
            })

        return {
            'team': team,
            'total_expected_points': expected_points,
            'games_remaining': len(fixtures),
            'expected_ppg': expected_points / max(len(fixtures), 1),
            'fixture_breakdown': fixture_predictions
        }

    def update_team_rating(self, team: str, new_rating: float):
        """Update a team's strength rating."""
        if 1 <= new_rating <= 10:
            self.team_ratings[team] = new_rating
            logger.info(f"Updated {team} rating to {new_rating}")


def main():
    """Test fixture difficulty module."""
    analyzer = FixtureDifficultyAnalyzer()

    print("=== 2025/26 Fixture Difficulty Comparison ===")
    comparison = analyzer.get_difficulty_comparison(
        ['Liverpool', 'Arsenal', 'Manchester City', 'Chelsea', 'Newcastle']
    )
    print(comparison.to_string())

    print("\n=== Next 5 Games Difficulty ===")
    for team in ['Liverpool', 'Arsenal', 'Manchester City', 'Chelsea']:
        next5 = analyzer.calculate_next5_difficulty(team)
        print(f"{team}: {next5['avg_difficulty']:.2f} avg difficulty")


if __name__ == "__main__":
    main()