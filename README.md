# We'll create a reusable Python script for match score prediction using a Poisson model,
# plus run a quick demo and display probabilities in a table for the user.

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

SCRIPT_CODE = r'''
"""
Poisson-based Football (Soccer) Match Score Predictor
-----------------------------------------------------
- Estimates team attack/defense strengths from a dataset of past match results
- Computes scoreline probabilities for two given teams using a Poisson model
- Outputs win/draw probabilities, most likely scores, and an N x N probability matrix

USAGE (programmatic):
    from poisson_match_predictor import MatchModel
    model = MatchModel.from_dataframe(matches_df)  # or MatchModel.from_csv("matches.csv")
    result = model.predict("Team A", "Team B")
    print(result["summary"])
    print(result["matrix"])  # pandas.DataFrame of scoreline probabilities

USAGE (CLI):
    python poisson_match_predictor.py --csv matches.csv --home "Team A" --away "Team B" --max_goals 6

CSV format expected (columns, case-insensitive):
    date, home_team, away_team, home_goals, away_goals
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import argparse

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    needed = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    remap = {}
    for need in needed:
        # choose a matching column ignoring case/underscores/spaces
        target = None
        for c in df.columns:
            key = c.lower().replace("_","").replace(" ","")
            if key == need.replace("_",""):
                target = c
                break
        if target is None:
            raise ValueError(f"Missing required column like '{need}'")
        remap[target] = need
    out = df.rename(columns=remap).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    return out.dropna(subset=["home_team","away_team","home_goals","away_goals"])

@dataclass
class TeamStrength:
    attack: float  # >1 means stronger than average
    defense: float # <1 means concedes fewer than average

class MatchModel:
    def __init__(self, strengths: Dict[str, TeamStrength], home_advantage: float, league_avg_goals_home: float, league_avg_goals_away: float):
        self.strengths = strengths
        self.home_advantage = home_advantage
        self.league_avg_goals_home = league_avg_goals_home
        self.league_avg_goals_away = league_avg_goals_away

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, decay_half_life_games: Optional[float]=None):
        df = _standardize_columns(df)
        # optional exponential decay weighting by game recency (by row order or date)
        if decay_half_life_games is not None and decay_half_life_games > 0:
            # If dates are present, sort by date; else by original order
            if df["date"].notna().all():
                df = df.sort_values("date")
            df = df.reset_index(drop=True)
            idx = np.arange(len(df))
            lam = np.log(2)/decay_half_life_games
            weights = np.exp(lam*(idx - idx.max()))
            df["_w"] = weights
        else:
            df["_w"] = 1.0

        league_avg_goals_home = (df["_w"] * df["home_goals"]).sum() / df["_w"].sum()
        league_avg_goals_away = (df["_w"] * df["away_goals"]).sum() / df["_w"].sum()

        # Compute team-level for/against per match (separately for home/away), then normalize
        teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
        team_stats = {t: {"home_for":0.0,"home_gp":0.0,"home_against":0.0,
                          "away_for":0.0,"away_gp":0.0,"away_against":0.0} for t in teams}

        for _, row in df.iterrows():
            w = row["_w"]
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            team_stats[ht]["home_for"] += w*hg
            team_stats[ht]["home_against"] += w*ag
            team_stats[ht]["home_gp"] += w

            team_stats[at]["away_for"] += w*ag
            team_stats[at]["away_against"] += w*hg
            team_stats[at]["away_gp"] += w

        strengths: Dict[str, TeamStrength] = {}
        eps = 1e-8
        for t, s in team_stats.items():
            home_att = (s["home_for"] / max(s["home_gp"], eps)) / max(league_avg_goals_home, eps) if s["home_gp"]>0 else 1.0
            away_att = (s["away_for"] / max(s["away_gp"], eps)) / max(league_avg_goals_away, eps) if s["away_gp"]>0 else 1.0
            # combine home/away attack into a single attack factor
            attack = 0.6*home_att + 0.4*away_att if s["home_gp"]>0 and s["away_gp"]>0 else (home_att if s["home_gp"]>0 else away_att)

            home_def = (s["home_against"] / max(s["home_gp"], eps)) / max(league_avg_goals_away, eps) if s["home_gp"]>0 else 1.0
            away_def = (s["away_against"] / max(s["away_gp"], eps)) / max(league_avg_goals_home, eps) if s["away_gp"]>0 else 1.0
            defense = 0.6*home_def + 0.4*away_def if s["home_gp"]>0 and s["away_gp"]>0 else (home_def if s["home_gp"]>0 else away_def)

            strengths[t] = TeamStrength(attack=float(attack), defense=float(defense))

        # Estimate home advantage as ratio of home goals to away goals across the league
        home_advantage = max(league_avg_goals_home / max(league_avg_goals_away, eps), 0.5)
        return cls(strengths, home_advantage, float(league_avg_goals_home), float(league_avg_goals_away))

    @classmethod
    def from_csv(cls, path: str, **kwargs):
        df = pd.read_csv(path)
        return cls.from_dataframe(df, **kwargs)

    def team_factor(self, team: str) -> TeamStrength:
        # Fallback to neutral (1.0, 1.0) if team not in model
        return self.strengths.get(team, TeamStrength(1.0, 1.0))

    def lambdas(self, home_team: str, away_team: str) -> Tuple[float,float]:
        h = self.team_factor(home_team)
        a = self.team_factor(away_team)
        lam_home = self.league_avg_goals_home * self.home_advantage * h.attack * max(1e-6, 2 - a.defense)
        lam_away = self.league_avg_goals_away * a.attack * max(1e-6, 2 - h.defense)
        # Clip to reasonable range
        lam_home = float(np.clip(lam_home, 0.05, 4.5))
        lam_away = float(np.clip(lam_away, 0.05, 4.5))
        return lam_home, lam_away

    def predict(self, home_team: str, away_team: str, max_goals: int = 6) -> Dict[str, object]:
        lam_h, lam_a = self.lambdas(home_team, away_team)
        # Poisson probabilities for 0..max_goals (with tail mass at 'max_goals+')
        k = np.arange(0, max_goals+1)
        def pois_pmf(lam):
            from math import exp, factorial
            return np.array([np.exp(-lam) * (lam**i) / np.math.factorial(i) for i in k])

        ph = pois_pmf(lam_h)
        pa = pois_pmf(lam_a)

        # probability matrix
        P = np.outer(ph, pa)
        # normalize to ensure numerical stability (tail mass ignored)
        P = P / P.sum()

        # outcomes
        home_win = np.triu(P, k=1).sum()
        draw = np.trace(P)
        away_win = np.tril(P, k=-1).sum()

        # most likely scores
        idx = np.unravel_index(np.argmax(P), P.shape)
        ml_home, ml_away = int(idx[0]), int(idx[1])

        matrix = pd.DataFrame(P, index=[f"H{i}" for i in range(P.shape[0])], columns=[f"A{j}" for j in range(P.shape[1])])
        summary = {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "home_win_prob": float(home_win),
            "draw_prob": float(draw),
            "away_win_prob": float(away_win),
            "expected_goals_home": float((np.arange(P.shape[0]) * P.sum(axis=1)).sum()),
            "expected_goals_away": float((np.arange(P.shape[1]) * P.sum(axis=0)).sum()),
            "most_likely_score": f"{ml_home}-{ml_away}"
        }
        return {"summary": summary, "matrix": matrix}

def _demo():
    # Minimal demo dataset (fictional small league)
    demo = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="7D"),
        "home_team": ["Alpha","Bravo","Charlie","Delta","Alpha","Bravo","Charlie","Delta","Alpha","Bravo"],
        "away_team": ["Bravo","Charlie","Delta","Alpha","Charlie","Delta","Alpha","Bravo","Delta","Alpha"],
        "home_goals": [2,1,3,0,2,1,2,1,1,2],
        "away_goals": [1,1,1,2,2,2,0,1,1,0],
    })
    model = MatchModel.from_dataframe(demo, decay_half_life_games=8)
    res = model.predict("Alpha", "Charlie", max_goals=6)
    return demo, res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson Football Score Predictor")
    parser.add_argument("--csv", type=str, required=True, help="Path to matches CSV")
    parser.add_argument("--home", type=str, required=True, help="Home team name")
    parser.add_argument("--away", type=str, required=True, help="Away team name")
    parser.add_argument("--max_goals", type=int, default=6, help="Max goals per side for matrix")
    parser.add_argument("--half_life", type=float, default=0.0, help="Exponential decay half-life in games (0 to disable)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    model = MatchModel.from_dataframe(df, decay_half_life_games=(args.half_life if args.half_life>0 else None))
    res = model.predict(args.home, args.away, max_goals=args.max_goals)
    print(pd.Series(res["summary"]))
    print()
    print(res["matrix"].round(4))
'''

# Write the script to /mnt/data
path = Path("/mnt/data/poisson_match_predictor.py")
path.write_text(SCRIPT_CODE)

# Run a small demo now to produce a probability matrix for the user to see
from importlib import util, machinery
spec = util.spec_from_file_location("poisson_match_predictor", str(path))
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)

demo_df, demo_res = mod._demo()

# Display the summary and matrix to the user
import caas_jupyter_tools as cj
# Show the matrix as an interactive table
cj.display_dataframe_to_user("Scoreline Probability Matrix (Demo: Alpha vs Charlie)", demo_res["matrix"].round(4))

# Also save the demo dataset for the user
demo_csv_path = Path("/mnt/data/demo_matches.csv")
demo_df.to_csv(demo_csv_path, index=False)

# Prepare a concise printed summary for the notebook output
summary_series = pd.Series(demo_res["summary"])
summary_series
