"""Predict-time lineup-based Elo penalty.

Soccer outcomes hinge heavily on whether the top 3-5 players are in the XI.
This module applies a predict-time Elo penalty for missing key players whose
absence is publicly known (announced lineup, injury list, suspension).

Predict-time only — does NOT touch training. The model still learns from
results that already bake in whoever played; we only adjust ratings forward
when we have late-breaking team news the trained model couldn't have seen.

Two JSON files:

  players.json — per-team key-player Elo weights:
    {
      "<league_code>": {
        "<betman_team_name>": {
          "<player_name>": <elo_value>,
          ...
        }
      }
    }
    elo_value is the Elo points the player is "worth" relative to a replacement.
    Rough guide: solid starter ~25, top-5 player ~50, world-class ~80.

  lineups.json — per-match missing players (run-time team news):
    {
      "<gm_ts>": {
        "<match_seq>": {
          "home_missing": ["<player_name>", ...],
          "away_missing": ["<player_name>", ...]
        }
      }
    }

The penalty per team is the sum of elo_values for missing players whose
names appear in players.json[league_code][team_name]. Unknown names contribute 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LineupPenalty:
    home_penalty: float
    away_penalty: float
    home_missing: tuple[str, ...]
    away_missing: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return self.home_penalty == 0.0 and self.away_penalty == 0.0


NULL_PENALTY = LineupPenalty(0.0, 0.0, (), ())


class LineupAdjuster:
    def __init__(
        self,
        players_path: Path | None = None,
        lineups_path: Path | None = None,
    ) -> None:
        self.players: dict = {}
        self.lineups: dict = {}
        if players_path and players_path.exists():
            with players_path.open(encoding="utf-8") as handle:
                self.players = json.load(handle)
        if lineups_path and lineups_path.exists():
            with lineups_path.open(encoding="utf-8") as handle:
                self.lineups = json.load(handle)

    @property
    def has_data(self) -> bool:
        return bool(self.lineups)

    def penalty_for(
        self,
        gm_ts: int,
        match_seq: int,
        league_code: str,
        home_team: str,
        away_team: str,
    ) -> LineupPenalty:
        round_block = self.lineups.get(str(gm_ts)) or self.lineups.get(gm_ts) or {}
        match_block = round_block.get(str(match_seq)) or round_block.get(match_seq) or {}
        if not match_block:
            return NULL_PENALTY
        home_missing = tuple(match_block.get("home_missing", []) or [])
        away_missing = tuple(match_block.get("away_missing", []) or [])
        if not home_missing and not away_missing:
            return NULL_PENALTY

        league_players = self.players.get(league_code, {}) or {}
        home_team_players = league_players.get(home_team, {}) or {}
        away_team_players = league_players.get(away_team, {}) or {}
        h_pen = sum(float(home_team_players.get(p, 0.0)) for p in home_missing)
        a_pen = sum(float(away_team_players.get(p, 0.0)) for p in away_missing)
        return LineupPenalty(h_pen, a_pen, home_missing, away_missing)
