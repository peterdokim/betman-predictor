from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MarketDefinition:
    key: str
    gm_id: str
    name_ko: str
    cli_name: str
    label_map: dict[str, str]
    default_history_rounds: int
    home_advantage: float
    k_factor: float
    bandwidth: float
    recency_decay: float
    min_same_league_samples: int
    prior_weight: float

    @property
    def ordered_codes(self) -> tuple[str, ...]:
        return tuple(self.label_map.keys())

    @property
    def result_codes(self) -> set[str]:
        return set(self.label_map)


@dataclass(frozen=True)
class RoundReference:
    gm_id: str
    gm_ts: int
    round_no: int | None
    round_year: int | None
    sale_status: str | None
    source: str
    game_name: str | None = None


@dataclass(frozen=True)
class MatchRecord:
    market_key: str
    gm_id: str
    gm_ts: int
    round_no: int | None
    round_year: int | None
    match_seq: int
    league_code: str
    league_name: str
    home_id: str
    home_name: str
    away_id: str
    away_name: str
    game_datetime: datetime | None
    game_date_str: str
    domestic: bool
    result_code: str | None


@dataclass(frozen=True)
class Prediction:
    match: MatchRecord
    pick_code: str
    pick_label: str
    probabilities: dict[str, float]
    home_rating: float
    away_rating: float
    rating_gap: float
    sample_scope: str


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    cache_dir: Path

