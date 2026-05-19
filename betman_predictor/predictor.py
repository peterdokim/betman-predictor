from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from betman_predictor.lineup_adjust import LineupAdjuster, NULL_PENALTY
from betman_predictor.models import MarketDefinition, MatchRecord, Prediction, RoundReference


def _parse_datetime(ms_timestamp: int | None) -> datetime | None:
    if not ms_timestamp:
        return None
    return datetime.fromtimestamp(ms_timestamp / 1000, tz=timezone.utc).astimezone()


def _safe_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def extract_match_records(
    detail: dict,
    market: MarketDefinition,
    round_ref: RoundReference,
) -> list[MatchRecord]:
    records: list[MatchRecord] = []
    for row in detail.get("schedulesList", []):
        records.append(
            MatchRecord(
                market_key=market.key,
                gm_id=market.gm_id,
                gm_ts=round_ref.gm_ts,
                round_no=round_ref.round_no,
                round_year=round_ref.round_year,
                match_seq=int(row.get("matchSeq") or 0),
                league_code=_safe_text(row.get("leagueCode"), "UNKNOWN"),
                league_name=_safe_text(row.get("leagueName"), "Unknown League"),
                home_id=_safe_text(row.get("homeId"), _safe_text(row.get("homeName"), "HOME")),
                home_name=_safe_text(row.get("homeName"), "HOME"),
                away_id=_safe_text(row.get("awayId"), _safe_text(row.get("awayName"), "AWAY")),
                away_name=_safe_text(row.get("awayName"), "AWAY"),
                game_datetime=_parse_datetime(row.get("gameDate")),
                game_date_str=_safe_text(row.get("gameDateStr"), ""),
                domestic=bool(row.get("domastic")),
                result_code=_safe_text(row.get("gameResult"), "") or None,
            )
        )
    records.sort(key=lambda item: (item.match_seq, item.home_name, item.away_name))
    return records


def extract_vote_distribution(
    detail: dict,
    market: MarketDefinition,
) -> dict[int, dict[str, float]]:
    """Returns {match_seq: {"A": share, "B": share, "D": share, "_total": votes}}.

    Betman's voteStatus.homeVoteStatusList holds one entry per match (index = matchSeq - 1).
    Each entry's awayVoteStatusList carries 3 outcome buckets in market.ordered_codes order.
    """
    result: dict[int, dict[str, float]] = {}
    vote_status = detail.get("voteStatus") or {}
    home_list = vote_status.get("homeVoteStatusList") or []
    codes = market.ordered_codes

    for index, entry in enumerate(home_list):
        away_list = (entry or {}).get("awayVoteStatusList") or []
        if len(away_list) < len(codes):
            continue
        counts = [float((away_list[i] or {}).get("voteCount") or 0.0) for i in range(len(codes))]
        total = sum(counts)
        if total <= 0:
            continue
        match_seq = index + 1
        shares = {code: counts[i] / total for i, code in enumerate(codes)}
        shares["_total"] = total
        result[match_seq] = shares
    return result


@dataclass(frozen=True)
class CalibrationSample:
    diff: float
    league_code: str
    result_code: str
    order_index: int


class HistoricalPredictor:
    """Small Elo-style model with kernel calibration for 3-way Betman markets."""

    FALLBACK_DATETIME = datetime(1970, 1, 1, tzinfo=timezone.utc)
    SCORE_MAP = {"A": 1.0, "B": 0.5, "D": 0.0}

    def __init__(
        self,
        market: MarketDefinition,
        lineup_adjuster: LineupAdjuster | None = None,
        preseed_by_name: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.market = market
        self.team_ratings: dict[str, float] = {}
        self.samples: list[CalibrationSample] = []
        self.global_counts: Counter[str] = Counter()
        self.fitted = False
        self.lineup_adjuster = lineup_adjuster
        # preseed: {league_code: {team_name: elo_offset_from_1500}}
        # Used to bootstrap teams that appear rarely in Betman history but have
        # rich attack/defence data from external sources (football-data.co.uk).
        self.preseed_by_name = preseed_by_name or {}

    def _team_key(self, league_code: str, team_id: str) -> str:
        return f"{league_code}:{team_id}"

    def _get_rating(self, league_code: str, team_id: str) -> float:
        key = self._team_key(league_code, team_id)
        return self.team_ratings.get(key, 1500.0)

    def _set_rating(self, league_code: str, team_id: str, value: float) -> None:
        key = self._team_key(league_code, team_id)
        self.team_ratings[key] = value

    def _rating_gap(self, match: MatchRecord) -> tuple[float, float, float]:
        home_rating = self._get_rating(match.league_code, match.home_id)
        away_rating = self._get_rating(match.league_code, match.away_id)
        ha = self.market.param_for(match.league_code, "home_advantage")
        diff = home_rating + ha - away_rating
        return home_rating, away_rating, diff

    def _bootstrap_preseed(self, matches: Iterable[MatchRecord]) -> None:
        """Pre-populate team_ratings from external attack/defence data, by team name."""
        if not self.preseed_by_name:
            return
        for match in matches:
            for league, tid, tname in (
                (match.league_code, match.home_id, match.home_name),
                (match.league_code, match.away_id, match.away_name),
            ):
                key = self._team_key(league, tid)
                if key in self.team_ratings:
                    continue
                offset = self.preseed_by_name.get(league, {}).get(tname)
                if offset is not None:
                    self.team_ratings[key] = 1500.0 + float(offset)

    @staticmethod
    def _expected_home_score(diff: float) -> float:
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def fit(self, matches: Iterable[MatchRecord]) -> None:
        ordered = sorted(
            [match for match in matches if match.result_code in self.market.result_codes],
            key=lambda match: (
                match.game_datetime or self.FALLBACK_DATETIME,
                match.gm_ts,
                match.match_seq,
            ),
        )

        self._bootstrap_preseed(ordered)

        for index, match in enumerate(ordered):
            home_rating, away_rating, diff = self._rating_gap(match)
            self.samples.append(
                CalibrationSample(
                    diff=diff,
                    league_code=match.league_code,
                    result_code=match.result_code or "",
                    order_index=index,
                )
            )
            self.global_counts[match.result_code or ""] += 1

            actual_home = self.SCORE_MAP[match.result_code or "B"]
            expected_home = self._expected_home_score(diff)
            k = self.market.param_for(match.league_code, "k_factor")
            delta = k * (actual_home - expected_home)

            self._set_rating(match.league_code, match.home_id, home_rating + delta)
            self._set_rating(match.league_code, match.away_id, away_rating - delta)

        self.fitted = True

    def _estimate_probabilities(self, diff: float, league_code: str) -> tuple[dict[str, float], str]:
        if not self.samples:
            uniform = 1.0 / len(self.market.ordered_codes)
            return ({code: uniform for code in self.market.ordered_codes}, "uniform")

        same_league = [sample for sample in self.samples if sample.league_code == league_code]
        if len(same_league) >= self.market.min_same_league_samples:
            pool = same_league
            scope = "league"
        else:
            pool = self.samples
            scope = "market"

        total_seen = max(len(self.samples), 1)
        priors_total = sum(self.global_counts.values()) or 1
        counts = {
            code: (self.global_counts.get(code, 0) / priors_total) * self.market.prior_weight
            for code in self.market.ordered_codes
        }
        bandwidth = self.market.param_for(league_code, "bandwidth")
        recency_decay = self.market.param_for(league_code, "recency_decay")

        for sample in pool:
            distance_weight = math.exp(-abs(sample.diff - diff) / bandwidth)
            age = total_seen - sample.order_index
            recency_weight = recency_decay**age
            counts[sample.result_code] = counts.get(sample.result_code, 0.0) + (distance_weight * recency_weight)

        total = sum(counts.values()) or 1.0
        probabilities = {code: counts.get(code, 0.0) / total for code in self.market.ordered_codes}
        return probabilities, scope

    def predict_round(self, matches: Iterable[MatchRecord]) -> list[Prediction]:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        match_list = list(matches)
        self._bootstrap_preseed(match_list)

        predictions: list[Prediction] = []
        for match in sorted(match_list, key=lambda item: (item.match_seq, item.home_name, item.away_name)):
            home_rating, away_rating, diff = self._rating_gap(match)

            penalty = NULL_PENALTY
            if self.lineup_adjuster is not None:
                penalty = self.lineup_adjuster.penalty_for(
                    gm_ts=match.gm_ts,
                    match_seq=match.match_seq,
                    league_code=match.league_code,
                    home_team=match.home_name,
                    away_team=match.away_name,
                )
                if not penalty.is_empty:
                    home_rating -= penalty.home_penalty
                    away_rating -= penalty.away_penalty
                    ha = self.market.param_for(match.league_code, "home_advantage")
                    diff = home_rating + ha - away_rating

            probabilities, scope = self._estimate_probabilities(diff, match.league_code)
            if not penalty.is_empty:
                scope = f"{scope}+lineup"
            pick_code = max(probabilities, key=probabilities.get)
            predictions.append(
                Prediction(
                    match=match,
                    pick_code=pick_code,
                    pick_label=self.market.label_map[pick_code],
                    probabilities=probabilities,
                    home_rating=home_rating,
                    away_rating=away_rating,
                    rating_gap=diff,
                    sample_scope=scope,
                )
            )
        return predictions
