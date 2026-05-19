"""MCP server exposing the Betman betting predictor as callable tools.

Run with:
    python mcp_server.py

Or wire into an MCP client config (e.g. Claude Desktop / Claude Code):
    {
      "mcpServers": {
        "betman": {
          "command": "python",
          "args": ["C:/Projects/betman-betting-predictor/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit(
        "The 'mcp' package is required. Install with: python -m pip install mcp"
    ) from exc

from betman_predictor.baseball_data import BaseballEnricher, NULL_ENRICHMENT
from betman_predictor.client import BetmanClient
from betman_predictor.config import ALL_MARKET_KEYS, MARKETS, MARKET_ALIASES
from betman_predictor.double_chance import (
    expected_hit_summary,
    select_double_chances,
    stake_breakdown,
)
from betman_predictor.ml_predictor import (
    MLConfig,
    MLPredictor,
    ensemble_predictions,
    sklearn_available,
)
from betman_predictor.models import MarketDefinition, RoundReference
from betman_predictor.predictor import (
    HistoricalPredictor,
    extract_match_records,
    extract_vote_distribution,
)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("betman-predictor")


def _client() -> BetmanClient:
    return BetmanClient(cache_dir=CACHE_DIR, timeout_seconds=20, max_retries=4)


def _resolve_market(market: str) -> MarketDefinition:
    key = MARKET_ALIASES.get(market.lower())
    if key is None:
        valid = ", ".join(ALL_MARKET_KEYS)
        raise ValueError(f"Unknown market '{market}'. Valid: {valid}")
    return MARKETS[key]


def _choose_target_round(
    client: BetmanClient,
    market: MarketDefinition,
    override_round: int | None,
    refresh: bool,
) -> RoundReference:
    if override_round is not None:
        return RoundReference(
            gm_id=market.gm_id,
            gm_ts=override_round,
            round_no=None,
            round_year=None,
            sale_status=None,
            source="override",
            game_name=market.name_ko,
        )
    return client.discover_target_round(market, refresh=refresh)


def _load_history(
    client: BetmanClient,
    market: MarketDefinition,
    target_round: RoundReference,
    history_rounds: int,
    refresh: bool,
) -> tuple[list, dict]:
    fetch_limit = history_rounds + 12
    rounds = client.get_closed_rounds(market.gm_id, limit=fetch_limit, refresh=refresh)
    candidates: list[RoundReference] = []
    for row in rounds:
        gm_ts = int(row.get("gmTs", 0))
        if gm_ts >= target_round.gm_ts:
            continue
        if row.get("saleStatus") != "PayoStart":
            continue
        candidates.append(
            RoundReference(
                gm_id=market.gm_id,
                gm_ts=gm_ts,
                round_no=int(row.get("gmOsidTs") or 0) or None,
                round_year=int(row.get("gmOsidTsYear") or 0) or None,
                sale_status=row.get("saleStatus"),
                source="history",
                game_name=(row.get("gameMaster") or {}).get("gameName"),
            )
        )
        if len(candidates) >= history_rounds:
            break

    if not candidates:
        raise RuntimeError(f"No completed historical rounds for {market.name_ko}.")

    matches: list = []
    vote_lookup: dict = {}
    for round_ref in candidates:
        detail = client.get_round_detail(
            gm_id=round_ref.gm_id,
            gm_ts=round_ref.gm_ts,
            game_year="" if round_ref.round_year is None else str(round_ref.round_year),
            refresh=refresh,
        )
        round_matches = extract_match_records(detail, market, round_ref)
        round_matches = [m for m in round_matches if m.result_code in market.result_codes]
        matches.extend(round_matches)
        for match_seq, shares in extract_vote_distribution(detail, market).items():
            vote_lookup[(round_ref.gm_ts, match_seq)] = shares
    return matches, vote_lookup


def _build_predictions(
    market: MarketDefinition,
    history_matches: list,
    target_matches: list,
    vote_lookup: dict,
    model: str,
    ml_algo: str,
    ensemble_weight: float,
    enricher: BaseballEnricher | None,
) -> list:
    if model in ("ml", "ensemble") and not sklearn_available():
        model = "elo"

    if model == "elo":
        predictor = HistoricalPredictor(market)
        predictor.fit(history_matches)
        return predictor.predict_round(target_matches)

    ml_config = MLConfig(algorithm=ml_algo, calibrate=True)
    if model == "ml":
        ml = MLPredictor(market, ml_config, enricher=enricher)
        ml.fit(history_matches, vote_lookup=vote_lookup)
        return ml.predict_round(target_matches, vote_lookup=vote_lookup)

    elo = HistoricalPredictor(market)
    elo.fit(history_matches)
    elo_preds = elo.predict_round(target_matches)
    ml = MLPredictor(market, ml_config, enricher=enricher)
    ml.fit(history_matches, vote_lookup=vote_lookup)
    ml_preds = ml.predict_round(target_matches, vote_lookup=vote_lookup)
    weight = max(0.0, min(1.0, ensemble_weight))
    return ensemble_predictions(market, elo_preds, ml_preds, weight_primary=weight)


@mcp.tool()
def list_markets() -> list[dict[str, Any]]:
    """List the supported Betman markets (soccer 승무패, baseball 승1패)."""
    return [
        {
            "key": m.key,
            "gm_id": m.gm_id,
            "name_ko": m.name_ko,
            "labels": m.label_map,
            "default_history_rounds": m.default_history_rounds,
        }
        for m in MARKETS.values()
    ]


@mcp.tool()
def discover_target_round(market: str, refresh: bool = False) -> dict[str, Any]:
    """Find the next upcoming Betman round for a market.

    Args:
        market: "soccer", "baseball", or aliases ("g011", "g024", "football").
        refresh: If True, bypass the local cache and refetch from Betman.
    """
    market_def = _resolve_market(market)
    target = _choose_target_round(_client(), market_def, override_round=None, refresh=refresh)
    return {
        "market": market_def.key,
        "gm_id": target.gm_id,
        "gm_ts": target.gm_ts,
        "round_no": target.round_no,
        "round_year": target.round_year,
        "sale_status": target.sale_status,
        "source": target.source,
        "game_name": target.game_name,
    }


@mcp.tool()
def get_round_matches(market: str, round: int, refresh: bool = False) -> list[dict[str, Any]]:
    """List the 14 match slots in a specific Betman round without running predictions.

    Args:
        market: "soccer" or "baseball".
        round: gmTs round identifier.
        refresh: If True, bypass cache.
    """
    market_def = _resolve_market(market)
    client = _client()
    target = _choose_target_round(client, market_def, override_round=round, refresh=refresh)
    detail = client.get_round_detail(
        gm_id=target.gm_id,
        gm_ts=target.gm_ts,
        game_year="",
        refresh=refresh,
    )
    matches = extract_match_records(detail, market_def, target)
    return [
        {
            "match_seq": m.match_seq,
            "league_code": m.league_code,
            "league_name": m.league_name,
            "home": m.home_name,
            "away": m.away_name,
            "game_date": m.game_date_str,
            "domestic": m.domestic,
            "result_code": m.result_code,
        }
        for m in matches
    ]


@mcp.tool()
def predict_round(
    market: str,
    round: int | None = None,
    model: str = "ensemble",
    ml_algo: str = "logreg",
    ensemble_weight: float = 0.5,
    history_rounds: int | None = None,
    double_chance_count: int = 0,
    top_n: int = 0,
    match_seqs: list[int] | None = None,
    enrich_mlb: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    """Predict outcomes for a Betman round and return a structured betting plan.

    Args:
        market: "soccer" or "baseball".
        round: gmTs round identifier. Omit to auto-discover the next upcoming round.
        model: "elo", "ml", or "ensemble" (default).
        ml_algo: "logreg" or "gbm" when model is "ml" or "ensemble".
        ensemble_weight: Elo weight in the ensemble blend (0..1). 0.8 is recommended for baseball.
        history_rounds: Training window length. Defaults to market-specific value.
        double_chance_count: Apply 1X/12/X2 coverage to the N least-confident matches.
        top_n: If >0, return only the N most-confident picks.
        match_seqs: Hand-pick specific match numbers (e.g. [4, 9, 14]) for a manual parlay.
        enrich_mlb: Display probable-starter ERA, pythag, park factor for MLB matches.
        refresh: Bypass local cache and refetch from Betman.

    Returns:
        Structured report with predictions, probabilities, bet types, parlay probability,
        and (optionally) MLB enrichment per match.
    """
    market_def = _resolve_market(market)
    client = _client()
    target_round = _choose_target_round(client, market_def, override_round=round, refresh=refresh)
    history = history_rounds or market_def.default_history_rounds

    history_matches, vote_lookup = _load_history(
        client, market_def, target_round, history, refresh
    )
    target_detail = client.get_round_detail(
        gm_id=target_round.gm_id,
        gm_ts=target_round.gm_ts,
        game_year="" if target_round.round_year is None else str(target_round.round_year),
        refresh=refresh,
    )
    target_matches = extract_match_records(target_detail, market_def, target_round)
    for match_seq, shares in extract_vote_distribution(target_detail, market_def).items():
        vote_lookup[(target_round.gm_ts, match_seq)] = shares

    enricher: BaseballEnricher | None = None
    if enrich_mlb and market_def.key == "baseball":
        enricher = BaseballEnricher(cache_root=ROOT / "cache" / "baseball", verbose=False)

    predictions = _build_predictions(
        market=market_def,
        history_matches=history_matches,
        target_matches=target_matches,
        vote_lookup=vote_lookup,
        model=model,
        ml_algo=ml_algo,
        ensemble_weight=ensemble_weight,
        enricher=enricher,
    )

    if match_seqs:
        wanted = set(match_seqs)
        predictions = [p for p in predictions if p.match.match_seq in wanted]

    bet_plan = select_double_chances(market_def, predictions, double_chance_count)
    if top_n > 0:
        bet_plan = sorted(bet_plan, key=lambda i: i.bet_probability, reverse=True)[:top_n]
        bet_plan = sorted(bet_plan, key=lambda i: i.prediction.match.match_seq)

    summary = expected_hit_summary(bet_plan)
    stake_info = stake_breakdown(bet_plan, 0.0)

    enrichment_by_seq: dict[int, Any] = {}
    if enricher is not None:
        for m in target_matches:
            if m.league_name != "MLB":
                continue
            try:
                data = enricher.lookup(m)
            except Exception:
                continue
            if data is NULL_ENRICHMENT:
                continue
            enrichment_by_seq[m.match_seq] = data

    items = []
    for item in bet_plan:
        payload = {
            "match_seq": item.prediction.match.match_seq,
            "league_name": item.prediction.match.league_name,
            "game_date": item.prediction.match.game_date_str,
            "home_team": item.prediction.match.home_name,
            "away_team": item.prediction.match.away_name,
            "pick": item.prediction.pick_label,
            "pick_code": item.prediction.pick_code,
            "bet_type": item.bet_type,
            "bet_label": item.bet_label,
            "covered_codes": list(item.covered_codes),
            "bet_probability_pct": round(item.bet_probability * 100, 2),
            "probabilities_pct": {
                market_def.label_map[code]: round(prob * 100, 2)
                for code, prob in item.prediction.probabilities.items()
            },
            "rating_gap": round(item.prediction.rating_gap, 2),
            "sample_scope": item.prediction.sample_scope,
        }
        enrichment = enrichment_by_seq.get(item.prediction.match.match_seq)
        if enrichment is not None:
            payload["enrichment"] = {
                "home_starter_era": enrichment.home_starter_era,
                "away_starter_era": enrichment.away_starter_era,
                "home_recent_pythag": enrichment.home_recent_pythag,
                "away_recent_pythag": enrichment.away_recent_pythag,
                "home_recent_runs_per_game": enrichment.home_recent_runs_per_game,
                "away_recent_runs_per_game": enrichment.away_recent_runs_per_game,
                "park_factor": enrichment.park_factor,
            }
        items.append(payload)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market": market_def.key,
        "market_name": market_def.name_ko,
        "gm_id": market_def.gm_id,
        "gm_ts": target_round.gm_ts,
        "round_no": target_round.round_no,
        "source": target_round.source,
        "model": model,
        "ml_algo": ml_algo if model in ("ml", "ensemble") else None,
        "ensemble_weight": ensemble_weight if model == "ensemble" else None,
        "history_rounds": history,
        "summary": summary,
        "stake": stake_info,
        "predictions": items,
    }


@mcp.tool()
def list_recent_rounds(market: str, limit: int = 10, refresh: bool = False) -> list[dict[str, Any]]:
    """List the most recent closed rounds for a market.

    Args:
        market: "soccer" or "baseball".
        limit: Maximum number of rounds to return (default 10).
        refresh: Bypass cache.
    """
    market_def = _resolve_market(market)
    rounds = _client().get_closed_rounds(market_def.gm_id, limit=limit, refresh=refresh)
    return [
        {
            "gm_ts": int(r.get("gmTs", 0)),
            "round_no": int(r.get("gmOsidTs") or 0) or None,
            "round_year": int(r.get("gmOsidTsYear") or 0) or None,
            "sale_status": r.get("saleStatus"),
            "game_name": (r.get("gameMaster") or {}).get("gameName"),
        }
        for r in rounds[:limit]
    ]


if __name__ == "__main__":
    mcp.run()
