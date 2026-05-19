from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from betman_predictor.baseball_data import BaseballEnricher, MatchEnrichment, NULL_ENRICHMENT
from betman_predictor.client import BetmanClient
from betman_predictor.config import ALL_MARKET_KEYS, MARKETS, MARKET_ALIASES
from betman_predictor.double_chance import (
    BetPlanItem,
    expected_hit_summary,
    select_double_chances,
    stake_breakdown,
)
from betman_predictor.external_data import build_preseed
from betman_predictor.lineup_adjust import LineupAdjuster
from betman_predictor.ml_predictor import (
    MLConfig,
    MLPredictor,
    OddsLookup,
    VoteLookup,
    ensemble_predictions,
    sklearn_available,
)
from betman_predictor.models import MarketDefinition, ProjectPaths, RoundReference
from betman_predictor.predictor import (
    HistoricalPredictor,
    extract_match_records,
    extract_vote_distribution,
)


VALID_MODELS = ("elo", "ml", "ensemble")
VALID_ML_ALGOS = ("logreg", "gbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Betman predictor for 축구 승무패 and 야구 승1패 14-game markets."
    )
    parser.add_argument(
        "--market",
        default="all",
        help="one of: all, soccer, baseball",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="override the target gmTs round number for a single market run",
    )
    parser.add_argument(
        "--history-rounds",
        type=int,
        default=None,
        help="number of completed historical rounds to train on",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached API files and refresh from Betman",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="directory for API caches",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the final prediction report to a JSON file",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="HTTP timeout for Betman requests",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="retry count for Betman requests",
    )
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        default="ensemble",
        help="prediction model: elo (lightweight), ml (scikit-learn), ensemble (blend)",
    )
    parser.add_argument(
        "--ml-algo",
        choices=VALID_ML_ALGOS,
        default="logreg",
        help="ML algorithm when --model is ml or ensemble",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.5,
        help="weight for the Elo predictor in --model ensemble (0..1)",
    )
    parser.add_argument(
        "--double-chance-count",
        type=int,
        default=0,
        help="apply double-chance coverage (1X/12/X2) to the top N least-confident matches",
    )
    parser.add_argument(
        "--stake-total",
        type=float,
        default=0.0,
        help="total stake to split evenly across the 14 tickets (only affects display)",
    )
    parser.add_argument(
        "--enrich-mlb",
        action="store_true",
        help="enrich baseball matches with MLB Stats API data (probable starter ERA/WHIP/K9, last-window team form)",
    )
    parser.add_argument(
        "--enrich-features",
        action="store_true",
        help="feed MLB enrichment into the ML model as training features (slow first run; cached after). Implies --enrich-mlb.",
    )
    parser.add_argument(
        "--enrich-workers",
        type=int,
        default=8,
        help="ThreadPoolExecutor size for concurrent MLB Stats API prefetch",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="if >0, print only the N most-confident picks from the round (use 3 or 4 for focused stake)",
    )
    parser.add_argument(
        "--match-seqs",
        type=str,
        default="",
        help="comma-separated match seqs to keep in the report (e.g. '4,9,14' for a 3-game parlay). Applied after --top-n / --double-chance-count.",
    )
    # Soccer-focused improvements (no-ops when the relevant data file is missing).
    parser.add_argument(
        "--lineup-players",
        default=None,
        help="path to JSON: {league_code: {team_name: {player_name: elo_value}}}",
    )
    parser.add_argument(
        "--lineup-missing",
        default=None,
        help="path to JSON: {gm_ts: {match_seq: {home_missing: [...], away_missing: [...]}}}",
    )
    parser.add_argument(
        "--external-history-dir",
        default=None,
        help="directory of football-data.co.uk CSVs to pre-seed Elo ratings (soccer only)",
    )
    parser.add_argument(
        "--odds-file",
        default=None,
        help="path to JSON: {gm_ts: {match_seq: {A: prob, B: prob, D: prob}}} for closing-odds blend (ML only)",
    )
    parser.add_argument(
        "--odds-weight",
        type=float,
        default=0.0,
        help="weight (0..1) of closing-odds in geometric blend with ML model (0 = disabled)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="disable isotonic per-class calibration on a holdout fold",
    )
    return parser.parse_args()


def load_odds_lookup(path: str | None) -> OddsLookup:
    if not path:
        return {}
    odds_path = Path(path)
    if not odds_path.exists():
        progress(f"  [warn] --odds-file '{path}' not found; skipping closing-odds blend")
        return {}
    with odds_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    out: OddsLookup = {}
    for gm_ts, round_block in (raw or {}).items():
        try:
            ts_int = int(gm_ts)
        except (TypeError, ValueError):
            continue
        for match_seq, codes in (round_block or {}).items():
            try:
                seq_int = int(match_seq)
            except (TypeError, ValueError):
                continue
            cleaned = {str(k): float(v) for k, v in (codes or {}).items()}
            total = sum(cleaned.values())
            if total > 0:
                cleaned = {k: v / total for k, v in cleaned.items()}
                out[(ts_int, seq_int)] = cleaned
    return out


def parse_match_seqs(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    seqs: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seqs.add(int(token))
        except ValueError as exc:
            raise SystemExit(f"--match-seqs got non-integer token '{token}'") from exc
    return seqs


def resolve_market_keys(raw_market: str) -> list[str]:
    if raw_market == "all":
        return list(ALL_MARKET_KEYS)

    key = MARKET_ALIASES.get(raw_market.lower())
    if key is None:
        valid = ", ".join(["all", *ALL_MARKET_KEYS])
        raise SystemExit(f"Unknown market '{raw_market}'. Valid values: {valid}")
    return [key]


def build_paths(args: argparse.Namespace) -> ProjectPaths:
    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, cache_dir=cache_dir)


def progress(message: str) -> None:
    print(message, flush=True)


def load_history_matches(
    client: BetmanClient,
    market: MarketDefinition,
    target_round: RoundReference,
    history_rounds: int,
    refresh: bool,
) -> tuple[list, VoteLookup]:
    fetch_limit = history_rounds + 12
    rounds = client.get_closed_rounds(market.gm_id, limit=fetch_limit, refresh=refresh)
    candidates = []
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
        raise RuntimeError(f"No completed historical rounds found for {market.name_ko}.")

    matches = []
    vote_lookup: VoteLookup = {}
    total = len(candidates)
    for index, round_ref in enumerate(candidates, start=1):
        detail = client.get_round_detail(
            gm_id=round_ref.gm_id,
            gm_ts=round_ref.gm_ts,
            game_year="" if round_ref.round_year is None else str(round_ref.round_year),
            refresh=refresh,
        )
        round_matches = extract_match_records(detail, market, round_ref)
        round_matches = [row for row in round_matches if row.result_code in market.result_codes]
        matches.extend(round_matches)

        round_votes = extract_vote_distribution(detail, market)
        for match_seq, shares in round_votes.items():
            vote_lookup[(round_ref.gm_ts, match_seq)] = shares

        if index == 1 or index % 10 == 0 or index == total:
            progress(f"  history {index}/{total} rounds cached for {market.name_ko}")

    return matches, vote_lookup


def choose_target_round(
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


def resolve_model_choice(requested: str) -> str:
    if requested in ("ml", "ensemble") and not sklearn_available():
        progress(
            f"  [warn] scikit-learn not installed; falling back from --model {requested} to elo. "
            "Install with: python -m pip install scikit-learn numpy"
        )
        return "elo"
    return requested


def run_market(
    client: BetmanClient,
    market: MarketDefinition,
    override_round: int | None,
    history_override: int | None,
    refresh: bool,
    model_choice: str,
    ml_algo: str,
    ensemble_weight: float,
    double_chance_count: int,
    stake_total: float,
    enricher: BaseballEnricher | None,
    enrich_features: bool,
    enrich_workers: int,
    top_n: int,
    match_seqs: set[int],
    lineup_adjuster: LineupAdjuster | None = None,
    preseed_by_name: dict[str, dict[str, float]] | None = None,
    odds_lookup: OddsLookup | None = None,
    odds_weight: float = 0.0,
    calibrate: bool = True,
) -> dict[str, Any]:
    target_round = choose_target_round(client, market, override_round, refresh)
    history_rounds = history_override or market.default_history_rounds
    effective_model = resolve_model_choice(model_choice)

    progress(
        f"\n[{market.name_ko}] target gmTs={target_round.gm_ts} "
        f"(source={target_round.source}, history_rounds={history_rounds}, model={effective_model})"
    )

    history_matches, vote_lookup = load_history_matches(
        client=client,
        market=market,
        target_round=target_round,
        history_rounds=history_rounds,
        refresh=refresh,
    )

    target_detail = client.get_round_detail(
        gm_id=target_round.gm_id,
        gm_ts=target_round.gm_ts,
        game_year="" if target_round.round_year is None else str(target_round.round_year),
        refresh=refresh,
    )
    target_matches = extract_match_records(target_detail, market, target_round)
    target_votes = extract_vote_distribution(target_detail, market)
    for match_seq, shares in target_votes.items():
        vote_lookup[(target_round.gm_ts, match_seq)] = shares

    training_enricher: BaseballEnricher | None = None
    if enrich_features and enricher is not None and effective_model in ("ml", "ensemble"):
        progress(
            f"  prefetching MLB enrichment for {len(history_matches)} training matches "
            f"+ {len(target_matches)} target matches (workers={enrich_workers})"
        )
        try:
            enricher.prefetch_for(
                list(history_matches) + list(target_matches),
                max_workers=enrich_workers,
                progress_callback=lambda msg: progress(f"    [enrich] {msg}"),
            )
            training_enricher = enricher
        except Exception as exc:
            progress(f"  [warn] enrichment prefetch failed: {exc}; continuing without ML enrichment")

    predictions = build_predictions(
        market=market,
        history_matches=history_matches,
        target_matches=target_matches,
        vote_lookup=vote_lookup,
        model_choice=effective_model,
        ml_algo=ml_algo,
        ensemble_weight=ensemble_weight,
        enricher=training_enricher,
        lineup_adjuster=lineup_adjuster,
        preseed_by_name=preseed_by_name,
        odds_lookup=odds_lookup,
        odds_weight=odds_weight,
        calibrate=calibrate,
    )
    if match_seqs:
        scoped_predictions = [p for p in predictions if p.match.match_seq in match_seqs]
        if not scoped_predictions:
            progress(
                f"  [warn] --match-seqs filter removed every match. Available seqs: "
                f"{sorted(p.match.match_seq for p in predictions)}"
            )
    else:
        scoped_predictions = predictions

    bet_plan = select_double_chances(market, scoped_predictions, double_chance_count)
    if top_n > 0:
        bet_plan = sorted(bet_plan, key=lambda item: item.bet_probability, reverse=True)[:top_n]
        bet_plan = sorted(bet_plan, key=lambda item: item.prediction.match.match_seq)
    summary = expected_hit_summary(bet_plan)
    stake_info = stake_breakdown(bet_plan, stake_total)
    enrichment_by_seq = enrich_target_matches(enricher, target_matches)

    print_market_report(market, target_round, bet_plan, summary, stake_info, enrichment_by_seq)

    return {
        "market": market.key,
        "market_name": market.name_ko,
        "gm_id": market.gm_id,
        "gm_ts": target_round.gm_ts,
        "round_no": target_round.round_no,
        "source": target_round.source,
        "history_rounds": history_rounds,
        "model": effective_model,
        "ml_algo": ml_algo if effective_model in ("ml", "ensemble") else None,
        "double_chance_count": double_chance_count,
        "expected_hit_summary": summary,
        "stake": stake_info,
        "predictions": [
            _prediction_payload(market, item, enrichment_by_seq.get(item.prediction.match.match_seq))
            for item in bet_plan
        ],
    }


def _prediction_payload(
    market: MarketDefinition,
    item: BetPlanItem,
    enrichment: MatchEnrichment | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "bet_probability": round(item.bet_probability * 100, 2),
        "probabilities": {
            market.label_map[code]: round(prob * 100, 2)
            for code, prob in item.prediction.probabilities.items()
        },
        "rating_gap": round(item.prediction.rating_gap, 2),
        "sample_scope": item.prediction.sample_scope,
    }
    if enrichment is not None:
        payload["enrichment"] = {
            "home_starter_era": enrichment.home_starter_era,
            "away_starter_era": enrichment.away_starter_era,
            "home_starter_whip": enrichment.home_starter_whip,
            "away_starter_whip": enrichment.away_starter_whip,
            "home_starter_k9": enrichment.home_starter_k9,
            "away_starter_k9": enrichment.away_starter_k9,
            "home_starter_recent_era": enrichment.home_starter_recent_era,
            "away_starter_recent_era": enrichment.away_starter_recent_era,
            "home_starter_recent_whip": enrichment.home_starter_recent_whip,
            "away_starter_recent_whip": enrichment.away_starter_recent_whip,
            "home_recent_pythag": enrichment.home_recent_pythag,
            "away_recent_pythag": enrichment.away_recent_pythag,
            "home_recent_runs_per_game": enrichment.home_recent_runs_per_game,
            "away_recent_runs_per_game": enrichment.away_recent_runs_per_game,
            "home_recent_runs_allowed_per_game": enrichment.home_recent_runs_allowed_per_game,
            "away_recent_runs_allowed_per_game": enrichment.away_recent_runs_allowed_per_game,
            "home_recent_team_ops": enrichment.home_recent_team_ops,
            "away_recent_team_ops": enrichment.away_recent_team_ops,
            "home_recent_team_obp": enrichment.home_recent_team_obp,
            "away_recent_team_obp": enrichment.away_recent_team_obp,
            "home_recent_team_slg": enrichment.home_recent_team_slg,
            "away_recent_team_slg": enrichment.away_recent_team_slg,
            "home_recent_games": enrichment.home_recent_games,
            "away_recent_games": enrichment.away_recent_games,
            "park_factor": enrichment.park_factor,
            "sources": list(enrichment.sources),
        }
    return payload


def build_predictions(
    market: MarketDefinition,
    history_matches: list,
    target_matches: list,
    vote_lookup: VoteLookup,
    model_choice: str,
    ml_algo: str,
    ensemble_weight: float,
    enricher: BaseballEnricher | None = None,
    lineup_adjuster: LineupAdjuster | None = None,
    preseed_by_name: dict[str, dict[str, float]] | None = None,
    odds_lookup: OddsLookup | None = None,
    odds_weight: float = 0.0,
    calibrate: bool = True,
) -> list:
    ml_config = MLConfig(algorithm=ml_algo, calibrate=calibrate)
    odds_lookup = odds_lookup or {}

    if model_choice == "elo":
        elo_predictor = HistoricalPredictor(
            market,
            lineup_adjuster=lineup_adjuster,
            preseed_by_name=preseed_by_name,
        )
        elo_predictor.fit(history_matches)
        return elo_predictor.predict_round(target_matches)

    if model_choice == "ml":
        ml_predictor = MLPredictor(
            market,
            ml_config,
            enricher=enricher,
            lineup_adjuster=lineup_adjuster,
            preseed_by_name=preseed_by_name,
        )
        ml_predictor.fit(history_matches, vote_lookup=vote_lookup)
        return ml_predictor.predict_round(
            target_matches,
            vote_lookup=vote_lookup,
            odds_lookup=odds_lookup,
            odds_weight=odds_weight,
        )

    elo_predictor = HistoricalPredictor(
        market,
        lineup_adjuster=lineup_adjuster,
        preseed_by_name=preseed_by_name,
    )
    elo_predictor.fit(history_matches)
    elo_preds = elo_predictor.predict_round(target_matches)

    ml_predictor = MLPredictor(
        market,
        ml_config,
        enricher=enricher,
        lineup_adjuster=lineup_adjuster,
        preseed_by_name=preseed_by_name,
    )
    ml_predictor.fit(history_matches, vote_lookup=vote_lookup)
    ml_preds = ml_predictor.predict_round(
        target_matches,
        vote_lookup=vote_lookup,
        odds_lookup=odds_lookup,
        odds_weight=odds_weight,
    )
    weight = max(0.0, min(1.0, ensemble_weight))
    return ensemble_predictions(market, elo_preds, ml_preds, weight_primary=weight)


def enrich_target_matches(
    enricher: BaseballEnricher | None,
    target_matches: list,
) -> dict[int, MatchEnrichment]:
    if enricher is None:
        return {}
    out: dict[int, MatchEnrichment] = {}
    for match in target_matches:
        if match.league_name != "MLB":
            continue
        try:
            data = enricher.lookup(match)
        except Exception:
            continue
        if data is NULL_ENRICHMENT:
            continue
        out[match.match_seq] = data
    return out


def print_market_report(
    market: MarketDefinition,
    target_round: RoundReference,
    bet_plan: list[BetPlanItem],
    summary: dict[str, float],
    stake_info: dict[str, float] | None,
    enrichment_by_seq: dict[int, MatchEnrichment] | None = None,
) -> None:
    round_label = f"{target_round.round_no}회차" if target_round.round_no is not None else str(target_round.gm_ts)
    print(f"\n=== {market.name_ko} | {round_label} | gmTs {target_round.gm_ts} ===")
    print("No  Date               League               Match                          Bet    Hit%   Probabilities")
    print("-" * 116)
    enrichment_by_seq = enrichment_by_seq or {}
    for item in bet_plan:
        prediction = item.prediction
        probs = " / ".join(
            f"{market.label_map[code]} {prediction.probabilities[code] * 100:5.1f}%"
            for code in market.ordered_codes
        )
        match_name = f"{prediction.match.home_name} vs {prediction.match.away_name}"
        bet_display = item.bet_label if item.bet_type == "double" else f"{item.bet_label}"
        print(
            f"{prediction.match.match_seq:>2}  "
            f"{prediction.match.game_date_str[:17]:17}  "
            f"{prediction.match.league_name[:18]:18}  "
            f"{match_name[:28]:28}  "
            f"{bet_display:^5}  "
            f"{item.bet_probability * 100:5.1f}%  {probs}"
        )
        enrichment = enrichment_by_seq.get(prediction.match.match_seq)
        if enrichment is not None:
            print("    " + _format_enrichment(enrichment))

    doubles = sum(1 for item in bet_plan if item.bet_type == "double")
    singles = len(bet_plan) - doubles
    parlay_prob = summary["all_correct_probability"]
    parlay_pct = parlay_prob * 100
    fair_multiplier = (1.0 / parlay_prob) if parlay_prob > 0 else float("inf")
    print(
        f"\nPlan: {singles} straight pick(s), {doubles} double-chance ticket(s) | "
        f"expected hits {summary['expected_hits']} / {summary['matches']}"
    )
    print(
        f"Parlay probability (all {summary['matches']} legs hit): "
        f"{parlay_pct:.4f}%  →  fair payout multiplier ≈ {fair_multiplier:,.1f}x"
    )
    if stake_info:
        print(
            f"Stake: total {stake_info['stake_total']:.0f} across "
            f"{stake_info['tickets']} tickets → {stake_info['stake_per_ticket']:.0f} per ticket"
        )


def _format_enrichment(data: MatchEnrichment) -> str:
    parts: list[str] = []
    h_era = _fmt_optional(data.home_starter_era, "{:.2f}")
    a_era = _fmt_optional(data.away_starter_era, "{:.2f}")
    h_re = _fmt_optional(data.home_starter_recent_era, "{:.2f}")
    a_re = _fmt_optional(data.away_starter_recent_era, "{:.2f}")
    if h_era != "?" or a_era != "?":
        parts.append(f"starter ERA H {h_era} (l30 {h_re}) / A {a_era} (l30 {a_re})")
    h_py = _fmt_optional(data.home_recent_pythag, "{:.3f}")
    a_py = _fmt_optional(data.away_recent_pythag, "{:.3f}")
    if h_py != "?" or a_py != "?":
        parts.append(f"pythag H {h_py} / A {a_py}")
    if data.home_recent_team_ops is not None and data.away_recent_team_ops is not None:
        parts.append(f"OPS H {data.home_recent_team_ops:.3f} / A {data.away_recent_team_ops:.3f}")
    if data.home_recent_runs_per_game is not None and data.home_recent_runs_allowed_per_game is not None:
        parts.append(
            f"H last {data.home_recent_games}g {data.home_recent_runs_per_game:.2f}rs/{data.home_recent_runs_allowed_per_game:.2f}ra"
        )
    if data.away_recent_runs_per_game is not None and data.away_recent_runs_allowed_per_game is not None:
        parts.append(
            f"A last {data.away_recent_games}g {data.away_recent_runs_per_game:.2f}rs/{data.away_recent_runs_allowed_per_game:.2f}ra"
        )
    if data.park_factor is not None and data.park_factor != 100:
        parts.append(f"park {data.park_factor}")
    return " | ".join(parts) if parts else "(no enrichment)"


def _fmt_optional(value: float | None, fmt: str) -> str:
    return fmt.format(value) if value is not None else "?"


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON report to {path}")


def main() -> int:
    args = parse_args()
    market_keys = resolve_market_keys(args.market)
    if args.round is not None and len(market_keys) != 1:
        raise SystemExit("--round can only be used when running a single market.")

    paths = build_paths(args)
    client = BetmanClient(
        cache_dir=paths.cache_dir,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    enricher: BaseballEnricher | None = None
    enable_enrichment = args.enrich_mlb or args.enrich_features
    if enable_enrichment:
        enricher = BaseballEnricher(cache_root=paths.root / "cache" / "baseball", verbose=False)
        mode = "ML training features" if args.enrich_features else "display only"
        progress(f"MLB enrichment enabled ({mode})")

    lineup_adjuster: LineupAdjuster | None = None
    if args.lineup_players or args.lineup_missing:
        lineup_adjuster = LineupAdjuster(
            players_path=Path(args.lineup_players) if args.lineup_players else None,
            lineups_path=Path(args.lineup_missing) if args.lineup_missing else None,
        )
        if lineup_adjuster.has_data:
            progress(
                f"Lineup adjuster loaded: {sum(len(v) for v in lineup_adjuster.lineups.values())} round(s) of team-news"
            )
        else:
            progress("  [warn] --lineup-missing file empty/absent; lineup adjuster idle")

    soccer_preseed: dict[str, dict[str, float]] = {}
    if args.external_history_dir:
        ext_dir = Path(args.external_history_dir)
        soccer_preseed = build_preseed(ext_dir)
        if soccer_preseed:
            team_count = sum(len(v) for v in soccer_preseed.values())
            progress(
                f"External history pre-seed: {team_count} teams across {len(soccer_preseed)} league(s) "
                f"from {ext_dir}"
            )
        else:
            progress(f"  [warn] --external-history-dir '{ext_dir}' produced no usable rows")

    odds_lookup = load_odds_lookup(args.odds_file)
    if odds_lookup and args.odds_weight <= 0.0:
        progress("  [warn] --odds-file loaded but --odds-weight is 0; closing-odds blend disabled")
    elif odds_lookup:
        progress(
            f"Closing-odds blend enabled: weight={args.odds_weight:.2f} on {len(odds_lookup)} match(es)"
        )

    reports = []
    for key in market_keys:
        market = MARKETS[key]
        market_enricher = enricher if (enricher and market.key == "baseball") else None
        market_preseed = soccer_preseed if market.key == "soccer" else None
        reports.append(
            run_market(
                client=client,
                market=market,
                override_round=args.round,
                history_override=args.history_rounds,
                refresh=args.refresh,
                model_choice=args.model,
                ml_algo=args.ml_algo,
                ensemble_weight=args.ensemble_weight,
                double_chance_count=args.double_chance_count,
                stake_total=args.stake_total,
                enricher=market_enricher,
                enrich_features=args.enrich_features,
                enrich_workers=args.enrich_workers,
                top_n=args.top_n,
                match_seqs=parse_match_seqs(args.match_seqs),
                lineup_adjuster=lineup_adjuster,
                preseed_by_name=market_preseed,
                odds_lookup=odds_lookup,
                odds_weight=args.odds_weight,
                calibrate=not args.no_calibrate,
            )
        )

    final_payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "reports": reports}
    if args.json_out:
        write_json_report(Path(args.json_out), final_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
