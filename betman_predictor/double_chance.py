"""Post-process Predictions into a double-chance betting plan.

A double chance covers two of the three outcomes for a single match. Given a
list of 14 predictions, this module ranks the matches and applies double-chance
coverage to the top N — i.e. the matches where the model is least confident
about ruling any single outcome out.

Ranking: 1 - min(P) weighted by data depth (vote_total proxied via match.gm_ts
order is unreliable, so we use a simple confidence proxy from the prediction
probabilities themselves: low entropy = high confidence in one outcome).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional

from betman_predictor.models import MarketDefinition, Prediction


PAIR_LABELS = {
    ("A", "B"): "1X",
    ("A", "D"): "12",
    ("B", "D"): "X2",
}


@dataclass(frozen=True)
class BetPlanItem:
    prediction: Prediction
    bet_type: str
    bet_label: str
    covered_codes: tuple[str, ...]
    bet_probability: float
    rank_score: float


def _pair_probability(prediction: Prediction, codes: tuple[str, str]) -> float:
    return float(prediction.probabilities.get(codes[0], 0.0) + prediction.probabilities.get(codes[1], 0.0))


def _entropy(probs: dict[str, float]) -> float:
    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= p * math.log(p)
    return h


def _confidence_weight(entropies: list[float], target: float) -> float:
    """Map an entropy value into [0.5, 1.0] using the spread across all matches.

    More entropy in this match relative to the round → larger weight, because
    we trust the model is genuinely uncertain (rather than thinly informed)."""

    if len(entropies) < 2:
        return 1.0
    median = statistics.median(entropies)
    spread = statistics.pstdev(entropies) or 1.0
    return 0.5 + 0.5 * math.tanh((target - median) / spread)


def select_double_chances(
    market: MarketDefinition,
    predictions: list[Prediction],
    count: int,
) -> list[BetPlanItem]:
    """Choose `count` matches for double-chance coverage; rest stay straight picks."""

    if count < 0:
        count = 0
    count = min(count, len(predictions))

    # Best double-chance candidate per match = drop the lowest-prob outcome.
    candidates: list[tuple[Prediction, tuple[str, str], float]] = []
    entropies: list[float] = []
    for prediction in predictions:
        ordered = sorted(
            market.ordered_codes,
            key=lambda code: prediction.probabilities.get(code, 0.0),
            reverse=True,
        )
        top_two = tuple(sorted(ordered[:2]))
        if top_two not in PAIR_LABELS:
            top_two = next(iter(PAIR_LABELS))
        pair_prob = _pair_probability(prediction, top_two)  # type: ignore[arg-type]
        candidates.append((prediction, top_two, pair_prob))  # type: ignore[arg-type]
        entropies.append(_entropy(prediction.probabilities))

    # Score each candidate; higher = more deserving of a double chance.
    scored: list[tuple[float, int]] = []
    for index, (prediction, _pair, pair_prob) in enumerate(candidates):
        weight = _confidence_weight(entropies, entropies[index])
        # pair_prob already encodes (1 - min_prob); multiply by confidence weight.
        scored.append((pair_prob * weight, index))

    chosen_indexes = {index for _score, index in sorted(scored, reverse=True)[:count]}

    plan: list[BetPlanItem] = []
    for index, prediction in enumerate(predictions):
        rank_score = scored[index][0]
        if index in chosen_indexes:
            _pred, pair, pair_prob = candidates[index]
            plan.append(
                BetPlanItem(
                    prediction=prediction,
                    bet_type="double",
                    bet_label=PAIR_LABELS[pair],
                    covered_codes=pair,
                    bet_probability=pair_prob,
                    rank_score=rank_score,
                )
            )
        else:
            pick_code = prediction.pick_code
            plan.append(
                BetPlanItem(
                    prediction=prediction,
                    bet_type="single",
                    bet_label=market.label_map.get(pick_code, pick_code),
                    covered_codes=(pick_code,),
                    bet_probability=float(prediction.probabilities.get(pick_code, 0.0)),
                    rank_score=rank_score,
                )
            )
    return plan


def expected_hit_summary(plan: Iterable[BetPlanItem]) -> dict[str, float]:
    plan_list = list(plan)
    if not plan_list:
        return {"matches": 0, "expected_hits": 0.0, "all_correct_probability": 0.0}

    expected_hits = sum(item.bet_probability for item in plan_list)
    all_correct = 1.0
    for item in plan_list:
        all_correct *= max(item.bet_probability, 1e-9)
    return {
        "matches": len(plan_list),
        "expected_hits": round(expected_hits, 3),
        "all_correct_probability": all_correct,
    }


def stake_breakdown(plan: list[BetPlanItem], stake_total: float) -> Optional[dict[str, float]]:
    if stake_total <= 0 or not plan:
        return None
    per_ticket = stake_total / len(plan)
    return {
        "stake_total": float(stake_total),
        "tickets": len(plan),
        "stake_per_ticket": round(per_ticket, 2),
    }
