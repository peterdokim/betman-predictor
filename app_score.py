"""CLI for G016 축구토토 스페셜 트리플 (score-special triple) predictions.

Independent of app.py. Trains a Dixon-Coles bivariate Poisson on the cached G016
history, predicts the 6x6 score grid for each of the 3 matches in a target round,
and prints the top-K triple tickets ranked by joint hit probability.

Example:
  python app_score.py --round 260017 --budget 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from betman_predictor.score_special import (
    DEFAULT_RHO,
    SCORE_BINS,
    build_top_triples,
    extract_score_matches,
    fit_team_rates,
    load_history,
    predict_round_cells,
    public_vote_distribution,
)


CACHE_DIR = Path(__file__).resolve().parent / "cache" / "api" / "round-details"


def _load_round(gm_ts: int) -> dict:
    path = CACHE_DIR / f"G016-{gm_ts}.json"
    if not path.exists():
        sys.exit(f"No cached round detail at {path}. Fetch it first with the live API.")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_grid(grid: list[list[float]], scale: float = 100.0) -> str:
    header = "       " + "  ".join(f"A={a if a < SCORE_BINS - 1 else '5+':>2}" for a in range(SCORE_BINS))
    lines = [header]
    for h in range(SCORE_BINS):
        row = "  ".join(f"{grid[h][a] * scale:5.2f}" for a in range(SCORE_BINS))
        label = f"H={h if h < SCORE_BINS - 1 else '5+':>2}"
        lines.append(f"  {label:>4}  {row}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="G016 score-special triple predictor.")
    parser.add_argument("--round", type=int, required=True, help="target gmTs (e.g. 260017)")
    parser.add_argument("--history-rounds", type=int, default=80, help="number of past rounds to train on")
    parser.add_argument("--budget", type=int, default=5000, help="total stake in KRW (each ticket = 100원)")
    parser.add_argument("--ticket-cost", type=int, default=100, help="cost per triple ticket in KRW")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="Dixon-Coles low-score correction (-0.2..0.0)")
    parser.add_argument(
        "--public-blend",
        type=float,
        default=0.20,
        help="weight on the public vote distribution (0=pure model, 1=pure public)",
    )
    parser.add_argument("--show-grids", action="store_true", help="print the per-match 6x6 probability grids")
    parser.add_argument("--top-cells", type=int, default=8, help="how many top cells to print per match")
    parser.add_argument(
        "--decompose",
        type=str,
        default="36,50,72",
        help="comma-separated combo-count targets to decompose into per-match cell picks (a*b*c)",
    )
    parser.add_argument(
        "--decompose-top",
        type=int,
        default=3,
        help="how many best decompositions to show per target combo count",
    )
    parser.add_argument(
        "--explicit-picks",
        type=str,
        default=None,
        help=(
            "evaluate a specific manual ticket. Format: 'M1_home={a,b}|M1_away={c,d};"
            "M2_home=...;M3_home=...'. Example for 본머스{1,2}∩크리스털{1,2} / 맨유{2,3}∩리버풀{1,2} / "
            "빌라{1,2,3}∩토트넘{1}: '1,2|1,2;2,3|1,2;1,2,3|1'"
        ),
    )
    parser.add_argument("--json-out", type=str, default=None, help="write the triple list to JSON")
    args = parser.parse_args()

    n_tickets = max(1, args.budget // args.ticket_cost)

    target_detail = _load_round(args.round)
    target_matches = extract_score_matches(target_detail, gm_ts_override=args.round)
    if len(target_matches) != 3:
        sys.exit(f"Expected 3 matches in round {args.round}, got {len(target_matches)}")

    history = load_history(CACHE_DIR, limit=args.history_rounds, exclude_gm_ts=args.round)
    print(f"[fit] {len(history)} historical matches across "
          f"{len({m.gm_ts for m in history})} rounds")
    rates = fit_team_rates(history)
    print(f"[fit] {len(rates.attack)} unique teams; "
          f"{len(rates.league_mean_home)} leagues with home/away means")

    public_grid = public_vote_distribution(target_detail)
    cells_by_match = predict_round_cells(
        target_matches,
        rates,
        public_vote_grid=public_grid,
        rho=args.rho,
        public_blend=args.public_blend,
    )

    print(f"\n=== G016 round {args.round} | budget {args.budget}원 = {n_tickets} tickets ===")
    for match in target_matches:
        cells = cells_by_match[match.match_seq]
        print(f"\nMatch {match.match_seq}: {match.home_name} vs {match.away_name}  "
              f"({match.league_name}, {match.game_date_str})")
        print(f"  top {args.top_cells} cells (model prob %)")
        for c in cells[:args.top_cells]:
            print(f"    {c.label:>5}  {c.p_model * 100:5.2f}%")
        if args.show_grids:
            print("  full grid (probability %):")
            grid = [[0.0] * SCORE_BINS for _ in range(SCORE_BINS)]
            for c in cells:
                grid[c.home_bin][c.away_bin] = c.p_model
            print(_format_grid(grid))

    triples = build_top_triples(cells_by_match, n_tickets)

    print(f"\n=== top {len(triples)} triple tickets by joint hit probability ===")
    print(f"{'#':>3}  {'ticket':<22}  {'P(hit)':>9}  {'fair odds':>11}  {'cost':>6}")
    print("-" * 60)
    cumulative = 0.0
    for i, ticket in enumerate(triples, start=1):
        cumulative += ticket.p_hit
        cost = i * args.ticket_cost
        print(f"{i:>3}  {ticket.label:<22}  {ticket.p_hit * 100:8.4f}%  "
              f"{ticket.fair_multiplier:>10,.0f}x  {cost:>5}원")

    print(f"\nTotal P(at least one ticket hits): "
          f"{(1 - _none_hit_prob(triples)) * 100:.4f}%")
    print(f"Sum of independent-cell probabilities (upper bound): {cumulative * 100:.4f}%")
    print(f"Expected hits (sum of P): {cumulative:.4f}")

    # Explicit ticket evaluation (user-specified home/away score sets per match)
    if args.explicit_picks:
        picks = _parse_explicit_picks(args.explicit_picks)
        _print_explicit_evaluation(
            target_matches=target_matches,
            cells_by_match=cells_by_match,
            picks=picks,
            ticket_cost=args.ticket_cost,
        )

    # Per-match decompositions: pick a cells in M1, b in M2, c in M3 (a*b*c total combos)
    if args.decompose:
        targets = [int(t.strip()) for t in args.decompose.split(",") if t.strip()]
        for target_combos in targets:
            _print_decompositions(
                target_matches=target_matches,
                cells_by_match=cells_by_match,
                target_combos=target_combos,
                top_n=args.decompose_top,
                ticket_cost=args.ticket_cost,
            )

    if args.json_out:
        out = {
            "gm_ts": args.round,
            "budget_krw": args.budget,
            "ticket_cost_krw": args.ticket_cost,
            "n_tickets": len(triples),
            "matches": [
                {
                    "match_seq": m.match_seq,
                    "home_name": m.home_name,
                    "away_name": m.away_name,
                    "league_name": m.league_name,
                    "game_date_str": m.game_date_str,
                    "top_cells": [
                        {"label": c.label, "p_model": c.p_model}
                        for c in cells_by_match[m.match_seq][:args.top_cells]
                    ],
                }
                for m in target_matches
            ],
            "tickets": [
                {
                    "rank": i,
                    "label": t.label,
                    "p_hit": t.p_hit,
                    "fair_multiplier": t.fair_multiplier,
                    "cells": [
                        {"match_seq": c.match_seq, "home_bin": c.home_bin,
                         "away_bin": c.away_bin, "label": c.label, "p_model": c.p_model}
                        for c in t.cells
                    ],
                }
                for i, t in enumerate(triples, start=1)
            ],
        }
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.json_out}")
    return 0


def _none_hit_prob(triples: list) -> float:
    p = 1.0
    for t in triples:
        p *= (1.0 - t.p_hit)
    return p


def _cumulative_top_p(cells: list, n: int) -> float:
    """Sum of model probabilities for the top-n cells of one match (P that any one of them is correct)."""
    return sum(c.p_model for c in cells[:n])


def _parse_explicit_picks(spec: str) -> list[tuple[set[int], set[int]]]:
    """Parse '1,2|1,2;2,3|1,2;1,2,3|1' into [(home_bins, away_bins)] x 3 matches."""
    matches = [seg.strip() for seg in spec.split(";") if seg.strip()]
    if len(matches) != 3:
        raise SystemExit(f"--explicit-picks needs 3 ;-separated match specs, got {len(matches)}")
    out: list[tuple[set[int], set[int]]] = []
    for i, seg in enumerate(matches, start=1):
        if "|" not in seg:
            raise SystemExit(f"match {i}: expected 'home|away', got '{seg}'")
        home_str, away_str = seg.split("|", 1)
        try:
            home = {int(x.strip()) for x in home_str.split(",") if x.strip()}
            away = {int(x.strip()) for x in away_str.split(",") if x.strip()}
        except ValueError as exc:
            raise SystemExit(f"match {i}: non-integer score in '{seg}'") from exc
        if not home or not away:
            raise SystemExit(f"match {i}: both home and away need at least one score")
        for v in home | away:
            if v < 0 or v > SCORE_BINS - 1:
                raise SystemExit(f"match {i}: score {v} out of range 0..{SCORE_BINS - 1} (5+ = bin 5)")
        out.append((home, away))
    return out


def _print_explicit_evaluation(
    target_matches: list,
    cells_by_match: dict[int, list],
    picks: list[tuple[set[int], set[int]]],
    ticket_cost: int,
) -> None:
    seqs = sorted(cells_by_match.keys())
    print("\n=== explicit ticket evaluation ===")
    per_match_p: list[float] = []
    total_combos = 1
    for idx, (seq, (home_bins, away_bins)) in enumerate(zip(seqs, picks)):
        cells = cells_by_match[seq]
        # All cells in (home_bins x away_bins)
        chosen = [c for c in cells if c.home_bin in home_bins and c.away_bin in away_bins]
        n_cells = len(home_bins) * len(away_bins)
        coverage = sum(c.p_model for c in chosen)
        per_match_p.append(coverage)
        total_combos *= n_cells
        m = target_matches[idx]
        home_str = ",".join(str(b) if b < SCORE_BINS - 1 else "5+" for b in sorted(home_bins))
        away_str = ",".join(str(b) if b < SCORE_BINS - 1 else "5+" for b in sorted(away_bins))
        cell_strs = ", ".join(f"{c.label}({c.p_model*100:.2f}%)"
                              for c in sorted(chosen, key=lambda c: c.p_model, reverse=True))
        print(f"\n  M{seq} {m.home_name} vs {m.away_name}")
        print(f"    {m.home_name} 득점 ∈ {{{home_str}}} × {m.away_name} 득점 ∈ {{{away_str}}}  "
              f"= {n_cells} 셀")
        print(f"    coverage {coverage*100:.2f}% : {cell_strs}")

    p_hit = per_match_p[0] * per_match_p[1] * per_match_p[2]
    fair = 1.0 / p_hit if p_hit > 0 else float("inf")
    cost = total_combos * ticket_cost
    print(f"\n  TOTAL  {total_combos} 조합 = {cost}원  |  "
          f"P(hit) {p_hit*100:.4f}%  |  fair odds {fair:,.0f}x")

    # Compare against the best-known decomposition for the same combo count
    decomps = _enumerate_decompositions(total_combos)
    best_p = 0.0
    best_abc = None
    cells1 = cells_by_match[seqs[0]]
    cells2 = cells_by_match[seqs[1]]
    cells3 = cells_by_match[seqs[2]]
    for a, b, c in decomps:
        p = (_cumulative_top_p(cells1, a)
             * _cumulative_top_p(cells2, b)
             * _cumulative_top_p(cells3, c))
        if p > best_p:
            best_p = p
            best_abc = (a, b, c)
    if best_abc:
        delta = (best_p - p_hit) * 100
        print(f"  vs best {best_abc[0]}×{best_abc[1]}×{best_abc[2]} top-cell decomposition: "
              f"{best_p*100:.4f}% (Δ {delta:+.4f}pp)")


def _enumerate_decompositions(target: int, max_per_match: int = 36) -> list[tuple[int, int, int]]:
    """All (a, b, c) with a*b*c == target and 1 <= a,b,c <= max_per_match."""
    out: list[tuple[int, int, int]] = []
    for a in range(1, max_per_match + 1):
        if target % a:
            continue
        rem_a = target // a
        for b in range(1, max_per_match + 1):
            if rem_a % b:
                continue
            c = rem_a // b
            if 1 <= c <= max_per_match:
                out.append((a, b, c))
    return out


def _print_decompositions(
    target_matches: list,
    cells_by_match: dict[int, list],
    target_combos: int,
    top_n: int,
    ticket_cost: int,
) -> None:
    decomps = _enumerate_decompositions(target_combos)
    if not decomps:
        print(f"\n[no decompositions found for {target_combos} combos]")
        return

    # P_round_hit(a,b,c) = (sum top-a in M1) * (sum top-b in M2) * (sum top-c in M3)
    seqs = sorted(cells_by_match.keys())
    cells1 = cells_by_match[seqs[0]]
    cells2 = cells_by_match[seqs[1]]
    cells3 = cells_by_match[seqs[2]]

    scored = []
    for a, b, c in decomps:
        p_hit = (_cumulative_top_p(cells1, a)
                 * _cumulative_top_p(cells2, b)
                 * _cumulative_top_p(cells3, c))
        scored.append(((a, b, c), p_hit))
    scored.sort(key=lambda t: t[1], reverse=True)

    cost = target_combos * ticket_cost
    print(f"\n=== {target_combos} 조합 분해 ({cost}원) — top {top_n} ===")
    for rank, ((a, b, c), p_hit) in enumerate(scored[:top_n], start=1):
        fair = 1.0 / p_hit if p_hit > 0 else float("inf")
        print(f"\n  [{rank}] {a} × {b} × {c} = {a*b*c} 조합  |  "
              f"P(hit) {p_hit*100:.4f}%  |  fair odds {fair:,.0f}x")
        for label, n_picks, cells in (
            (f"M{seqs[0]} {target_matches[0].home_name} vs {target_matches[0].away_name}", a, cells1),
            (f"M{seqs[1]} {target_matches[1].home_name} vs {target_matches[1].away_name}", b, cells2),
            (f"M{seqs[2]} {target_matches[2].home_name} vs {target_matches[2].away_name}", c, cells3),
        ):
            picks = cells[:n_picks]
            covered = sum(p.p_model for p in picks) * 100
            cell_strs = ", ".join(f"{p.label}({p.p_model*100:.2f}%)" for p in picks)
            print(f"      {label}")
            print(f"        {n_picks}개 셀 [{covered:.2f}% coverage]: {cell_strs}")


if __name__ == "__main__":
    raise SystemExit(main())
