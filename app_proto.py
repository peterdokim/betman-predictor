"""CLI for Betman 프로토 승부식 (G101) parlay analysis.

Modes:
  --list                 list all soccer full-time bet lines with model P / market P / EV
  --recommend N          show top-N value bets across the entire round (sorted by EV)
  --picks "215944,215945:O,215946"   evaluate a 3-4 leg parlay; auto-detects sides
                                      (or use explicit codes: H/D/A/O/U/ODD/EVEN)

Examples:
  python app_proto.py --round 260051 --recommend 20
  python app_proto.py --round 260051 --picks "215944:H,215779:오버"
  python app_proto.py --round 260051 --list --league EPL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from betman_predictor.proto import (
    MARKET_1X2,
    MARKET_HANDICAP,
    MARKET_SUM_ODDEVEN,
    MARKET_TOTAL,
    Pick,
    evaluate_lines,
    evaluate_parlay,
    parse_proto_round,
    recommend_top_ev,
    resolve_picks,
)
from betman_predictor.score_special import fit_team_rates, load_history


CACHE_DIR = Path(__file__).resolve().parent / "cache" / "api" / "round-details"
HISTORY_DIR = CACHE_DIR  # G016 history sits beside the proto cache


def _load_round(gm_id: str, gm_ts: int) -> dict:
    path = CACHE_DIR / f"{gm_id}-{gm_ts}.json"
    if not path.exists():
        sys.exit(f"No cached round detail at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _market_label(market: str) -> str:
    return {
        MARKET_1X2: "승무패",
        MARKET_HANDICAP: "핸디캡",
        MARKET_TOTAL: "언오버",
        MARKET_SUM_ODDEVEN: "홀짝",
    }.get(market, market)


def _fmt_match(line) -> str:
    return f"{line.home_name} vs {line.away_name}"


def _line_descriptor(line) -> str:
    if line.market == MARKET_1X2:
        return "승무패"
    if line.market == MARKET_HANDICAP:
        sign = "+" if line.win_handi >= 0 else ""
        return f"핸디 H{sign}{line.win_handi:g}"
    if line.market == MARKET_TOTAL:
        return f"O/U {line.win_handi:g}"
    if line.market == MARKET_SUM_ODDEVEN:
        return "SUM 홀짝"
    return line.bet_name


def _side_label(line, side: str) -> str:
    if line.market in (MARKET_1X2, MARKET_HANDICAP):
        return {"H": line.win_text or "승",
                "D": line.draw_text or "무",
                "A": line.lose_text or "패"}.get(side, side)
    if line.market == MARKET_TOTAL:
        return {"U": "언더", "O": "오버"}.get(side, side)
    if line.market == MARKET_SUM_ODDEVEN:
        return {"ODD": "홀", "EVEN": "짝"}.get(side, side)
    return side


def _print_list(evaluations, league_filter: str | None) -> None:
    print(f"{'notice':>7}  {'league':<14}  {'match':<32}  {'market':<14}  "
          f"{'side':<6}  {'allot':>6}  {'P_model':>8}  {'EV':>7}")
    print("-" * 110)
    for ev in evaluations:
        line = ev.line
        if league_filter and league_filter not in line.league_name:
            continue
        for side, allot in ev.side_allots.items():
            if allot <= 0:
                continue
            p = ev.side_probs.get(side, 0.0)
            ev_val = p * allot - 1.0
            print(
                f"{line.notice_no:>7}  {line.league_name[:14]:<14}  "
                f"{_fmt_match(line)[:32]:<32}  {_line_descriptor(line)[:14]:<14}  "
                f"{_side_label(line, side)[:6]:<6}  {allot:>6.2f}  "
                f"{p * 100:>7.2f}%  {ev_val * 100:>+6.2f}%"
            )


def _filter_evaluations(
    evaluations,
    league_substrings: list[str] | None,
    times_kst: list[str] | None,
    date_kst: str | None,
):
    from datetime import timezone, timedelta
    KST = timezone(timedelta(hours=9))
    out = []
    for ev in evaluations:
        line = ev.line
        if league_substrings:
            if not any(s in line.league_name for s in league_substrings):
                continue
        if times_kst or date_kst:
            if not line.game_datetime:
                continue
            kst = line.game_datetime.astimezone(KST)
            if times_kst and kst.strftime("%H:%M") not in times_kst:
                continue
            if date_kst and kst.strftime("%y-%m-%d") != date_kst:
                continue
        out.append(ev)
    return out


def _print_per_match_best(filtered_evaluations) -> None:
    """For each unique match (home, away, kickoff), print the single best-EV pick."""
    from datetime import timezone, timedelta
    KST = timezone(timedelta(hours=9))

    # Group all evaluations by match identity (home+away+kickoff)
    groups: dict[tuple, list] = {}
    for ev in filtered_evaluations:
        key = (ev.line.home_name, ev.line.away_name,
               ev.line.game_datetime.astimezone(KST).strftime("%y-%m-%d %H:%M") if ev.line.game_datetime else "")
        groups.setdefault(key, []).append(ev)

    print(f"\n=== Best pick per match ({len(groups)} matches) ===\n")
    rows = []
    for (home, away, kick), evs in groups.items():
        league = evs[0].line.league_name
        # Find best (line, side) by EV across this match's all markets
        best_ev = -1e9
        best_pick = None
        for ev in evs:
            for side, allot in ev.side_allots.items():
                if allot <= 0: continue
                p = ev.side_probs.get(side, 0.0)
                ev_val = p * allot - 1.0
                if ev_val > best_ev:
                    best_ev = ev_val
                    best_pick = (ev, side, p, allot)
        if best_pick:
            rows.append((kick, league, home, away, best_pick, best_ev))

    rows.sort(key=lambda r: (r[0], -r[5]))   # by kickoff time, then EV desc
    for kick, league, home, away, (ev, side, p, allot), ev_val in rows:
        line = ev.line
        market_label = _line_descriptor(line)
        side_label = _side_label(line, side)
        implied = (1.0 / allot)
        edge_pp = (p - implied) * 100
        print(f"  {kick}  {league[:14]:<14}  {home} vs {away}")
        print(f"      → notice={line.notice_no:<7}  {market_label:<14}  {side_label:<6}  "
              f"allot {allot:>5.2f}  P_model {p*100:5.2f}%  edge {edge_pp:+5.2f}pp  "
              f"EV {ev_val*100:+6.2f}%\n")


def _print_recommendations(recs) -> None:
    print(f"\n=== Top {len(recs)} EV-positive picks ===")
    print(f"{'#':>3}  {'notice':>7}  {'league':<12}  {'match':<28}  "
          f"{'market':<14}  {'side':<6}  {'allot':>6}  {'P_model':>8}  "
          f"{'edge':>7}  {'EV':>7}")
    print("-" * 116)
    for i, r in enumerate(recs, start=1):
        line = r.line
        print(
            f"{i:>3}  {line.notice_no:>7}  {line.league_name[:12]:<12}  "
            f"{_fmt_match(line)[:28]:<28}  {_line_descriptor(line)[:14]:<14}  "
            f"{_side_label(line, r.side)[:6]:<6}  {r.allot:>6.2f}  "
            f"{r.p_model * 100:>7.2f}%  {r.edge_pp:>+5.2f}pp  {r.ev * 100:>+6.2f}%"
        )


def _print_parlay(picks: list[Pick], result, stake: int) -> None:
    print(f"\n=== Parlay verification ({result.n_legs} legs) ===")
    print(f"{'#':>3}  {'notice':>7}  {'match':<32}  {'pick':<22}  "
          f"{'allot':>6}  {'P_model':>8}  {'edge':>7}")
    print("-" * 100)
    for i, (pick, p, allot) in enumerate(zip(picks, result.p_models, result.allots), start=1):
        implied = (1.0 / allot) if allot > 0 else 0.0
        edge = (p - implied) * 100
        print(
            f"{i:>3}  {pick.notice_no:>7}  {_fmt_match(pick.line)[:32]:<32}  "
            f"{pick.side_label[:22]:<22}  {allot:>6.2f}  "
            f"{p * 100:>7.2f}%  {edge:>+5.2f}pp"
        )

    print(f"\n  Joint P (model):     {result.joint_p * 100:.4f}%")
    print(f"  Parlay allot:        {result.parlay_allot:,.2f}x")
    print(f"  Fair allot (model):  {result.fair_allot:,.2f}x")
    print(f"  Edge:                allot is "
          f"{'+' if result.parlay_allot > result.fair_allot else '-'}"
          f"{abs(result.parlay_allot - result.fair_allot):,.2f}x "
          f"vs fair value")
    print(f"  EV per 100원:        {result.ev * 100:>+6.2f}원  "
          f"(P_hit × allot − 1 = {result.ev:+.4f})")
    if stake > 0:
        expected_return = stake * (1.0 + result.ev)
        print(f"  Stake {stake}원 → expected return {expected_return:,.0f}원, "
              f"hit payout {stake * result.parlay_allot:,.0f}원")


def main() -> int:
    parser = argparse.ArgumentParser(description="Betman proto parlay analyzer (G101).")
    parser.add_argument("--round", type=int, required=True, help="target gmTs (e.g. 260051)")
    parser.add_argument("--gm-id", default="G101", help="game id (default G101 = 축구 승부식)")
    parser.add_argument("--history-rounds", type=int, default=80,
                        help="G016 score-history rounds for Dixon-Coles training")
    parser.add_argument("--list", action="store_true",
                        help="dump every supported soccer full-time line")
    parser.add_argument("--league", type=str, default=None,
                        help="filter --list output by substring of leagueName")
    parser.add_argument("--leagues", type=str, default=None,
                        help="comma-separated league name substrings (e.g. 'K리그2,J1')")
    parser.add_argument("--time", type=str, default=None,
                        help="comma-separated KST start times (e.g. '16:00,16:30') -- "
                             "filters matches by kickoff time")
    parser.add_argument("--date", type=str, default=None,
                        help="KST date filter as YY-MM-DD (e.g. '26-05-03')")
    parser.add_argument("--per-match", action="store_true",
                        help="for each filtered match, print only its single best-EV line "
                             "(one bet per game)")
    parser.add_argument("--recommend", type=int, default=0,
                        help="print top-N value bets sorted by EV")
    parser.add_argument("--min-ev", type=float, default=0.0,
                        help="minimum EV (0.0 = at-or-above fair)")
    parser.add_argument("--public-blend", type=float, default=0.5,
                        help="0=pure model, 1=pure market consensus; 0.5 (default) is a "
                             "robust calibration when training data is sparse")
    parser.add_argument("--external-history", type=str, default=None,
                        help="path to a directory of football-data.co.uk CSVs to augment "
                             "the G016-cache training set (recommended)")
    parser.add_argument("--picks", type=str, default=None,
                        help="comma-separated picks like '215944:H,215779:오버,215800'")
    parser.add_argument("--stake", type=int, default=0,
                        help="stake (KRW) for expected-return display in --picks mode")
    parser.add_argument("--json-out", type=str, default=None,
                        help="write the parlay or recommendation to a JSON file")
    args = parser.parse_args()

    if not (args.list or args.recommend or args.picks or args.per_match):
        sys.exit("Pick at least one of --list / --recommend N / --picks / --per-match")

    detail = _load_round(args.gm_id, args.round)
    lines = parse_proto_round(detail)
    soccer_ft = [l for l in lines if l.market is not None]
    print(f"[round {args.round}] {len(lines)} total lines, "
          f"{len(soccer_ft)} supported soccer full-time lines")

    history = load_history(HISTORY_DIR, limit=args.history_rounds)
    print(f"[fit] {len(history)} G016 score-history matches")

    if args.external_history:
        from betman_predictor.external_data import load_external_history
        ext = load_external_history(Path(args.external_history))
        history = list(history) + ext
        print(f"[fit] +{len(ext)} external matches "
              f"(now {len(history)} total)")

    rates = fit_team_rates(history)
    print(f"[fit] {len(rates.attack)} teams, {len(rates.league_mean_home)} leagues, "
          f"public_blend={args.public_blend}")

    evaluations = evaluate_lines(soccer_ft, rates, public_blend=args.public_blend)

    league_substrings = None
    if args.leagues:
        league_substrings = [s.strip() for s in args.leagues.split(",") if s.strip()]
    times_kst = None
    if args.time:
        times_kst = [s.strip() for s in args.time.split(",") if s.strip()]

    if args.per_match:
        filtered = _filter_evaluations(evaluations, league_substrings, times_kst, args.date)
        _print_per_match_best(filtered)

    if args.list:
        _print_list(evaluations, args.league)

    recs: list = []
    if args.recommend:
        recs = recommend_top_ev(evaluations, top_n=args.recommend, min_ev=args.min_ev)
        _print_recommendations(recs)

    parlay_result = None
    picks: list[Pick] = []
    if args.picks:
        picks = resolve_picks(args.picks, soccer_ft)
        parlay_result = evaluate_parlay(picks, evaluations)
        _print_parlay(picks, parlay_result, args.stake)

    if args.json_out:
        payload = {
            "round": args.round,
            "gm_id": args.gm_id,
            "supported_lines": len(soccer_ft),
            "recommendations": [
                {
                    "notice_no": r.line.notice_no,
                    "league_name": r.line.league_name,
                    "home_name": r.line.home_name,
                    "away_name": r.line.away_name,
                    "market": r.line.market,
                    "side": r.side,
                    "allot": r.allot,
                    "p_model": r.p_model,
                    "edge_pp": r.edge_pp,
                    "ev": r.ev,
                }
                for r in recs
            ],
            "parlay": (
                {
                    "n_legs": parlay_result.n_legs,
                    "joint_p": parlay_result.joint_p,
                    "parlay_allot": parlay_result.parlay_allot,
                    "fair_allot": parlay_result.fair_allot,
                    "ev": parlay_result.ev,
                    "legs": [
                        {
                            "notice_no": p.notice_no,
                            "side": p.side,
                            "side_label": p.side_label,
                            "allot": p.allot,
                            "p_model": parlay_result.p_models[i],
                            "home_name": p.line.home_name,
                            "away_name": p.line.away_name,
                            "league_name": p.line.league_name,
                            "market": p.line.market,
                        }
                        for i, p in enumerate(picks)
                    ],
                }
                if parlay_result is not None
                else None
            ),
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\nSaved to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
