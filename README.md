# Betman Betting Predictor

A research tool that builds outcome predictions for Korea's Betman markets. Four products are supported today:

- **`축구 승무패`** (`G011`) — soccer 1X2, 14-game Toto
- **`야구 승1패`** (`G024`) — baseball with 1-run handicap (승 / 1 / 패), 14-game Toto
- **`축구토토 스페셜 트리플`** (`G016`) — 3-game score-special; Dixon-Coles bivariate Poisson over a 6×6 grid (`0..5+`)
- **`프로토 승부식`** (`G101`) — line-by-line single-bet analysis across 1X2, Asian handicap, Over/Under, and SUM odd/even

It pulls live and historical round data from public Betman endpoints, trains models on the cached history, and prints a per-match betting plan with single picks, double-chance coverage, parlay probabilities, and (for proto) EV vs. the listed allot. For MLB matches the baseball predictor can enrich each prediction with probable starter ERA, last-window team form, pythag, and park factor pulled from the public MLB Stats API. For European soccer it can pre-seed Elo from football-data.co.uk CSVs and apply predict-time penalties for known-missing key players.

> ⚠️ **Research and automation only.** This is not investment advice and does not guarantee betting performance. Real betting markets have an inherent house edge; treat any output as one signal among many, not a recommendation.

## Features

### Core 14-game Toto (`app.py` — G011 / G024)

- **Three predictors, selectable at runtime:**
  - `elo` — lightweight Elo rating + kernel-calibrated 3-way probabilities. No extra deps.
  - `ml` — scikit-learn pipeline (`StandardScaler → LogisticRegression`, or `HistGradientBoostingClassifier` with `--ml-algo gbm`) trained on 47 engineered features per match (32 base + 15 MLB enrichment).
  - `ensemble` (default) — geometric-mean blend of Elo and ML probabilities, with tunable weight (`--ensemble-weight`).
- **Engineered features inspired by [ProphitBet](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor)** — Elo gap, rolling 8-match form rates (overall + venue-specific), lifetime W/D/L per team, rest days, league frequency, plus a strong **public vote-share** signal preserved historically by Betman's own `voteStatus` payload.
- **MLB enrichment via [statsapi.mlb.com](https://statsapi.mlb.com/api/v1/)** — opt-in display of probable starter ERA / WHIP / K9 (season + last-30-day), team OPS / OBP / SLG over the last ~18 games, Pythagorean win expectancy, runs scored / allowed per game, and stadium park factor. Cached on disk; concurrent prefetch with `ThreadPoolExecutor`.
- **Soccer Elo pre-seed (`--external-history-dir`)** — ingests football-data.co.uk CSVs (top-5 European leagues, ~14k matches per 5 seasons) and converts Dixon-Coles attack/defence ratios into Elo offsets so teams that never appeared in the cached Toto rounds still start with a sensible rating.
- **Lineup-aware penalties (`--lineup-players` + `--lineup-missing`)** — predict-time Elo penalty applied per team for publicly known missing starters. Does not touch training; only adjusts ratings forward when there is late-breaking team news.
- **Closing-odds blend (`--odds-file` + `--odds-weight`)** — geometric blend of the ML probabilities with a per-match closing-line distribution from Pinnacle or similar, to harden calibration on a small training set.
- **Isotonic calibration on a holdout fold** — on by default for the ML branch; disable with `--no-calibrate`.
- **Double-chance betting plan** — ranks the 14 matches by `1 − min(P)` weighted by entropy spread and applies `1X` / `12` / `X2` coverage to the N least-confident games.
- **`--top-n` focused mode** — reduce the printed plan to the N most-confident picks (e.g. `--top-n 4` for a focused 3-4 game ticket).
- **Manual parlay mode (`--match-seqs`)** — hand-pick specific match numbers (`--match-seqs 4,9,14`) and the report shows just those legs plus the joint parlay probability and a fair-payout multiplier. Combine with `--double-chance-count` to hedge the least-confident leg.
- **Graceful fallback** — if `scikit-learn` is missing, `--model ml` and `--model ensemble` automatically fall back to `elo` with a warning.

### G016 score-special triple (`app_score.py`)

- **Dixon-Coles bivariate Poisson** over a 6-bin score axis (`0,1,2,3,4,5+` → 36 cells per match). League-mean goals, team attack/defence multipliers, and the low-score `ρ` correction are all fit from the cached G016 history.
- **Public-vote blend (`--public-blend`)** — geometric mix of the model grid with Betman's published `voteStatusPlay3` allots, since the public is a useful informational prior at small sample sizes.
- **Top-K triple ranking** — builds joint parlays from the strongest cells in each of the 3 matches, ranked by joint hit probability, with stake/EV breakdown.
- **Combo decomposition (`--decompose 36,50,72`)** — for fixed combo counts (`a×b×c`), shows the best way to split your ticket count across the three matches.
- **Explicit-pick evaluation (`--explicit-picks`)** — score a hand-built triple like `1,2|1,2;2,3|1,2;1,2,3|1` and see its model probability and EV.

### G101 proto 승부식 (`app_proto.py`)

- **Per-line model probabilities** for the four supported full-time soccer markets:
  - `betId=1` 승무패 (1X2) — H / D / A
  - `betId=5` 일반 승부핸디캡 (Asian handicap) — H / D / A (D only on integer lines)
  - `betId=78` 일반 언더오버 (Total) — U / O
  - `betId=17` 일반 홀짝 (SUM odd/even) — ODD / EVEN
- **EV = p_model × allot − 1** computed per side, per line. The recommender ranks every line in the round by EV and surfaces the top-N value bets.
- **Manual parlay scoring (`--picks "215944,215945:O,215946"`)** — auto-detects the side for bare `noticeNo`s (picks the lowest-allot side as the implied favorite), accepts explicit codes (`H/D/A/O/U/ODD/EVEN`) or Korean labels (`홈/무/원/오버/언더/홀/짝`), and returns each leg's model probability, market probability, EV, and joint parlay metrics.
- **Listing mode (`--list`)** — dump every supported full-time soccer line with model P, market P, and EV; filter by league with `--league EPL`.
- Probabilities feed off the same Dixon-Coles 6×6 grid as `app_score.py` so attack/defence ratings transfer between the two soccer-special CLIs.

## Quickstart

```bash
git clone https://github.com/<you>/betman-betting-predictor
cd betman-betting-predictor
python install.py     # installs requirements; recommends Python 3.11+
python app.py         # runs both 14-game markets with default settings
```

Windows users can also double-click `app.bat` after installation.

## Usage

### Both 14-game markets, default ensemble model

```powershell
python app.py
```

### Focused 3-4 game baseball ticket with MLB enrichment

```powershell
python app.py --market baseball --round 260013 --model ensemble --ensemble-weight 0.8 --ml-algo logreg --enrich-features --top-n 4 --stake-total 32000
```

### Hand-picked 2-5 game parlay

```powershell
python app.py --market baseball --round 260013 --model ensemble --ensemble-weight 0.8 --enrich-features --match-seqs 4,9,14
```

### Soccer round with external history pre-seed + lineup penalties + closing odds

```powershell
python app.py --market soccer --model ensemble --external-history-dir data/footballdata `
              --lineup-players data/players.json --lineup-missing data/lineups.json `
              --odds-file data/odds.json --odds-weight 0.35
```

### G016 score-special triple

```powershell
python app_score.py --round 260017 --budget 5000 --public-blend 0.25 --decompose 36,50,72
```

### G101 proto value bets

```powershell
python app_proto.py --round 260051 --recommend 20
python app_proto.py --round 260051 --picks "215944:H,215779:오버,215802"
python app_proto.py --round 260051 --list --league EPL
```

Pass match numbers as a comma-separated list (`--match-seqs`). The report prints just those legs and shows the parlay probability and fair-payout multiplier. To hedge the least-confident leg with double-chance coverage, add `--double-chance-count 1` (or 2 for two hedges). Combining manual seqs with double-chance applies the hedge *within* your chosen set, not across the whole 14-game round.

### `app.py` flags

| Flag | Default | What it does |
|---|---|---|
| `--market {soccer,baseball,all}` | `all` | Limit to one market. |
| `--round N` | (auto-discover) | Override the target round (`gmTs`). Single-market only. |
| `--history-rounds N` | 140 / 110 | Training window length. |
| `--model {elo,ml,ensemble}` | `ensemble` | Predictor type. |
| `--ml-algo {logreg,gbm}` | `logreg` | scikit-learn classifier. |
| `--ensemble-weight 0..1` | `0.5` | Elo weight in ensemble (0.8 = Elo-heavy, recommended for baseball). |
| `--double-chance-count N` | `0` | Apply `1X`/`12`/`X2` coverage to the N least-confident matches. |
| `--top-n N` | `0` | Print only the N highest-confidence picks. |
| `--match-seqs 4,9,14` | (all) | Hand-pick specific match numbers for a manual parlay. |
| `--stake-total KRW` | `0` | Total stake to split evenly across tickets (display only). |
| `--enrich-mlb` | off | Display probable starter ERA / pythag / park factor in the report. |
| `--enrich-features` | off | Also feed MLB enrichment into the ML model as training features (slow first run). |
| `--enrich-workers N` | `8` | Concurrent threads for MLB API prefetch. |
| `--external-history-dir DIR` | — | Pre-seed Elo from football-data.co.uk CSVs (soccer only). |
| `--lineup-players PATH` | — | JSON of `{league: {team: {player: elo_value}}}` for lineup penalty weights. |
| `--lineup-missing PATH` | — | JSON of `{gm_ts: {match_seq: {home_missing, away_missing}}}`. |
| `--odds-file PATH` | — | JSON of `{gm_ts: {match_seq: {A, B, D: prob}}}` for closing-odds blend (ML only). |
| `--odds-weight 0..1` | `0.0` | Weight of closing-odds in the geometric blend with ML. |
| `--no-calibrate` | off | Disable isotonic per-class calibration on a holdout fold. |
| `--refresh` | off | Bypass local cache and re-fetch Betman data. |
| `--json-out path` | — | Write the full prediction report as JSON. |

## MCP server

`mcp_server.py` exposes the predictor over the Model Context Protocol so it can be driven from Claude Desktop / Claude Code / any MCP-aware client.

```bash
python -m pip install mcp
python mcp_server.py
```

Or wire it into a client config:

```json
{
  "mcpServers": {
    "betman": {
      "command": "python",
      "args": ["C:/Projects/betman-betting-predictor/mcp_server.py"]
    }
  }
}
```

Tools exposed:

- `list_markets()` — supported markets and label maps.
- `discover_target_round(market, refresh?)` — next upcoming round.
- `list_recent_rounds(market, limit, refresh?)` — recent closed rounds.
- `get_round_matches(market, round, refresh?)` — match slots for a round without predicting.
- `predict_round(market, round?, model?, ml_algo?, ensemble_weight?, history_rounds?, double_chance_count?, top_n?, match_seqs?, enrich_mlb?, refresh?)` — full structured report.

A matching `SKILL.md` is shipped for Claude Code so the assistant can pick the right tool, model, and flags automatically when a user asks for a Betman pick.

## Architecture

```
betman-betting-predictor/
├── app.py                                 # CLI: 14-game G011 / G024
├── app_score.py                           # CLI: G016 score-special triple
├── app_proto.py                           # CLI: G101 proto 승부식
├── mcp_server.py                          # MCP tool surface (FastMCP)
├── SKILL.md                               # Claude Code skill metadata
├── install.py                             # one-shot installer
├── requirements.txt
├── betman_predictor/
│   ├── client.py                          # Betman public-endpoint HTTP client + on-disk cache
│   ├── config.py                          # Market definitions (G011 soccer, G024 baseball)
│   ├── models.py                          # MatchRecord, Prediction, RoundReference dataclasses
│   ├── predictor.py                       # Elo + kernel-calibrated 3-way model + vote extraction
│   ├── ml_predictor.py                    # FeatureBuilder + sklearn MLPredictor + ensemble blender
│   ├── double_chance.py                   # Bet-plan selector (1X/12/X2 ranking)
│   ├── score_special.py                   # Dixon-Coles bivariate Poisson for G016
│   ├── proto.py                           # G101 line parser, market probs, EV, parlay resolver
│   ├── external_data.py                   # football-data.co.uk CSV ingest → Elo pre-seed
│   ├── lineup_adjust.py                   # Predict-time lineup-based Elo penalties
│   └── baseball_data/
│       ├── mlb_client.py                  # MLB Stats API client (concurrent prefetch, week-bucketed cache)
│       ├── team_mapping.py                # Betman 2-letter codes → MLB team IDs + park factor table
│       └── enricher.py                    # High-level lookup_for(match) returning MatchEnrichment
├── cache/                                 # local API responses (gitignored)
└── README.md
```

The on-disk cache keeps Betman round JSONs (rarely change once settled) and MLB API responses (immutable for completed dates). First runs are slow; subsequent runs are instant. To force-refresh, pass `--refresh`.

## Backtest results (rolling-origin, 20 held-out 2026 baseball rounds, 275 matches)

| Model | Accuracy |
|---|---:|
| Random baseline | 33.3% |
| Elo only | 39.6% |
| ML logreg + enrich | 35.6% |
| ML gbm + enrich | 37.1% |
| Ensemble 50/50 + enrich | 41.8% |
| **Ensemble 80/20 Elo-heavy + enrich** | **42.5%** |

The ensemble beats either component alone by 3-6 pp because Elo and ML make *different* mistakes — averaging them is a Bayesian-style smoothing. Pure ML loses to Elo at this training-set size because the 47 features can overfit ~1300 matches; Elo's 2 parameters can't.

A 95% CI on a 275-sample 3-way evaluation is roughly ±4 pp, so the ranking is more reliable than any single number. **Honest individual-game prediction caps around 55-60% even for professional baseball models** — a lot of MLB outcome variance is irreducible noise (talent gaps still produce upsets ~25% of the time on raw skill alone).

## Acknowledgements

- **[ProphitBet](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor)** — feature-engineering inspiration (rolling-window team form, lifetime rates).
- **[MLB Stats API](https://statsapi.mlb.com)** — free public API that powers the baseball enrichment.
- **[football-data.co.uk](https://www.football-data.co.uk/)** — free historical CSVs that drive the soccer Elo pre-seed.
- **Dixon & Coles (1997)** — bivariate Poisson with low-score correction; backbone of the score-special and proto models.
- **Betman public endpoints** — vote-share + match metadata.

## Roadmap (open items)

- **KBO support** — the MLB enricher silently skips KBO matches in mixed rounds. A statiz / mykbostats adapter would close that gap (~40% of cached G024 rounds).
- **Bookmaker odds wiring** — `--odds-file` is plumbed end-to-end but there is no automated ingest yet from [The Odds API](https://the-odds-api.com) or similar.
- **Half-time / non-soccer proto markets** — `proto.py` parses every line but only the four full-time soccer markets get model probabilities today.

## License

MIT — see [LICENSE](LICENSE).
