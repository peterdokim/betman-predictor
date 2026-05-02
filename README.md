# Betman Betting Predictor

A research tool that builds outcome predictions for Korea's Betman 14-game Toto markets:

- **`축구 승무패`** (`G011`) — soccer 1X2
- **`야구 승1패`** (`G024`) — baseball with 1-run handicap (승 / 1 / 패)

It pulls live and historical round data from public Betman endpoints, trains models on the cached history, and prints a per-match betting plan with single picks and double-chance coverage. For MLB matches, it can also enrich each prediction with probable starter ERA, last-window team form, and park factor pulled from the public MLB Stats API.

> ⚠️ **Research and automation only.** This is not investment advice and does not guarantee betting performance. Real betting markets have an inherent house edge; treat any output as one signal among many, not a recommendation.

## Features

- **Three predictors, selectable at runtime:**
  - `elo` — lightweight Elo rating + kernel-calibrated 3-way probabilities. No extra deps.
  - `ml` — scikit-learn pipeline (`StandardScaler → LogisticRegression`, or `HistGradientBoostingClassifier` with `--ml-algo gbm`) trained on 32 engineered features per match.
  - `ensemble` (default) — geometric-mean blend of Elo and ML probabilities, with tunable weight.

- **Engineered features inspired by [ProphitBet](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor)** — Elo gap, rolling 8-match form rates (overall + venue-specific), lifetime W/D/L per team, rest days, league frequency, plus a strong **public vote-share** signal preserved historically by Betman's own `voteStatus` payload.

- **MLB enrichment via [statsapi.mlb.com](https://statsapi.mlb.com/api/v1/)** — opt-in display of probable starter ERA / WHIP / K9 (season + last-30-day), team OPS / OBP / SLG over the last ~18 games, Pythagorean win expectancy, runs scored / allowed per game, and stadium park factor. Cached on disk; concurrent prefetch with `ThreadPoolExecutor`.

- **Double-chance betting plan** — ranks the 14 matches by `1 − min(P)` weighted by entropy spread and applies `1X` / `12` / `X2` coverage to the N least-confident games.

- **`--top-n` focused mode** — reduce the printed plan to the N most-confident picks (e.g. `--top-n 4` for a focused 3-4 game ticket).

- **Graceful fallback** — if `scikit-learn` is missing, `--model ml` and `--model ensemble` automatically fall back to `elo` with a warning.

## Quickstart

```bash
git clone https://github.com/<you>/betman-betting-predictor
cd betman-betting-predictor
python install.py     # installs requirements; recommends Python 3.11+
python app.py         # runs both markets with default settings
```

Windows users can also double-click `app.bat` after installation.

## Usage

### Both markets, default ensemble model

```powershell
python app.py
```

### Focused 3-4 game baseball ticket with MLB enrichment

```powershell
python app.py --market baseball --round 260013 --model ensemble --ensemble-weight 0.8 --ml-algo logreg --enrich-features --top-n 4 --stake-total 32000
```

### Other useful flags

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
| `--stake-total KRW` | `0` | Total stake to split evenly across tickets (display only). |
| `--enrich-mlb` | off | Display probable starter ERA / pythag / park factor in the report. |
| `--enrich-features` | off | Also feed MLB enrichment into the ML model as training features (slow first run). |
| `--enrich-workers N` | `8` | Concurrent threads for MLB API prefetch. |
| `--refresh` | off | Bypass local cache and re-fetch Betman data. |
| `--json-out path` | — | Write the full prediction report as JSON. |

## Architecture

```
betman-betting-predictor/
├── app.py                                 # CLI entry point
├── install.py                             # one-shot installer
├── requirements.txt
├── betman_predictor/
│   ├── client.py                          # Betman public-endpoint HTTP client + on-disk cache
│   ├── config.py                          # Market definitions (G011 soccer, G024 baseball)
│   ├── models.py                          # MatchRecord, Prediction, RoundReference dataclasses
│   ├── predictor.py                       # Elo + kernel-calibrated 3-way model + vote extraction
│   ├── ml_predictor.py                    # FeatureBuilder + sklearn MLPredictor + ensemble blender
│   ├── double_chance.py                   # Bet-plan selector (1X/12/X2 ranking)
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
- **Betman public endpoints** — vote-share + match metadata.

## Roadmap (open items)

- **KBO support** — the MLB enricher silently skips KBO matches in mixed rounds. A statiz / mykbostats adapter would close that gap (~40% of cached G024 rounds).
- **Bookmaker odds integration** — adding Pinnacle moneyline (via [The Odds API](https://the-odds-api.com)) as a calibration / value-detection signal.
- **Lineup-aware features** — per-batter wOBA vs the opposing starter's handedness. Real upside but requires the announced-lineup window (typically 2-4 hours pre-game).

## License

MIT — see [LICENSE](LICENSE).
