---
name: betman-predict
description: Use this skill when the user wants outcome predictions for Korea's Betman 14-game Toto rounds — soccer 승무패 (G011) or baseball 승1패 (G024). Covers picking a round, choosing a model (elo / ml / ensemble), MLB enrichment, double-chance hedging, top-N focusing, and hand-picked parlays. Triggers on phrases like "predict the round", "betman pick", "야구 승1패", "축구 승무패", or any reference to gmTs / G011 / G024.
---

# Betman prediction skill

This repo predicts Korean Betman 14-game Toto markets. Drive it either through the **MCP server** (`mcp_server.py`, preferred when wired into the client) or the **CLI** (`python app.py`).

## When to use this skill

- User asks for a prediction, betting plan, or pick for a Betman round.
- User mentions one of the markets: `축구 승무패` / `soccer` / `G011`, or `야구 승1패` / `baseball` / `G024`.
- User gives a round identifier (`gmTs`) or asks "what's the next round."

Do **not** use this skill for general sports betting advice unrelated to Betman, or for markets outside G011 / G024.

## Decision tree

1. **Which market?** Default to running both (`market="all"` in CLI). If only one is asked for, run just that one.
2. **Which round?** If the user gives a `gmTs`, use it. Otherwise call `discover_target_round` (MCP) or omit `--round` (CLI) to auto-pick the next upcoming round.
3. **Which model?** Default to `ensemble`. For baseball, prefer `ensemble_weight=0.8` (Elo-heavy beat 50/50 by ~1 pp on the backtest).
4. **MLB enrichment?** Turn on (`enrich_mlb=True` / `--enrich-mlb`) only for baseball rounds with MLB games. First-time prefetch is slow; subsequent runs hit the cache.
5. **Bet shape:**
   - Full 14-game ticket → no extra flags.
   - Focused 3-4 game ticket → `top_n=4` / `--top-n 4`.
   - Hand-picked parlay → `match_seqs=[4,9,14]` / `--match-seqs 4,9,14`.
   - Hedge least-confident legs → `double_chance_count=N`.

## MCP tools (preferred)

If the `betman` MCP server is connected, prefer it over shelling out:

- `list_markets()` — confirm the two supported markets.
- `discover_target_round(market)` — find the next round to predict.
- `list_recent_rounds(market, limit)` — browse recent closed rounds.
- `get_round_matches(market, round)` — see the 14 match slots without predicting.
- `predict_round(market, round?, model?, ensemble_weight?, top_n?, match_seqs?, double_chance_count?, enrich_mlb?)` — the main tool. Returns a structured JSON report.

## CLI (fallback)

```bash
# Auto-discover both markets, default ensemble
python app.py

# Focused 4-game baseball ticket with MLB enrichment, Elo-heavy ensemble
python app.py --market baseball --model ensemble --ensemble-weight 0.8 --enrich-mlb --top-n 4

# Hand-picked parlay with one double-chance hedge
python app.py --market baseball --round 260013 --match-seqs 4,9,14 --double-chance-count 1
```

Full flag list lives in `README.md`. For JSON output add `--json-out report.json`.

## Reading the output

Each pick has:
- `pick` / `bet_label` — the recommended bet (`승` / `무` / `패` for soccer, `승` / `1` / `패` for baseball, or a double-chance like `1X`).
- `bet_probability_pct` — model's probability the bet hits.
- `probabilities_pct` — full 3-way distribution.
- `rating_gap` — Elo gap (positive = home favored).
- `enrichment` (baseball only) — starter ERA, pythag, park factor.

The `summary.all_correct_probability` is the joint parlay probability across the included legs; `1 / that` is the fair-payout multiplier.

## Honest caveats to include in any answer

- **Not investment advice.** Betman has a house edge; this is a research signal.
- **Backtest accuracy caps around 42-43%** for baseball ensembles (random = 33%, individual-game ceiling ~55-60%). Don't promise higher.
- **Sample-size caveats** — model confidence on a single pick is noisy; the parlay number is more informative than any individual leg.

## Setup notes

- First-time setup: `python install.py` (installs `requests`, `numpy`, `scikit-learn`).
- For the MCP server: also `python -m pip install mcp`.
- Cache lives in `./cache/` (gitignored); first runs hit the network, later runs are instant.
- If scikit-learn is missing, `ml` / `ensemble` silently fall back to `elo`.
