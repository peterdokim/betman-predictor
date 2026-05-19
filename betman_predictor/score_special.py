"""
G016 축구토토 스페셜 트리플 (Score Special Triple) predictor.

Each ticket is one cell per match across 3 matches. Each cell is (home_bin, away_bin)
where bins map raw scores to {0, 1, 2, 3, 4, 5+} -> indices {0..5}. So 6x6 = 36 cells per
match, 36^3 = 46,656 raw triple combinations -- but we score & rank cells per match
independently and form parlays from the top.

Probabilities come from a Dixon-Coles-style bivariate Poisson:
  - league baseline mean goals (lambda_league_home, lambda_league_away)
  - team attack/defence multipliers fit from history with a low-rho correction for
    the well-known excess of 0-0/1-1/0-1/1-0 vs independent Poisson.
  - rho (Dixon-Coles correction) fixed at -0.10 (typical fit for top-5 leagues).
The cell prob is then aggregated across the bins (last bin = 5+ tail).

Vote-share feed (voteStatusPlay3) gives Betman cell allots; we expose them so the
caller can compute EV = p_model * allot - 1 per cell.

This module is intentionally independent of predictor.py / ml_predictor.py because
the label space is 36-class per match, not 3-class. The existing soccer Elo (G011)
is reused as a *prior* on team strength to bootstrap teams without enough G016 history.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants

SCORE_BINS = 6                  # 0,1,2,3,4,5+
TAIL_INDEX = SCORE_BINS - 1     # 5+
MAX_GOALS_FOR_PMF = 12          # truncate the Poisson sum
DEFAULT_RHO = -0.10             # Dixon-Coles low-score correction
DEFAULT_LEAGUE_MEAN = 2.7       # fallback total goals/match (PL-ish)
HOME_GOAL_SHARE = 0.55          # split of league mean between home/away
SHRINK_TO_LEAGUE = 6.0          # shrinkage games -- low N teams pull toward 1.0


@dataclass(frozen=True)
class ScoreMatch:
    """One match inside a G016 round."""
    gm_ts: int
    match_seq: int
    league_code: str
    league_name: str
    home_id: str
    home_name: str
    away_id: str
    away_name: str
    game_datetime: datetime | None
    game_date_str: str
    home_score: int | None         # None for unsettled (target round)
    away_score: int | None


@dataclass
class CellPrediction:
    match_seq: int
    home_bin: int                   # 0..5 (5 = "5+")
    away_bin: int
    label: str                      # e.g. "1:2" or "5+:0"
    p_model: float                  # model probability for this cell
    allot: float | None             # Betman triple-mode allot (배당) for this cell
    ev: float | None                # p_model * allot - 1, or None when no allot

    @property
    def gameResult_code(self) -> str:
        # Betman encodes the cell as a 2-char string: f"{home_bin}{away_bin}"
        return f"{self.home_bin}{self.away_bin}"


@dataclass
class TripleTicket:
    cells: tuple[CellPrediction, CellPrediction, CellPrediction]

    @property
    def p_hit(self) -> float:
        return self.cells[0].p_model * self.cells[1].p_model * self.cells[2].p_model

    @property
    def fair_multiplier(self) -> float:
        return 1.0 / self.p_hit if self.p_hit > 0 else float("inf")

    @property
    def label(self) -> str:
        return " | ".join(c.label for c in self.cells)


# ---------------------------------------------------------------------------
# Round-detail extraction


def _parse_dt(ms: int | None) -> datetime | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()


def _bin_score(raw: int) -> int:
    if raw < 0:
        return 0
    if raw >= TAIL_INDEX:
        return TAIL_INDEX
    return raw


def _decode_game_result(code: str | None) -> tuple[int, int] | None:
    """Betman's gameResult for G016 is a 2-digit string '<home_bin><away_bin>'."""
    if not code:
        return None
    s = str(code).strip()
    if len(s) != 2 or not s.isdigit():
        return None
    return int(s[0]), int(s[1])


def extract_score_matches(detail: dict, gm_ts_override: int | None = None) -> list[ScoreMatch]:
    out: list[ScoreMatch] = []
    gm_ts = gm_ts_override or int(detail.get("gmTs") or 0)
    for row in detail.get("schedulesList", []) or []:
        decoded = _decode_game_result(row.get("gameResult"))
        home_score = decoded[0] if decoded else None
        away_score = decoded[1] if decoded else None
        # NOTE: We key both home_id and away_id on the *display name* so external
        # football-data history (English/Korean names) can share team-rating keys
        # with Betman's own G016 history. Betman's numeric homeId is ignored; this
        # is fine because Betman keeps homeName stable across rounds.
        home_name = str(row.get("homeName") or "HOME")
        away_name = str(row.get("awayName") or "AWAY")
        out.append(ScoreMatch(
            gm_ts=gm_ts,
            match_seq=int(row.get("matchSeq") or 0),
            league_code=str(row.get("leagueCode") or "UNKNOWN"),
            league_name=str(row.get("leagueName") or "Unknown League"),
            home_id=home_name,
            home_name=home_name,
            away_id=away_name,
            away_name=away_name,
            game_datetime=_parse_dt(row.get("gameDate")),
            game_date_str=str(row.get("gameDateStr") or ""),
            home_score=home_score,
            away_score=away_score,
        ))
    out.sort(key=lambda m: m.match_seq)
    return out


def extract_market_grids(detail: dict) -> dict[str, list[list[float]]]:
    """Pull the three vote/allot grids exposed by the round detail.

    - voteStatusPlay1: cumulative public vote counts across the *single-match* score
      market (G008 score-prediction); the values are vote counts, not allots.
    - voteStatusPlay2: vote counts for double-match score combos (G015).
    - voteStatusPlay3: vote counts AND allot multipliers for the triple-match combo
      (G016) -- but the allot here is for one specific triple ticket, not per cell.

    In practice the only directly useful per-match-cell signal we can extract is the
    *vote share* in Play1, which acts as a market-implied probability for each cell of
    each match. We aggregate Play1 across all matches in this round (the API ships
    Play1 as a single 6x6 grid representing the public's distribution across the
    score market). Callers that want per-match market priors should query G008 for
    that specific match -- which is out of scope here.

    Returns {grid_name: 6x6 grid}. Empty dict if no grids are present.
    """
    out: dict[str, list[list[float]]] = {}
    for key in ("voteStatusPlay1", "voteStatusPlay2", "voteStatusPlay3"):
        block = detail.get(key) or {}
        home_list = block.get("homeVoteStatusList") or []
        if not home_list:
            continue
        grid: list[list[float]] = []
        for entry in home_list[:SCORE_BINS]:
            away_list = (entry or {}).get("awayVoteStatusList") or []
            row: list[float] = []
            for cell in away_list[:SCORE_BINS]:
                cell = cell or {}
                # voteStatusPlay3 carries 'allot' (multiplier); Play1/Play2 carry only voteCount.
                value = cell.get("allot") if key == "voteStatusPlay3" else cell.get("voteCount")
                row.append(float(value or 0.0))
            while len(row) < SCORE_BINS:
                row.append(0.0)
            grid.append(row)
        while len(grid) < SCORE_BINS:
            grid.append([0.0] * SCORE_BINS)
        out[key] = grid
    return out


def public_vote_distribution(detail: dict) -> list[list[float]]:
    """Return the 6x6 public-vote distribution for the single-match score market.

    The values in voteStatusPlay1 are raw vote counts, summed across whichever cell
    the public picked. Normalising gives a market-implied probability map.
    """
    grids = extract_market_grids(detail)
    play1 = grids.get("voteStatusPlay1")
    if not play1:
        return [[1.0 / (SCORE_BINS * SCORE_BINS)] * SCORE_BINS for _ in range(SCORE_BINS)]
    total = sum(sum(row) for row in play1) or 1.0
    return [[v / total for v in row] for row in play1]


# ---------------------------------------------------------------------------
# Dixon-Coles bivariate Poisson


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _dc_correction(home_goals: int, away_goals: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles tau() multiplier on the (h,a) cell of independent Poissons."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam_h * lam_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lam_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_grid_probabilities(lam_h: float, lam_a: float, rho: float = DEFAULT_RHO) -> list[list[float]]:
    """Return 6x6 grid where cell[h][a] is P(home_bin=h, away_bin=a).

    Internally we evaluate raw score pairs up to (MAX_GOALS_FOR_PMF, MAX_GOALS_FOR_PMF),
    then aggregate the tail (>= 5) into bin index 5.
    """
    grid = [[0.0] * SCORE_BINS for _ in range(SCORE_BINS)]
    for h in range(MAX_GOALS_FOR_PMF + 1):
        ph = _poisson_pmf(h, lam_h)
        if ph < 1e-12:
            continue
        for a in range(MAX_GOALS_FOR_PMF + 1):
            pa = _poisson_pmf(a, lam_a)
            if pa < 1e-12:
                continue
            p = ph * pa * _dc_correction(min(h, 1), min(a, 1), lam_h, lam_a, rho)
            if p <= 0:
                continue
            grid[_bin_score(h)][_bin_score(a)] += p
    # renormalise -- the DC correction breaks unit mass slightly
    total = sum(sum(row) for row in grid)
    if total > 0:
        for h in range(SCORE_BINS):
            for a in range(SCORE_BINS):
                grid[h][a] /= total
    return grid


# ---------------------------------------------------------------------------
# Team strength model


@dataclass
class TeamRates:
    attack: dict[str, float] = field(default_factory=dict)        # team_key -> mean goals scored
    defence: dict[str, float] = field(default_factory=dict)       # team_key -> mean goals conceded
    games: dict[str, int] = field(default_factory=dict)
    league_mean_home: dict[str, float] = field(default_factory=dict)   # league_code -> mean home goals
    league_mean_away: dict[str, float] = field(default_factory=dict)


def _team_key(league_code: str, team_id: str) -> str:
    return f"{league_code}:{team_id}"


def _untail(b: int) -> int:
    """Treat tail bin (5+) as exactly 5 for fitting -- coarse but stable."""
    return min(b, TAIL_INDEX)


def fit_team_rates(history: Iterable[ScoreMatch]) -> TeamRates:
    rates = TeamRates()

    # Pass 1: league means (home + away separately)
    home_totals: dict[str, list[int]] = {}
    away_totals: dict[str, list[int]] = {}
    for m in history:
        if m.home_score is None or m.away_score is None:
            continue
        home_totals.setdefault(m.league_code, []).append(_untail(m.home_score))
        away_totals.setdefault(m.league_code, []).append(_untail(m.away_score))

    for lg, vals in home_totals.items():
        rates.league_mean_home[lg] = (sum(vals) / len(vals)) if vals else DEFAULT_LEAGUE_MEAN * HOME_GOAL_SHARE
    for lg, vals in away_totals.items():
        rates.league_mean_away[lg] = (sum(vals) / len(vals)) if vals else DEFAULT_LEAGUE_MEAN * (1 - HOME_GOAL_SHARE)

    # Pass 2: team attack/defence as ratio vs league mean for the venue, shrunk by N games.
    team_for: dict[str, list[float]] = {}     # goals scored ratio
    team_against: dict[str, list[float]] = {}
    team_games: dict[str, int] = {}

    for m in history:
        if m.home_score is None or m.away_score is None:
            continue
        lg_h = rates.league_mean_home.get(m.league_code) or (DEFAULT_LEAGUE_MEAN * HOME_GOAL_SHARE)
        lg_a = rates.league_mean_away.get(m.league_code) or (DEFAULT_LEAGUE_MEAN * (1 - HOME_GOAL_SHARE))

        h_key = _team_key(m.league_code, m.home_id)
        a_key = _team_key(m.league_code, m.away_id)

        team_for.setdefault(h_key, []).append(_untail(m.home_score) / max(lg_h, 0.1))
        team_against.setdefault(h_key, []).append(_untail(m.away_score) / max(lg_a, 0.1))
        team_for.setdefault(a_key, []).append(_untail(m.away_score) / max(lg_a, 0.1))
        team_against.setdefault(a_key, []).append(_untail(m.home_score) / max(lg_h, 0.1))
        team_games[h_key] = team_games.get(h_key, 0) + 1
        team_games[a_key] = team_games.get(a_key, 0) + 1

    for key, ratios in team_for.items():
        n = len(ratios)
        sample = sum(ratios) / n
        # Shrink: prior is 1.0 (league average) with weight SHRINK_TO_LEAGUE
        rates.attack[key] = (sample * n + 1.0 * SHRINK_TO_LEAGUE) / (n + SHRINK_TO_LEAGUE)
    for key, ratios in team_against.items():
        n = len(ratios)
        sample = sum(ratios) / n
        rates.defence[key] = (sample * n + 1.0 * SHRINK_TO_LEAGUE) / (n + SHRINK_TO_LEAGUE)
    rates.games = team_games
    return rates


def predict_match_grid(match: ScoreMatch, rates: TeamRates, rho: float = DEFAULT_RHO) -> list[list[float]]:
    lg_h = rates.league_mean_home.get(match.league_code) or (DEFAULT_LEAGUE_MEAN * HOME_GOAL_SHARE)
    lg_a = rates.league_mean_away.get(match.league_code) or (DEFAULT_LEAGUE_MEAN * (1 - HOME_GOAL_SHARE))
    h_atk = rates.attack.get(_team_key(match.league_code, match.home_id), 1.0)
    h_def = rates.defence.get(_team_key(match.league_code, match.home_id), 1.0)
    a_atk = rates.attack.get(_team_key(match.league_code, match.away_id), 1.0)
    a_def = rates.defence.get(_team_key(match.league_code, match.away_id), 1.0)
    lam_h = max(0.05, lg_h * h_atk * a_def)
    lam_a = max(0.05, lg_a * a_atk * h_def)
    return score_grid_probabilities(lam_h, lam_a, rho=rho)


# ---------------------------------------------------------------------------
# Top-N picks per match + parlay enumeration


def cell_label(home_bin: int, away_bin: int) -> str:
    h = "5+" if home_bin >= TAIL_INDEX else str(home_bin)
    a = "5+" if away_bin >= TAIL_INDEX else str(away_bin)
    return f"{h}:{a}"


def predict_round_cells(
    target_matches: Sequence[ScoreMatch],
    rates: TeamRates,
    public_vote_grid: list[list[float]] | None = None,
    rho: float = DEFAULT_RHO,
    public_blend: float = 0.0,
) -> dict[int, list[CellPrediction]]:
    """Build per-match 6x6 cell predictions.

    If `public_vote_grid` is supplied (e.g. from voteStatusPlay1), we can softly blend
    it into the model output as `public_blend * public + (1-public_blend) * model`.
    Default 0.0 = pure Dixon-Coles. Useful for value-detection: cells where
    p_model >> p_public are "model thinks the market is wrong."
    """
    out: dict[int, list[CellPrediction]] = {}
    public_blend = max(0.0, min(1.0, public_blend))
    for match in target_matches:
        grid = predict_match_grid(match, rates, rho=rho)
        if public_vote_grid and public_blend > 0.0:
            grid = [
                [(1 - public_blend) * grid[h][a] + public_blend * public_vote_grid[h][a]
                 for a in range(SCORE_BINS)]
                for h in range(SCORE_BINS)
            ]
            total = sum(sum(row) for row in grid) or 1.0
            grid = [[v / total for v in row] for row in grid]
        cells: list[CellPrediction] = []
        for h in range(SCORE_BINS):
            for a in range(SCORE_BINS):
                cells.append(CellPrediction(
                    match_seq=match.match_seq,
                    home_bin=h,
                    away_bin=a,
                    label=cell_label(h, a),
                    p_model=grid[h][a],
                    allot=None,
                    ev=None,
                ))
        cells.sort(key=lambda c: c.p_model, reverse=True)
        out[match.match_seq] = cells
    return out


def build_top_triples(
    cells_by_match: dict[int, list[CellPrediction]],
    budget_tickets: int,
) -> list[TripleTicket]:
    """Pick the `budget_tickets` triples with the highest joint hit probability.

    With independent matches, joint P = p1 * p2 * p3, so the top-K triples are simply
    the cartesian product of each match's sorted-by-p cells, expanded greedily.
    We use a best-first search: maintain a heap of (negative joint_p, indices) and pop
    K times; each popped triple expands its 3 neighbours by incrementing one index.
    """
    seqs = sorted(cells_by_match.keys())
    if len(seqs) != 3:
        raise ValueError(f"Score-special triple needs exactly 3 matches, got {len(seqs)}")

    a, b, c = (cells_by_match[s] for s in seqs)
    import heapq

    seen: set[tuple[int, int, int]] = set()
    start = (0, 0, 0)
    heap: list[tuple[float, tuple[int, int, int]]] = [(-(a[0].p_model * b[0].p_model * c[0].p_model), start)]
    seen.add(start)
    out: list[TripleTicket] = []

    while heap and len(out) < budget_tickets:
        neg_p, (i, j, k) = heapq.heappop(heap)
        out.append(TripleTicket(cells=(a[i], b[j], c[k])))
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if ni >= len(a) or nj >= len(b) or nk >= len(c):
                continue
            key = (ni, nj, nk)
            if key in seen:
                continue
            seen.add(key)
            joint = a[ni].p_model * b[nj].p_model * c[nk].p_model
            heapq.heappush(heap, (-joint, key))
    return out


# ---------------------------------------------------------------------------
# History loading helper


def load_history(cache_dir: Path, limit: int = 80, exclude_gm_ts: int | None = None) -> list[ScoreMatch]:
    out: list[ScoreMatch] = []
    files = sorted(cache_dir.glob("G016-*.json"), reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        gm_ts = int(data.get("gmTs") or 0)
        if exclude_gm_ts and gm_ts == exclude_gm_ts:
            continue
        matches = extract_score_matches(data, gm_ts_override=gm_ts)
        # Only keep matches with a settled score
        out.extend(m for m in matches if m.home_score is not None and m.away_score is not None)
        if len({m.gm_ts for m in out}) >= limit:
            break
    return out
