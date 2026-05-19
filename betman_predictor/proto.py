"""
Betman 프로토 승부식 (Proto win/loss / handicap / over-under / sum) integration.

The round detail for G101 ships a flattened table at compSchedules.datas: ~1000 rows,
each row is one bet line for one match. Same match appears multiple times, once per
market (1X2 / handicap line / O/U line / SUM holzzak).

This module:
  1. Parses compSchedules into structured ProtoLine records.
  2. Converts a Dixon-Coles 6x6 score grid (from score_special.py) into per-market
     model probabilities (P_win, P_under, P_odd ...).
  3. Computes EV = p_model * allot - 1 per side per line.
  4. Resolves user picks by noticeNo + auto-detected pick code.

Markets handled (full-time soccer only):
  betId=1   승무패          (1X2)               -- side: H / D / A
  betId=5   일반 승부핸디캡   (Asian handicap)   -- side: H / D / A (D only if integer line)
  betId=78  일반 언더오버     (Total)             -- side: U / O
  betId=17  일반 홀짝 (SUM)  (Sum odd/even)     -- side: ODD / EVEN

Half-time variants (betId>=100) and other sports are surfaced in the parser but the
recommender skips them by default since the Dixon-Coles model is calibrated on
full-time soccer scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from betman_predictor.score_special import (
    SCORE_BINS,
    TAIL_INDEX,
    ScoreMatch,
    TeamRates,
    predict_match_grid,
)


# ---------------------------------------------------------------------------
# Mojibake repair: the live API ships strings that were latin-1 decoded from cp949.

def _fix(s: str | None) -> str:
    if not s:
        return ""
    try:
        return s.encode("latin-1").decode("cp949")
    except Exception:
        return s


# ---------------------------------------------------------------------------
# Market codes (full-time soccer only)

MARKET_1X2 = "1X2"
MARKET_HANDICAP = "AH"
MARKET_TOTAL = "OU"
MARKET_SUM_ODDEVEN = "OE"

# Mapping from Betman betId (full-time soccer) to internal market code
SOCCER_FT_MARKETS = {
    1: MARKET_1X2,
    5: MARKET_HANDICAP,
    78: MARKET_TOTAL,
    17: MARKET_SUM_ODDEVEN,
}


@dataclass(frozen=True)
class ProtoLine:
    notice_no: str               # unique identifier per line
    item_code: str               # 'SC', 'BS', 'BK'
    league_code: str
    league_name: str
    bet_id: int                  # market category
    bet_name: str                # e.g. '축구 핸디캡'
    bet_type_name: str           # e.g. '일반 승부핸디캡'
    market: str | None           # internal market code (1X2/AH/OU/OE) or None if unsupported
    home_id: str
    home_name: str
    away_id: str
    away_name: str
    game_datetime: datetime | None
    game_date_str: str
    game_key: str
    win_text: str
    win_allot: float
    draw_text: str
    draw_allot: float
    lose_text: str
    lose_allot: float
    win_handi: float             # signed handicap or O/U line, market-dependent
    draw_handi: float
    lose_handi: float
    handi_code: int


# ---------------------------------------------------------------------------
# Parsing


def _parse_dt(ms: int | float | None) -> datetime | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()


def parse_proto_round(detail: dict) -> list[ProtoLine]:
    comp = detail.get("compSchedules") or {}
    keys: list[str] = comp.get("keys") or []
    rows: list[list] = comp.get("datas") or []
    if not keys or not rows:
        return []
    idx = {k: i for i, k in enumerate(keys)}

    out: list[ProtoLine] = []
    for row in rows:
        item_code = row[idx["itemCode"]]
        bet_id = int(row[idx["betId"]] or 0)
        market = SOCCER_FT_MARKETS.get(bet_id) if item_code == "SC" else None

        out.append(ProtoLine(
            notice_no=str(row[idx["noticeNo"]] or "").strip(),
            item_code=item_code,
            league_code=str(row[idx["leagueCode"]] or "UNKNOWN"),
            league_name=_fix(row[idx["leagueName"]]),
            bet_id=bet_id,
            bet_name=_fix(row[idx["betNm"]]),
            bet_type_name=_fix(row[idx["betTypNm"]]),
            market=market,
            home_id=str(row[idx["homeId"]] or ""),
            home_name=_fix(row[idx["homeName"]]),
            away_id=str(row[idx["awayId"]] or ""),
            away_name=_fix(row[idx["awayName"]]),
            game_datetime=_parse_dt(row[idx["gameDate"]]),
            game_date_str=_fix(row[idx.get("gameDateStr", -1)]) if idx.get("gameDateStr", -1) >= 0 else "",
            game_key=_fix(row[idx["gameKey"]]),
            win_text=_fix(row[idx["winTxt"]]),
            win_allot=float(row[idx["winAllot"]] or 0.0),
            draw_text=_fix(row[idx["drawTxt"]]),
            draw_allot=float(row[idx["drawAllot"]] or 0.0),
            lose_text=_fix(row[idx["loseTxt"]]),
            lose_allot=float(row[idx["loseAllot"]] or 0.0),
            win_handi=float(row[idx["winHandi"]] or 0.0),
            draw_handi=float(row[idx["drawHandi"]] or 0.0),
            lose_handi=float(row[idx["loseHandi"]] or 0.0),
            handi_code=int(row[idx["handi"]] or 0),
        ))
    return out


# ---------------------------------------------------------------------------
# Pick model: the resolved bet on a single line.

@dataclass(frozen=True)
class Pick:
    notice_no: str
    side: str                    # H / D / A / O / U / ODD / EVEN
    line: ProtoLine

    @property
    def allot(self) -> float:
        if self.side == "H":
            return self.line.win_allot
        if self.side == "D":
            return self.line.draw_allot
        if self.side == "A":
            return self.line.lose_allot
        if self.side in ("U", "ODD"):    # winTxt holds 언더 / 홀
            return self.line.win_allot
        if self.side in ("O", "EVEN"):
            return self.line.lose_allot
        return 0.0

    @property
    def side_label(self) -> str:
        if self.line.market == MARKET_1X2:
            return {"H": self.line.win_text or "승",
                    "D": self.line.draw_text or "무",
                    "A": self.line.lose_text or "패"}[self.side]
        if self.line.market == MARKET_HANDICAP:
            sign = "+" if self.line.win_handi >= 0 else ""
            line_label = f"H {sign}{self.line.win_handi:g}"
            return {"H": f"{line_label} 승",
                    "D": f"{line_label} 무",
                    "A": f"{line_label} 패"}[self.side]
        if self.line.market == MARKET_TOTAL:
            line_label = f"{self.line.win_handi:g}"
            return {"U": f"언더 {line_label}",
                    "O": f"오버 {line_label}"}[self.side]
        if self.line.market == MARKET_SUM_ODDEVEN:
            return {"ODD": "홀", "EVEN": "짝"}[self.side]
        return self.side


# Parses '215944:H' or '215944:O' or '215944' (auto-pick best side)
def parse_pick_token(token: str) -> tuple[str, str | None]:
    """Returns (notice_no, raw_side or None)."""
    token = token.strip()
    if ":" in token:
        notice, side = token.split(":", 1)
        return notice.strip(), side.strip().upper()
    return token, None


def auto_detect_side(raw: str | None, line: ProtoLine) -> str:
    """Convert a raw user side ('H'/'홈'/'O'/'OVER'/...) into a canonical side code.

    If raw is None, return the side with the higher implied probability (lower allot).
    """
    if raw is None:
        return _highest_implied_side(line)

    r = raw.strip().upper()
    if line.market == MARKET_1X2:
        if r in ("H", "HOME", "1", "승", "WIN"):
            return "H"
        if r in ("D", "DRAW", "X", "무"):
            return "D"
        if r in ("A", "AWAY", "2", "패", "LOSS"):
            return "A"
    if line.market == MARKET_HANDICAP:
        if r in ("H", "HOME", "1", "승"):
            return "H"
        if r in ("D", "DRAW", "X", "무"):
            return "D"
        if r in ("A", "AWAY", "2", "패"):
            return "A"
    if line.market == MARKET_TOTAL:
        if r in ("U", "UNDER", "언더"):
            return "U"
        if r in ("O", "OVER", "오버"):
            return "O"
    if line.market == MARKET_SUM_ODDEVEN:
        if r in ("ODD", "O", "홀"):
            return "ODD"
        if r in ("EVEN", "E", "짝"):
            return "EVEN"
    raise ValueError(f"Cannot interpret side '{raw}' for market {line.market}")


def _highest_implied_side(line: ProtoLine) -> str:
    """Pick the side with lowest implied odd (= highest market probability)."""
    if line.market == MARKET_1X2:
        candidates = [("H", line.win_allot), ("D", line.draw_allot), ("A", line.lose_allot)]
    elif line.market == MARKET_HANDICAP:
        if line.draw_allot > 0:
            candidates = [("H", line.win_allot), ("D", line.draw_allot), ("A", line.lose_allot)]
        else:
            candidates = [("H", line.win_allot), ("A", line.lose_allot)]
    elif line.market == MARKET_TOTAL:
        candidates = [("U", line.win_allot), ("O", line.lose_allot)]
    elif line.market == MARKET_SUM_ODDEVEN:
        candidates = [("ODD", line.win_allot), ("EVEN", line.lose_allot)]
    else:
        return ""
    candidates = [(s, a) for s, a in candidates if a > 0]
    if not candidates:
        return ""
    return min(candidates, key=lambda t: t[1])[0]


# ---------------------------------------------------------------------------
# Model probabilities from a Dixon-Coles 6x6 grid.

def grid_to_market_probs(grid: list[list[float]], line: ProtoLine) -> dict[str, float]:
    """Returns a dict keyed by side (H/D/A/U/O/ODD/EVEN) with model probabilities.

    The grid is over (home_bin, away_bin) where bin TAIL_INDEX represents '5+'.
    For markets that depend on actual scores (handicap, totals, sum), we approximate
    the tail bin as exactly TAIL_INDEX -- coarse but consistent with how we trained.
    """
    out: dict[str, float] = {}
    if line.market == MARKET_1X2:
        ph = pd = pa = 0.0
        for h in range(SCORE_BINS):
            for a in range(SCORE_BINS):
                p = grid[h][a]
                if h > a: ph += p
                elif h == a: pd += p
                else: pa += p
        out["H"] = ph; out["D"] = pd; out["A"] = pa
        return out

    if line.market == MARKET_HANDICAP:
        # Handi added to HOME score. Examples:
        #   winHandi=-1.0 (line=-1, integer): home wins iff home-away >= 2;
        #     draw iff home-away == 1; away iff home-away <= 0.
        #   winHandi=-1.5 (half line): no draw possible. home iff home-away >= 2; away iff <= 1.
        h_handi = line.win_handi
        ph = pd = pa = 0.0
        # Determine integer-line behaviour: if winHandi is an integer AND drawAllot > 0
        is_integer_line = (abs(h_handi - round(h_handi)) < 1e-9) and line.draw_allot > 0
        for h in range(SCORE_BINS):
            for a in range(SCORE_BINS):
                p = grid[h][a]
                margin = (h + h_handi) - a   # home-side margin after handicap
                if is_integer_line and abs(margin) < 1e-9:
                    pd += p
                elif margin > 0:
                    ph += p
                else:
                    pa += p
        out["H"] = ph; out["D"] = pd; out["A"] = pa
        return out

    if line.market == MARKET_TOTAL:
        line_value = line.win_handi  # the totals line, e.g. 2.5
        pu = po = pp = 0.0
        is_integer_line = abs(line_value - round(line_value)) < 1e-9 and line.draw_allot > 0
        for h in range(SCORE_BINS):
            for a in range(SCORE_BINS):
                p = grid[h][a]
                total = h + a
                if is_integer_line and abs(total - line_value) < 1e-9:
                    pp += p   # push -- pure integer total markets are rare here, lump into 'D'
                elif total < line_value:
                    pu += p
                else:
                    po += p
        out["U"] = pu; out["O"] = po
        if pp > 0:
            out["D"] = pp
        return out

    if line.market == MARKET_SUM_ODDEVEN:
        po = pe = 0.0
        for h in range(SCORE_BINS):
            for a in range(SCORE_BINS):
                p = grid[h][a]
                if (h + a) % 2 == 0:
                    pe += p
                else:
                    po += p
        out["ODD"] = po; out["EVEN"] = pe
        return out

    return out


def line_match_to_score_match(line: ProtoLine) -> ScoreMatch:
    """Adapter so we can call predict_match_grid with the existing TeamRates.

    We key team ratings by *name* (not Betman's numeric homeId) so external
    football-data history can share keys -- see extract_score_matches docstring.
    """
    return ScoreMatch(
        gm_ts=0,
        match_seq=0,
        league_code=line.league_code,
        league_name=line.league_name,
        home_id=line.home_name,
        home_name=line.home_name,
        away_id=line.away_name,
        away_name=line.away_name,
        game_datetime=line.game_datetime,
        game_date_str=line.game_date_str,
        home_score=None,
        away_score=None,
    )


# ---------------------------------------------------------------------------
# Evaluation

@dataclass(frozen=True)
class LineEvaluation:
    line: ProtoLine
    side_probs: dict[str, float]
    side_allots: dict[str, float]

    def ev(self, side: str) -> float | None:
        p = self.side_probs.get(side)
        a = self.side_allots.get(side)
        if p is None or a is None or a <= 0:
            return None
        return p * a - 1.0

    def implied_prob(self, side: str) -> float | None:
        a = self.side_allots.get(side)
        return (1.0 / a) if a and a > 0 else None


def _market_implied_probs(line: ProtoLine, sides: list[str]) -> dict[str, float] | None:
    """Bookmaker implied probabilities, normalised to sum to 1 (overround removed)."""
    if line.market in (MARKET_1X2, MARKET_HANDICAP):
        raw = {"H": line.win_allot, "D": line.draw_allot, "A": line.lose_allot}
    elif line.market == MARKET_TOTAL:
        raw = {"U": line.win_allot, "O": line.lose_allot}
    elif line.market == MARKET_SUM_ODDEVEN:
        raw = {"ODD": line.win_allot, "EVEN": line.lose_allot}
    else:
        return None
    inv = {s: (1.0 / a) for s, a in raw.items() if a and a > 0}
    if not inv:
        return None
    total = sum(inv.values())
    return {s: v / total for s, v in inv.items()}


def evaluate_lines(
    lines: Iterable[ProtoLine],
    rates: TeamRates,
    public_blend: float = 0.0,
) -> list[LineEvaluation]:
    """Compute per-line side probabilities.

    public_blend in [0,1]: weight on the bookmaker-implied (de-vigged) distribution.
    0 = pure Dixon-Coles model; 1 = pure market consensus; 0.5 = even mix.
    """
    out: list[LineEvaluation] = []
    grid_cache: dict[tuple[str, str, str], list[list[float]]] = {}
    public_blend = max(0.0, min(1.0, public_blend))
    for line in lines:
        if line.market is None:
            continue
        cache_key = (line.league_code, line.home_id, line.away_id)
        grid = grid_cache.get(cache_key)
        if grid is None:
            grid = predict_match_grid(line_match_to_score_match(line), rates)
            grid_cache[cache_key] = grid
        side_probs = grid_to_market_probs(grid, line)

        if public_blend > 0.0:
            implied = _market_implied_probs(line, list(side_probs.keys()))
            if implied:
                side_probs = {
                    s: (1 - public_blend) * side_probs.get(s, 0.0)
                       + public_blend * implied.get(s, 0.0)
                    for s in side_probs
                }
                total = sum(side_probs.values()) or 1.0
                side_probs = {s: p / total for s, p in side_probs.items()}

        if line.market == MARKET_1X2 or line.market == MARKET_HANDICAP:
            allots = {"H": line.win_allot, "D": line.draw_allot, "A": line.lose_allot}
        elif line.market == MARKET_TOTAL:
            allots = {"U": line.win_allot, "O": line.lose_allot}
        else:  # SUM_ODDEVEN
            allots = {"ODD": line.win_allot, "EVEN": line.lose_allot}
        out.append(LineEvaluation(line=line, side_probs=side_probs, side_allots=allots))
    return out


@dataclass
class Recommendation:
    line: ProtoLine
    side: str
    p_model: float
    allot: float
    ev: float
    edge_pp: float               # p_model - implied_prob (in pp)


def recommend_top_ev(
    evaluations: Sequence[LineEvaluation],
    top_n: int,
    min_ev: float = 0.0,
) -> list[Recommendation]:
    out: list[Recommendation] = []
    for ev in evaluations:
        for side, allot in ev.side_allots.items():
            if allot <= 0:
                continue
            p = ev.side_probs.get(side, 0.0)
            ev_val = p * allot - 1.0
            if ev_val < min_ev:
                continue
            implied = (1.0 / allot)
            out.append(Recommendation(
                line=ev.line,
                side=side,
                p_model=p,
                allot=allot,
                ev=ev_val,
                edge_pp=(p - implied) * 100,
            ))
    out.sort(key=lambda r: r.ev, reverse=True)
    return out[:top_n]


# ---------------------------------------------------------------------------
# Pick resolution


def resolve_picks(
    picks_spec: str,
    lines: Sequence[ProtoLine],
) -> list[Pick]:
    """Parse a comma/semicolon-separated picks spec and bind each token to a ProtoLine.

    Token grammar (auto-detect):
      - '215944'           -> notice_no=215944, side auto-picked (highest implied prob)
      - '215944:H'         -> explicit side
      - '215944:오버'       -> Korean side label
    """
    by_notice: dict[str, ProtoLine] = {}
    for line in lines:
        if line.notice_no:
            by_notice[line.notice_no] = line

    out: list[Pick] = []
    for raw_token in picks_spec.replace(";", ",").split(","):
        if not raw_token.strip():
            continue
        notice, raw_side = parse_pick_token(raw_token)
        line = by_notice.get(notice)
        if line is None:
            raise ValueError(f"noticeNo {notice} not found in this round")
        if line.market is None:
            raise ValueError(f"noticeNo {notice} ({line.bet_name}) is not a supported market")
        side = auto_detect_side(raw_side, line)
        out.append(Pick(notice_no=notice, side=side, line=line))
    return out


@dataclass
class ParlayResult:
    picks: list[Pick]
    p_models: list[float]
    allots: list[float]
    joint_p: float
    parlay_allot: float
    fair_allot: float
    ev: float

    @property
    def n_legs(self) -> int:
        return len(self.picks)


def evaluate_parlay(picks: Sequence[Pick], evaluations: Sequence[LineEvaluation]) -> ParlayResult:
    by_notice = {ev.line.notice_no: ev for ev in evaluations}
    p_models: list[float] = []
    allots: list[float] = []
    for pick in picks:
        ev = by_notice.get(pick.notice_no)
        if ev is None:
            raise ValueError(f"noticeNo {pick.notice_no} not in evaluations")
        p = ev.side_probs.get(pick.side, 0.0)
        a = ev.side_allots.get(pick.side, 0.0)
        p_models.append(p)
        allots.append(a)

    joint_p = 1.0
    parlay_allot = 1.0
    for p, a in zip(p_models, allots):
        joint_p *= p
        parlay_allot *= a

    fair = (1.0 / joint_p) if joint_p > 0 else float("inf")
    ev_val = joint_p * parlay_allot - 1.0
    return ParlayResult(
        picks=list(picks),
        p_models=p_models,
        allots=allots,
        joint_p=joint_p,
        parlay_allot=parlay_allot,
        fair_allot=fair,
        ev=ev_val,
    )
