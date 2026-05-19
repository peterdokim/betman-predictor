"""External football-data.co.uk CSV ingestion for proto-style soccer markets.

Why: the G016 Toto cache only ships ~80 rounds × 3 matches = 240 matches, scattered
across whichever 3 EPL fixtures Sports Toto picked that week. Most teams in the live
proto round (G101) aren't in that set, so the Dixon-Coles model falls back to league
averages and produces wildly miscalibrated probabilities.

football-data.co.uk has free CSVs of every top-5 European league dating back decades.
Five seasons × eight leagues ≈ 14,000 matches with full-time scores -- enough to put
attack/defence ratings on every team that shows up in proto rounds.

The trick is name reconciliation. Betman uses 한국어 transliterations
("바이에른뮌헨", "맨체스U") while football-data uses native English ("Bayern Munich",
"Man United"). We solve this with an explicit mapping table for the leagues we care
about. Anything not in the map keeps its football-data name and gets matched in
fit_team_rates by the league prior only -- which is still better than nothing.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

from betman_predictor.score_special import ScoreMatch, TeamRates, fit_team_rates


# ---------------------------------------------------------------------------
# League code mapping: football-data Div -> our internal league_code
# (must match what we see in betman payloads so attack/defence ratings transfer)

# Betman leagueCode values, observed in live G101/G016 payloads.
LEAGUE_CODE_MAP = {
    "E0": "52",          # 잉글랜드 프리미어리그
    "E1": "2",           # 잉글랜드 챔피언십
    "D1": "56",          # 독일 분데스리가
    "I1": "53",          # 이탈리아 세리에 A
    "SP1": "54",         # 스페인 라리가
    "F1": "67",          # 프랑스 리그 1
    "N1": "N1_FD",       # 네덜란드 — Betman 코드 미확인, 학습은 되나 라이브 매칭 X
    "P1": "P1_FD",       # 포르투갈 — Betman 코드 미확인
}


# ---------------------------------------------------------------------------
# Team-name mapping: football-data spelling -> Betman 한글 spelling
#
# Betman 팀명은 round-detail의 homeName/awayName 필드에서 가져옴. 일치하지 않으면
# attack/defence가 별개 팀으로 학습되어 데이터가 분산됨. 핵심 규칙:
#   - 베트맨이 짧게 자름: "맨체스U" (5자 한도). 매핑은 *베트맨 표기*로 맞춰야 함
#   - 같은 팀이라도 시즌마다 베트맨 표기가 다를 수 있음 (드물지만 가능)
#   - 매핑되지 않으면 키 그대로 두면 됨 — 학습은 되되 베트맨 라이브와 매칭 안됨
#
# football-data Div 별로 분리해서 수동 작성. 누락은 fuzzy fallback이 처리.

TEAM_NAME_MAP_BY_LEAGUE: dict[str, dict[str, str]] = {
    "E0": {  # EPL (베트맨 표기는 round 260017/260051에서 직접 추출)
        "Arsenal": "아스널",
        "Aston Villa": "A빌라",
        "Bournemouth": "본머스",
        "Brentford": "브렌트퍼",
        "Brighton": "브라이튼",
        "Burnley": "번리",
        "Chelsea": "첼시",
        "Crystal Palace": "크리스털",
        "Everton": "에버튼",
        "Fulham": "풀럼",
        "Leeds": "리즈",
        "Liverpool": "리버풀",
        "Man City": "맨체스C",
        "Man United": "맨체스U",
        "Newcastle": "뉴캐슬",
        "Nott'm Forest": "노팅엄",
        "Sunderland": "선덜랜드",
        "Tottenham": "토트넘",
        "West Ham": "웨스트햄",
        "Wolves": "울버햄튼",
        # historical promoted/relegated
        "Leicester": "레스터",
        "Southampton": "사우샘튼",
        "Ipswich": "입스위치",
        "Sheffield United": "셰필드U",
        "Luton": "루턴",
    },
    "E1": {  # Championship
        "Birmingham": "버밍엄",
        "Blackburn": "블랙번",
        "Bristol City": "브리스틀C",
        "Burnley": "번리",
        "Cardiff": "카디프",
        "Charlton": "찰턴",
        "Coventry": "코번트리",
        "Derby": "더비",
        "Hull": "헐",
        "Ipswich": "입스위치",
        "Leeds": "리즈",
        "Leicester": "레스터",
        "Luton": "루턴",
        "Middlesbrough": "미들즈브러",
        "Millwall": "밀월",
        "Norwich": "노리치",
        "Oxford": "옥스퍼드",
        "Plymouth": "플리머스",
        "Portsmouth": "포츠머스",
        "Preston": "프레스턴",
        "QPR": "QPR",
        "Sheffield United": "셰필드U",
        "Sheffield Weds": "셰필드W",
        "Stoke": "스토크",
        "Sunderland": "선덜랜드",
        "Swansea": "스완지",
        "Watford": "왓퍼드",
        "West Brom": "웨스트브롬",
        "Wrexham": "렉섬",
    },
    "D1": {  # Bundesliga
        "Augsburg": "아우크스부르크",
        "Bayern Munich": "바이에른뮌헨",
        "Bochum": "보훔",
        "Dortmund": "도르트문트",
        "Ein Frankfurt": "프랑크푸르트",
        "Freiburg": "프라이부르크",
        "Heidenheim": "하이덴하임",
        "Hoffenheim": "호펜하임",
        "Holstein Kiel": "홀슈타인킬",
        "Leverkusen": "레버쿠젠",
        "M'gladbach": "묀헨글라드바흐",
        "Mainz": "마인츠05",
        "RB Leipzig": "라이프치히",
        "St Pauli": "장크트파울리",
        "Stuttgart": "슈투트가르트",
        "Union Berlin": "우니온베를린",
        "Werder Bremen": "브레멘",
        "Wolfsburg": "볼프스부르크",
        "FC Koln": "쾰른",
        "Hamburg": "함부르크",
    },
    "I1": {  # Serie A
        "Atalanta": "아탈란타",
        "Bologna": "볼로냐",
        "Cagliari": "칼리아리",
        "Como": "코모",
        "Cremonese": "크레모네세",
        "Empoli": "엠폴리",
        "Fiorentina": "피오렌티나",
        "Frosinone": "프로시노네",
        "Genoa": "제노아",
        "Hellas Verona": "엘라스베로나",
        "Inter": "인테르나치오날레밀라노",
        "Juventus": "유벤투스",
        "Lazio": "라치오",
        "Lecce": "레체",
        "Milan": "AC밀란",
        "Monza": "몬차",
        "Napoli": "나폴리",
        "Parma": "파르마",
        "Pisa": "피사",
        "Roma": "로마",
        "Salernitana": "살레르니타나",
        "Sassuolo": "사수올로",
        "Torino": "토리노",
        "Udinese": "우디네세",
        "Venezia": "베네치아",
    },
    "SP1": {  # La Liga
        "Alaves": "알라베스",
        "Almeria": "알메리아",
        "Ath Bilbao": "빌바오",
        "Ath Madrid": "AT마드리드",
        "Barcelona": "바르셀로나",
        "Betis": "베티스",
        "Cadiz": "카디스",
        "Celta": "셀타비고",
        "Elche": "엘체",
        "Espanol": "에스파뇰",
        "Getafe": "헤타페",
        "Girona": "지로나",
        "Granada": "그라나다",
        "Las Palmas": "라스팔마스",
        "Leganes": "레가네스",
        "Levante": "레반테",
        "Mallorca": "마요르카",
        "Osasuna": "오사수나",
        "Oviedo": "오비에도",
        "Rayo Vallecano": "라요바예카노",
        "Real Madrid": "R마드리드",
        "Sevilla": "세비야",
        "Sociedad": "R소시에다드",
        "Valencia": "발렌시아",
        "Valladolid": "바야돌리드",
        "Villarreal": "비야레알",
    },
    "F1": {  # Ligue 1
        "Angers": "앙제",
        "Auxerre": "오세르",
        "Brest": "브레스트",
        "Clermont": "클레르몽",
        "Le Havre": "르아브르",
        "Lens": "랑스",
        "Lille": "릴",
        "Lorient": "로리앙",
        "Lyon": "리옹",
        "Marseille": "마르세유",
        "Metz": "메스",
        "Monaco": "AS모나코",
        "Montpellier": "몽펠리에",
        "Nantes": "낭트",
        "Nice": "니스",
        "Paris SG": "PSG",
        "Reims": "랭스",
        "Rennes": "렌",
        "St Etienne": "생테티엔",
        "Strasbourg": "스트라스부르",
        "Toulouse": "툴루즈",
    },
    "N1": {  # Eredivisie -- short list for now
        "Ajax": "아약스",
        "AZ Alkmaar": "AZ",
        "Feyenoord": "페예노르트",
        "PSV Eindhoven": "PSV",
        "Twente": "트벤테",
        "Utrecht": "위트레흐트",
    },
    "P1": {  # Primeira Liga
        "Benfica": "벤피카",
        "Braga": "브라가",
        "Porto": "FC포르투",
        "Sp Lisbon": "S리스본",
    },
}


# ---------------------------------------------------------------------------
# Date parsing -- football-data uses dd/mm/yyyy or dd/mm/yy

def _parse_date(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# CSV → ScoreMatch records

def parse_csv(path: Path) -> list[ScoreMatch]:
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []

    div = rows[0].get("Div") or path.stem.split("_", 1)[-1]
    league_code = LEAGUE_CODE_MAP.get(div, f"FD_{div}")
    name_map = TEAM_NAME_MAP_BY_LEAGUE.get(div, {})

    out: list[ScoreMatch] = []
    for r in rows:
        if not r.get("HomeTeam") or not r.get("AwayTeam"):
            continue
        try:
            fthg = int(r["FTHG"])
            ftag = int(r["FTAG"])
        except (KeyError, ValueError, TypeError):
            continue
        home_raw = r["HomeTeam"].strip()
        away_raw = r["AwayTeam"].strip()
        home_name = name_map.get(home_raw, home_raw)
        away_name = name_map.get(away_raw, away_raw)
        # Use the canonical (Korean) name as both id and display name -- this is what
        # fit_team_rates keys on, and what proto.py looks up via line.home_id.
        # But Betman's line.home_id is a numeric internal ID, NOT a name. So we cannot
        # match by id; we patch fit/predict to *also* try name-based lookup (see below).
        dt = _parse_date(r.get("Date") or "")
        out.append(ScoreMatch(
            gm_ts=int(dt.strftime("%y%m%d")) if dt else 0,
            match_seq=0,
            league_code=league_code,
            league_name=div,
            home_id=home_name,           # use name as id so live lookups by name work
            home_name=home_name,
            away_id=away_name,
            away_name=away_name,
            game_datetime=dt,
            game_date_str=dt.strftime("%y.%m.%d") if dt else "",
            home_score=fthg,
            away_score=ftag,
        ))
    return out


def load_external_history(directory: Path) -> list[ScoreMatch]:
    out: list[ScoreMatch] = []
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.csv")):
        out.extend(parse_csv(path))
    return out


# ---------------------------------------------------------------------------
# Elo pre-seed: convert Dixon-Coles attack/defence ratios into Elo offsets.
#
# attack > 1 ⇒ scores more than league average (good).
# defence > 1 ⇒ concedes more than league average (BAD).
# Strength = log(attack) - log(defence). A strength of ~0.5 corresponds roughly
# to top-vs-bottom of a top-5 league. We map that to ~150 Elo, so scale = 300.
# This is intentionally conservative — Elo updates during fit() will refine.

ELO_PRESEED_SCALE = 300.0


def attack_defence_to_elo_offsets(rates: TeamRates) -> dict[str, dict[str, float]]:
    """Return {league_code: {team_name: elo_offset_from_1500}}.

    Team keys are reconstructed from `f"{league_code}:{team_name}"` which is the
    convention `score_special.fit_team_rates` uses (because external_data.parse_csv
    sets home_id == home_name == Korean Betman name).
    """
    out: dict[str, dict[str, float]] = {}
    for compound_key, attack in rates.attack.items():
        if ":" not in compound_key:
            continue
        league_code, team_name = compound_key.split(":", 1)
        defence = rates.defence.get(compound_key, 1.0)
        # Soft floor avoids log of near-zero ratios for tiny-sample teams.
        atk = max(float(attack), 0.1)
        df = max(float(defence), 0.1)
        strength = math.log(atk) - math.log(df)
        out.setdefault(league_code, {})[team_name] = strength * ELO_PRESEED_SCALE
    return out


def build_preseed(directory: Path) -> dict[str, dict[str, float]]:
    """Convenience: load CSV directory, fit Dixon-Coles rates, return Elo preseed."""
    history = load_external_history(directory)
    if not history:
        return {}
    rates = fit_team_rates(history)
    return attack_defence_to_elo_offsets(rates)
