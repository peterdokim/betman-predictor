"""Maps Betman's two-letter MLB team codes to MLB Stats API team IDs.

Betman uses stable legacy abbreviations (e.g. MO = Montreal Expos = current
Washington Nationals, FL = Florida Marlins = current Miami Marlins). This
mapping is static — every code observed in cached G024 rounds is covered.
"""

from __future__ import annotations


BETMAN_TO_MLB_TEAM_ID: dict[str, int] = {
    "AN": 108,  # LAA Angels
    "AT": 144,  # ATL Braves
    "AZ": 109,  # ARI Diamondbacks
    "BA": 110,  # BAL Orioles
    "BO": 111,  # BOS Red Sox
    "CC": 112,  # CHC Cubs
    "CI": 113,  # CIN Reds
    "CL": 114,  # CLE Guardians
    "CO": 115,  # COL Rockies
    "CW": 145,  # CWS White Sox
    "DE": 116,  # DET Tigers
    "FL": 146,  # MIA Marlins (Florida legacy)
    "HO": 117,  # HOU Astros
    "KC": 118,  # KC Royals
    "LA": 119,  # LAD Dodgers
    "MI": 158,  # MIL Brewers
    "MN": 142,  # MIN Twins
    "MO": 120,  # WSH Nationals (Montreal legacy)
    "NM": 121,  # NYM Mets
    "NY": 147,  # NYY Yankees
    "OA": 133,  # OAK Athletics
    "PH": 143,  # PHI Phillies
    "PI": 134,  # PIT Pirates
    "SD": 135,  # SD Padres
    "SE": 136,  # SEA Mariners
    "SF": 137,  # SF Giants
    "SL": 138,  # STL Cardinals
    "TB": 139,  # TB Rays
    "TE": 140,  # TEX Rangers
    "TO": 141,  # TOR Blue Jays
}


def betman_to_mlb_id(betman_team_id: str | None) -> int | None:
    if not betman_team_id:
        return None
    return BETMAN_TO_MLB_TEAM_ID.get(betman_team_id.strip().upper())


# Three-year average park factors (100 = neutral, >100 = hitter-friendly,
# <100 = pitcher-friendly), keyed by MLB Stats API home team ID. Drawn from
# Statcast/Baseball Reference 2022-2024 averages. Static enough that maintaining
# it by hand is cheaper than scraping.
PARK_FACTOR_BY_HOME_TEAM_ID: dict[int, int] = {
    108: 99,   # LAA — Angel Stadium
    109: 98,   # ARI — Chase Field
    110: 102,  # BAL — Camden Yards
    111: 105,  # BOS — Fenway Park
    112: 103,  # CHC — Wrigley Field
    113: 109,  # CIN — Great American Ball Park
    114: 100,  # CLE — Progressive Field
    115: 117,  # COL — Coors Field
    116: 102,  # DET — Comerica Park
    117: 100,  # HOU — Minute Maid Park
    118: 100,  # KC  — Kauffman Stadium
    119: 96,   # LAD — Dodger Stadium
    120: 99,   # WSH — Nationals Park
    121: 97,   # NYM — Citi Field
    133: 97,   # OAK — Sutter Health Park (relocated to Sacramento)
    134: 100,  # PIT — PNC Park
    135: 93,   # SD  — Petco Park
    136: 95,   # SEA — T-Mobile Park
    137: 94,   # SF  — Oracle Park
    138: 100,  # STL — Busch Stadium
    139: 97,   # TB  — Steinbrenner Field (Trop replacement)
    140: 104,  # TEX — Globe Life Field
    141: 100,  # TOR — Rogers Centre
    142: 99,   # MIN — Target Field
    143: 103,  # PHI — Citizens Bank Park
    144: 100,  # ATL — Truist Park
    145: 97,   # CWS — Guaranteed Rate Field
    146: 95,   # MIA — loanDepot Park
    147: 103,  # NYY — Yankee Stadium
    158: 100,  # MIL — American Family Field
}


def park_factor_for_home_team(home_team_id: int | None) -> int:
    """Park factor (100 = neutral). Falls back to 100 for unknown venues."""
    if home_team_id is None:
        return 100
    return PARK_FACTOR_BY_HOME_TEAM_ID.get(home_team_id, 100)
