from __future__ import annotations

from betman_predictor.models import MarketDefinition


MARKETS: dict[str, MarketDefinition] = {
    "soccer": MarketDefinition(
        key="soccer",
        gm_id="G011",
        name_ko="축구 승무패",
        cli_name="soccer",
        label_map={"A": "승", "B": "무", "D": "패"},
        default_history_rounds=140,
        home_advantage=55.0,
        k_factor=26.0,
        bandwidth=95.0,
        recency_decay=0.999,
        min_same_league_samples=28,
        prior_weight=3.0,
    ),
    "baseball": MarketDefinition(
        key="baseball",
        gm_id="G024",
        name_ko="야구 승1패",
        cli_name="baseball",
        label_map={"A": "승", "B": "1", "D": "패"},
        default_history_rounds=110,
        home_advantage=28.0,
        k_factor=24.0,
        bandwidth=90.0,
        recency_decay=0.999,
        min_same_league_samples=36,
        prior_weight=3.0,
    ),
}

MARKET_ALIASES = {
    "soccer": "soccer",
    "soccer_wdl": "soccer",
    "football": "soccer",
    "g011": "soccer",
    "baseball": "baseball",
    "baseball_w1l": "baseball",
    "g024": "baseball",
}

ALL_MARKET_KEYS = tuple(MARKETS.keys())

