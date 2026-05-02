"""External baseball data sources used to enrich Betman match records.

Currently ships with an MLB Stats API adapter. KBO support is a planned
follow-up via statiz.sportsservice.co.kr.
"""

from betman_predictor.baseball_data.enricher import (
    BaseballEnricher,
    MatchEnrichment,
    NULL_ENRICHMENT,
)

__all__ = ["BaseballEnricher", "MatchEnrichment", "NULL_ENRICHMENT"]
