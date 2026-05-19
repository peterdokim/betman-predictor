"""Machine-learning predictor for Betman 3-way markets.

Inspired by ProphitBet's rolling team-form features (HW/AW/HL/AL, lifetime win
rates) and standard-scaling + linear/tree classifiers. Adapted to Betman, where
goal counts are not exposed but the public vote distribution is preserved
historically and acts as a strong consensus feature.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from betman_predictor.lineup_adjust import LineupAdjuster, NULL_PENALTY
from betman_predictor.models import MarketDefinition, MatchRecord, Prediction


FALLBACK_DT = datetime(1970, 1, 1, tzinfo=timezone.utc)
DEFAULT_FORM_WINDOW = 8
DEFAULT_ELO = 1500.0
SCORE_MAP = {"A": 1.0, "B": 0.5, "D": 0.0}

VoteLookup = dict[tuple[int, int], dict[str, float]]

DEFAULT_STARTER_ERA = 4.20
DEFAULT_STARTER_WHIP = 1.30
DEFAULT_STARTER_K9 = 8.5
DEFAULT_PYTHAG = 0.5
DEFAULT_RPG = 4.5
DEFAULT_TEAM_OPS = 0.720
DEFAULT_TEAM_OBP = 0.320
DEFAULT_TEAM_SLG = 0.400
DEFAULT_PARK_FACTOR = 100.0


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class _TeamState:
    elo: float = DEFAULT_ELO
    last_match_dt: Optional[datetime] = None
    overall: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_FORM_WINDOW))
    home_only: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_FORM_WINDOW))
    away_only: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_FORM_WINDOW))
    total_played: int = 0
    total_wins: int = 0
    total_draws: int = 0
    total_losses: int = 0


class FeatureBuilder:
    """Streaming feature builder. Call features_for(...) BEFORE update(...) for
    each match in chronological order so features reflect prior matches only."""

    FEATURE_NAMES = (
        "elo_gap",
        "home_elo_centered",
        "away_elo_centered",
        "home_form_overall_w",
        "home_form_overall_d",
        "home_form_overall_l",
        "home_form_home_w",
        "home_form_home_d",
        "home_form_home_l",
        "away_form_overall_w",
        "away_form_overall_d",
        "away_form_overall_l",
        "away_form_away_w",
        "away_form_away_d",
        "away_form_away_l",
        "home_lifetime_w",
        "home_lifetime_d",
        "home_lifetime_l",
        "away_lifetime_w",
        "away_lifetime_d",
        "away_lifetime_l",
        "home_played_log",
        "away_played_log",
        "home_rest_days",
        "away_rest_days",
        "vote_share_a",
        "vote_share_b",
        "vote_share_d",
        "vote_total_log",
        "vote_present",
        "domestic",
        "league_frequency_log",
        # MLB enrichment block — defaults + present-mask when not available.
        # Only the 15 features that backtest favorably are fed to the model.
        # The richer fields (OPS, l30 ERA, park factor) are still extracted
        # by the enricher for display in the report — they just don't move
        # accuracy on a 1300-match training set, so we don't add them as
        # training features.
        "home_starter_era",
        "away_starter_era",
        "starter_era_diff",
        "home_starter_whip",
        "away_starter_whip",
        "home_starter_k9",
        "away_starter_k9",
        "home_pythag",
        "away_pythag",
        "pythag_diff",
        "home_rpg",
        "away_rpg",
        "home_rapg",
        "away_rapg",
        "enrichment_present",
    )

    def __init__(
        self,
        market: MarketDefinition,
        form_window: int = DEFAULT_FORM_WINDOW,
        enricher: Optional[object] = None,
    ) -> None:
        self.market = market
        self.form_window = form_window
        self.team_state: dict[str, _TeamState] = {}
        self.league_counts: dict[str, int] = defaultdict(int)
        self.enricher = enricher

    def _team_key(self, league_code: str, team_id: str) -> str:
        return f"{league_code}:{team_id}"

    def _ensure_team(self, key: str) -> _TeamState:
        state = self.team_state.get(key)
        if state is None:
            state = _TeamState(
                overall=deque(maxlen=self.form_window),
                home_only=deque(maxlen=self.form_window),
                away_only=deque(maxlen=self.form_window),
            )
            self.team_state[key] = state
        return state

    @staticmethod
    def _form_rates(window: deque) -> tuple[float, float, float]:
        if not window:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        n = len(window)
        w = sum(1 for r in window if r == "W")
        d = sum(1 for r in window if r == "D")
        l = sum(1 for r in window if r == "L")
        return w / n, d / n, l / n

    @staticmethod
    def _lifetime_rates(state: _TeamState) -> tuple[float, float, float]:
        if state.total_played <= 0:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        return (
            state.total_wins / state.total_played,
            state.total_draws / state.total_played,
            state.total_losses / state.total_played,
        )

    def features_for(
        self,
        match: MatchRecord,
        votes: Optional[dict[str, float]] = None,
    ) -> list[float]:
        home_state = self._ensure_team(self._team_key(match.league_code, match.home_id))
        away_state = self._ensure_team(self._team_key(match.league_code, match.away_id))

        ha = self.market.param_for(match.league_code, "home_advantage")
        elo_gap = home_state.elo + ha - away_state.elo
        home_elo_c = home_state.elo - DEFAULT_ELO
        away_elo_c = away_state.elo - DEFAULT_ELO

        h_o_w, h_o_d, h_o_l = self._form_rates(home_state.overall)
        h_h_w, h_h_d, h_h_l = self._form_rates(home_state.home_only)
        a_o_w, a_o_d, a_o_l = self._form_rates(away_state.overall)
        a_a_w, a_a_d, a_a_l = self._form_rates(away_state.away_only)

        h_lt_w, h_lt_d, h_lt_l = self._lifetime_rates(home_state)
        a_lt_w, a_lt_d, a_lt_l = self._lifetime_rates(away_state)

        match_dt = match.game_datetime or FALLBACK_DT
        h_rest = self._rest_days(home_state.last_match_dt, match_dt)
        a_rest = self._rest_days(away_state.last_match_dt, match_dt)

        if votes:
            va = float(votes.get("A", 1.0 / 3.0))
            vb = float(votes.get("B", 1.0 / 3.0))
            vd = float(votes.get("D", 1.0 / 3.0))
            vtotal = float(votes.get("_total", 0.0))
            vpresent = 1.0
        else:
            va = vb = vd = 1.0 / 3.0
            vtotal = 0.0
            vpresent = 0.0

        league_freq_log = math.log1p(float(self.league_counts.get(match.league_code, 0)))
        enrichment_block = self._enrichment_block(match)

        return [
            elo_gap,
            home_elo_c,
            away_elo_c,
            h_o_w, h_o_d, h_o_l,
            h_h_w, h_h_d, h_h_l,
            a_o_w, a_o_d, a_o_l,
            a_a_w, a_a_d, a_a_l,
            h_lt_w, h_lt_d, h_lt_l,
            a_lt_w, a_lt_d, a_lt_l,
            math.log1p(float(home_state.total_played)),
            math.log1p(float(away_state.total_played)),
            h_rest,
            a_rest,
            va, vb, vd,
            math.log1p(vtotal),
            vpresent,
            1.0 if match.domestic else 0.0,
            league_freq_log,
            *enrichment_block,
        ]

    def _enrichment_block(self, match: MatchRecord) -> list[float]:
        if self.enricher is None:
            return _enrichment_defaults(present=False)
        try:
            data = self.enricher.lookup(match)
        except Exception:
            return _enrichment_defaults(present=False)
        if data is None:
            return _enrichment_defaults(present=False)

        h_era = data.home_starter_era if data.home_starter_era is not None else DEFAULT_STARTER_ERA
        a_era = data.away_starter_era if data.away_starter_era is not None else DEFAULT_STARTER_ERA
        h_whip = data.home_starter_whip if data.home_starter_whip is not None else DEFAULT_STARTER_WHIP
        a_whip = data.away_starter_whip if data.away_starter_whip is not None else DEFAULT_STARTER_WHIP
        h_k9 = data.home_starter_k9 if data.home_starter_k9 is not None else DEFAULT_STARTER_K9
        a_k9 = data.away_starter_k9 if data.away_starter_k9 is not None else DEFAULT_STARTER_K9
        h_py = data.home_recent_pythag if data.home_recent_pythag is not None else DEFAULT_PYTHAG
        a_py = data.away_recent_pythag if data.away_recent_pythag is not None else DEFAULT_PYTHAG
        h_rpg = data.home_recent_runs_per_game if data.home_recent_runs_per_game is not None else DEFAULT_RPG
        a_rpg = data.away_recent_runs_per_game if data.away_recent_runs_per_game is not None else DEFAULT_RPG
        h_rapg = data.home_recent_runs_allowed_per_game if data.home_recent_runs_allowed_per_game is not None else DEFAULT_RPG
        a_rapg = data.away_recent_runs_allowed_per_game if data.away_recent_runs_allowed_per_game is not None else DEFAULT_RPG

        any_real = any(
            v is not None
            for v in (
                data.home_starter_era,
                data.away_starter_era,
                data.home_recent_pythag,
                data.away_recent_pythag,
            )
        )
        return [
            float(h_era),
            float(a_era),
            float(h_era - a_era),
            float(h_whip),
            float(a_whip),
            float(h_k9),
            float(a_k9),
            float(h_py),
            float(a_py),
            float(h_py - a_py),
            float(h_rpg),
            float(a_rpg),
            float(h_rapg),
            float(a_rapg),
            1.0 if any_real else 0.0,
        ]

    @staticmethod
    def _rest_days(last_dt: Optional[datetime], current_dt: datetime) -> float:
        if last_dt is None:
            return 14.0
        delta = (current_dt - last_dt).total_seconds() / 86400.0
        return float(min(max(delta, 0.0), 30.0))

    def update(self, match: MatchRecord) -> None:
        if match.result_code not in self.market.result_codes:
            return

        home_state = self._ensure_team(self._team_key(match.league_code, match.home_id))
        away_state = self._ensure_team(self._team_key(match.league_code, match.away_id))

        ha = self.market.param_for(match.league_code, "home_advantage")
        k = self.market.param_for(match.league_code, "k_factor")
        diff = home_state.elo + ha - away_state.elo
        expected_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        actual_home = SCORE_MAP[match.result_code]
        delta = k * (actual_home - expected_home)
        home_state.elo += delta
        away_state.elo -= delta

        if match.result_code == "A":
            home_tag, away_tag = "W", "L"
            home_state.total_wins += 1
            away_state.total_losses += 1
        elif match.result_code == "D":
            home_tag, away_tag = "L", "W"
            home_state.total_losses += 1
            away_state.total_wins += 1
        else:
            home_tag = away_tag = "D"
            home_state.total_draws += 1
            away_state.total_draws += 1

        home_state.overall.append(home_tag)
        home_state.home_only.append(home_tag)
        away_state.overall.append(away_tag)
        away_state.away_only.append(away_tag)
        home_state.total_played += 1
        away_state.total_played += 1

        match_dt = match.game_datetime or FALLBACK_DT
        home_state.last_match_dt = match_dt
        away_state.last_match_dt = match_dt
        self.league_counts[match.league_code] += 1


@dataclass(frozen=True)
class MLConfig:
    algorithm: str = "logreg"  # logreg | gbm
    form_window: int = DEFAULT_FORM_WINDOW
    blend_with_priors: float = 0.05  # mix toward training-set base rates
    calibrate: bool = True            # isotonic per-class calibration on a holdout fold
    calibration_holdout: float = 0.15  # last X% of training (chronologically) used for fit
    min_samples_for_calibration: int = 200


OddsLookup = dict[tuple[int, int], dict[str, float]]


class MLPredictor:
    """ProphitBet-inspired ML predictor for Betman 3-way markets."""

    def __init__(
        self,
        market: MarketDefinition,
        config: Optional[MLConfig] = None,
        enricher: Optional[object] = None,
        lineup_adjuster: Optional[LineupAdjuster] = None,
        preseed_by_name: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        if not sklearn_available():
            raise RuntimeError(
                "scikit-learn and numpy are required for the ML predictor. "
                "Install via `python -m pip install scikit-learn numpy`."
            )
        self.market = market
        self.config = config or MLConfig()
        self.feature_builder = FeatureBuilder(
            market, form_window=self.config.form_window, enricher=enricher
        )
        self.pipeline = None
        self.classes_: tuple[str, ...] = ()
        self.priors_: dict[str, float] = {}
        self.training_size = 0
        self.fitted = False
        self.lineup_adjuster = lineup_adjuster
        self.preseed_by_name = preseed_by_name or {}
        self.calibrators_: dict[str, object] | None = None

    def _bootstrap_preseed(self, matches: Iterable[MatchRecord]) -> None:
        if not self.preseed_by_name:
            return
        for match in matches:
            for league, tid, tname in (
                (match.league_code, match.home_id, match.home_name),
                (match.league_code, match.away_id, match.away_name),
            ):
                key = self.feature_builder._team_key(league, tid)
                if key in self.feature_builder.team_state:
                    continue
                offset = self.preseed_by_name.get(league, {}).get(tname)
                if offset is not None:
                    state = self.feature_builder._ensure_team(key)
                    state.elo = DEFAULT_ELO + float(offset)

    def _build_pipeline(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.config.algorithm == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.06,
                max_iter=400,
                l2_regularization=1.0,
                early_stopping=False,
            )
            return Pipeline([("clf", model)])

        # solver=lbfgs handles multinomial automatically in modern sklearn;
        # avoid the multi_class kwarg, which is deprecated in sklearn>=1.5.
        model = LogisticRegression(solver="lbfgs", C=0.6, max_iter=2000)
        return Pipeline([("scale", StandardScaler()), ("clf", model)])

    def fit(
        self,
        matches: Iterable[MatchRecord],
        vote_lookup: Optional[VoteLookup] = None,
    ) -> None:
        import numpy as np

        ordered = sorted(
            (m for m in matches if m.result_code in self.market.result_codes),
            key=lambda m: (m.game_datetime or FALLBACK_DT, m.gm_ts, m.match_seq),
        )

        self._bootstrap_preseed(ordered)

        X_rows: list[list[float]] = []
        y: list[str] = []
        prior_counts: dict[str, int] = defaultdict(int)
        vote_lookup = vote_lookup or {}

        for match in ordered:
            votes = vote_lookup.get((match.gm_ts, match.match_seq))
            X_rows.append(self.feature_builder.features_for(match, votes=votes))
            y.append(match.result_code or "")
            prior_counts[match.result_code or ""] += 1
            self.feature_builder.update(match)

        if not X_rows:
            raise RuntimeError("No labeled training matches available for ML predictor.")

        X = np.asarray(X_rows, dtype=np.float64)
        y_arr = np.asarray(y)

        total = sum(prior_counts.values())
        self.priors_ = {code: prior_counts.get(code, 0) / total for code in self.market.ordered_codes}
        self.training_size = total

        calibrate = (
            self.config.calibrate
            and total >= self.config.min_samples_for_calibration
            and 0.0 < self.config.calibration_holdout < 0.5
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if calibrate:
                # Train base on chronological prefix, fit per-class isotonic on the held-out tail,
                # then refit base on the full data so the production model uses every sample.
                from sklearn.isotonic import IsotonicRegression

                split = int(round(total * (1.0 - self.config.calibration_holdout)))
                split = max(1, min(total - 1, split))
                X_train, X_cal = X[:split], X[split:]
                y_train, y_cal = y_arr[:split], y_arr[split:]

                base = self._build_pipeline()
                # Guard: the holdout tail might miss a class; fall back to no calibration.
                base.fit(X_train, y_train)
                if len(set(y_cal.tolist())) < 2:
                    calibrate = False

                if calibrate:
                    classes = list(base.named_steps["clf"].classes_)
                    proba_cal = base.predict_proba(X_cal)
                    self.calibrators_ = {}
                    for cls_idx, cls in enumerate(classes):
                        targets = (y_cal == cls).astype(np.float64)
                        if targets.sum() == 0 or targets.sum() == len(targets):
                            continue
                        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        iso.fit(proba_cal[:, cls_idx], targets)
                        self.calibrators_[cls] = iso

                # Final model trained on everything.
                self.pipeline = self._build_pipeline()
                self.pipeline.fit(X, y_arr)
            else:
                self.calibrators_ = None
                self.pipeline = self._build_pipeline()
                self.pipeline.fit(X, y_arr)

        self.classes_ = tuple(self.pipeline.named_steps["clf"].classes_)
        self.fitted = True

    def predict_round(
        self,
        matches: Iterable[MatchRecord],
        vote_lookup: Optional[VoteLookup] = None,
        odds_lookup: Optional[OddsLookup] = None,
        odds_weight: float = 0.0,
    ) -> list[Prediction]:
        if not self.fitted or self.pipeline is None:
            raise RuntimeError("MLPredictor must be fitted before predicting.")

        import numpy as np

        vote_lookup = vote_lookup or {}
        odds_lookup = odds_lookup or {}
        odds_weight = max(0.0, min(1.0, float(odds_weight)))
        match_list = list(matches)
        self._bootstrap_preseed(match_list)
        ordered = sorted(match_list, key=lambda m: (m.match_seq, m.home_name, m.away_name))

        rows: list[list[float]] = []
        for match in ordered:
            votes = vote_lookup.get((match.gm_ts, match.match_seq))
            rows.append(self.feature_builder.features_for(match, votes=votes))

        if not rows:
            return []

        X = np.asarray(rows, dtype=np.float64)
        proba = self.pipeline.predict_proba(X)

        # Per-class isotonic calibration (then renormalise per-row to keep simplex).
        if self.calibrators_:
            adjusted = proba.copy()
            for cls_idx, cls in enumerate(self.classes_):
                iso = self.calibrators_.get(cls)
                if iso is not None:
                    adjusted[:, cls_idx] = iso.transform(adjusted[:, cls_idx])
            row_sums = adjusted.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            proba = adjusted / row_sums

        predictions: list[Prediction] = []
        blend = self.config.blend_with_priors
        for match, row_proba in zip(ordered, proba):
            probabilities = {code: 0.0 for code in self.market.ordered_codes}
            for cls_index, cls in enumerate(self.classes_):
                if cls in probabilities:
                    probabilities[cls] = float(row_proba[cls_index])
            if blend > 0.0:
                for code in probabilities:
                    probabilities[code] = (1.0 - blend) * probabilities[code] + blend * self.priors_.get(code, 0.0)
                total = sum(probabilities.values()) or 1.0
                probabilities = {code: prob / total for code, prob in probabilities.items()}

            scope_tags: list[str] = [f"ml-{self.config.algorithm}-n{self.training_size}"]
            if self.calibrators_:
                scope_tags.append("iso")

            # Closing-odds geometric blend (predict-time only).
            if odds_weight > 0.0:
                odds_row = odds_lookup.get((match.gm_ts, match.match_seq))
                if odds_row:
                    merged: dict[str, float] = {}
                    for code in self.market.ordered_codes:
                        p_model = max(probabilities.get(code, 0.0), 1e-9)
                        p_odds = max(float(odds_row.get(code, 0.0)), 1e-9)
                        merged[code] = (p_model ** (1.0 - odds_weight)) * (p_odds ** odds_weight)
                    total = sum(merged.values()) or 1.0
                    probabilities = {code: prob / total for code, prob in merged.items()}
                    scope_tags.append(f"odds{odds_weight:.2f}")

            home_state = self.feature_builder.team_state.get(
                f"{match.league_code}:{match.home_id}", _TeamState()
            )
            away_state = self.feature_builder.team_state.get(
                f"{match.league_code}:{match.away_id}", _TeamState()
            )
            home_elo = home_state.elo
            away_elo = away_state.elo
            ha = self.market.param_for(match.league_code, "home_advantage")

            # Lineup-aware predict-time penalty: shift probabilities toward the
            # team that has its key players, by decreasing the penalised side's
            # implied probabilities through a logistic shift on the Elo gap.
            if self.lineup_adjuster is not None:
                penalty = self.lineup_adjuster.penalty_for(
                    gm_ts=match.gm_ts,
                    match_seq=match.match_seq,
                    league_code=match.league_code,
                    home_team=match.home_name,
                    away_team=match.away_name,
                )
                if not penalty.is_empty:
                    home_elo -= penalty.home_penalty
                    away_elo -= penalty.away_penalty
                    # Apply the gap shift as a softmax tilt on the 3-way distribution.
                    # The shift is the *change* in expected home-score driven by the
                    # net Elo penalty (away_penalty - home_penalty).
                    net_shift = penalty.away_penalty - penalty.home_penalty
                    if abs(net_shift) > 1e-6:
                        new_diff = home_elo + ha - away_elo
                        old_diff = new_diff - net_shift
                        old_e = 1.0 / (1.0 + 10 ** (-old_diff / 400.0))
                        new_e = 1.0 / (1.0 + 10 ** (-new_diff / 400.0))
                        # Move proba mass between A (home win) and D (away win)
                        # in proportion to the change in expected score; leave B (draw)
                        # untouched, then renormalise.
                        delta = new_e - old_e
                        probabilities["A"] = max(probabilities.get("A", 0.0) + delta, 1e-6)
                        probabilities["D"] = max(probabilities.get("D", 0.0) - delta, 1e-6)
                        total = sum(probabilities.values()) or 1.0
                        probabilities = {code: prob / total for code, prob in probabilities.items()}
                        scope_tags.append("lineup")

            rating_gap = home_elo + ha - away_elo
            pick_code = max(probabilities, key=probabilities.get)
            predictions.append(
                Prediction(
                    match=match,
                    pick_code=pick_code,
                    pick_label=self.market.label_map[pick_code],
                    probabilities=probabilities,
                    home_rating=home_elo,
                    away_rating=away_elo,
                    rating_gap=rating_gap,
                    sample_scope="+".join(scope_tags),
                )
            )
        return predictions


def _enrichment_defaults(present: bool) -> list[float]:
    return [
        DEFAULT_STARTER_ERA,
        DEFAULT_STARTER_ERA,
        0.0,
        DEFAULT_STARTER_WHIP,
        DEFAULT_STARTER_WHIP,
        DEFAULT_STARTER_K9,
        DEFAULT_STARTER_K9,
        DEFAULT_PYTHAG,
        DEFAULT_PYTHAG,
        0.0,
        DEFAULT_RPG,
        DEFAULT_RPG,
        DEFAULT_RPG,
        DEFAULT_RPG,
        1.0 if present else 0.0,
    ]


def ensemble_predictions(
    market: MarketDefinition,
    primary: list[Prediction],
    secondary: list[Prediction],
    weight_primary: float = 0.5,
) -> list[Prediction]:
    """Geometric-mean blend of two sets of predictions for the same matches."""

    if not primary:
        return secondary
    if not secondary:
        return primary
    if abs(weight_primary - 0.5) < 1e-9:
        weight_primary = 0.5
    weight_secondary = 1.0 - weight_primary

    secondary_by_seq = {item.match.match_seq: item for item in secondary}
    blended: list[Prediction] = []
    for item in primary:
        other = secondary_by_seq.get(item.match.match_seq)
        if other is None:
            blended.append(item)
            continue
        merged: dict[str, float] = {}
        for code in market.ordered_codes:
            p1 = max(item.probabilities.get(code, 0.0), 1e-9)
            p2 = max(other.probabilities.get(code, 0.0), 1e-9)
            merged[code] = (p1 ** weight_primary) * (p2 ** weight_secondary)
        total = sum(merged.values()) or 1.0
        merged = {code: prob / total for code, prob in merged.items()}
        pick_code = max(merged, key=merged.get)
        blended.append(
            Prediction(
                match=item.match,
                pick_code=pick_code,
                pick_label=market.label_map[pick_code],
                probabilities=merged,
                home_rating=item.home_rating,
                away_rating=item.away_rating,
                rating_gap=item.rating_gap,
                sample_scope=f"ensemble({item.sample_scope}+{other.sample_scope})",
            )
        )
    return blended
