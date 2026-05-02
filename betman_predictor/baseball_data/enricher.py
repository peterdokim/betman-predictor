"""High-level enrichment for Betman MLB matches.

Given a Betman MatchRecord (homeId/awayId/game_datetime), produces a small
MatchEnrichment with: probable starter ERA/WHIP/K9 for each side, plus
last-window team form (runs scored/allowed per game and Pythagorean win
expectancy).

All numeric fields are Optional — callers must check `present` flags before
using them. Unreachable APIs or missing pitchers degrade silently to None.

For ML training over many historical matches, call `prefetch_for(matches)`
first — it pre-warms the cache for all unique (date, pitcher, team-week)
keys in a concurrent batch, after which `lookup(match)` is purely cached.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from betman_predictor.baseball_data.mlb_client import MLBStatsClient, week_bucket_end
from betman_predictor.baseball_data.team_mapping import betman_to_mlb_id, park_factor_for_home_team
from betman_predictor.models import MatchRecord


PYTHAG_EXPONENT = 1.83  # Bill James' baseball Pythagorean exponent


@dataclass(frozen=True)
class MatchEnrichment:
    home_starter_era: Optional[float] = None
    away_starter_era: Optional[float] = None
    home_starter_whip: Optional[float] = None
    away_starter_whip: Optional[float] = None
    home_starter_k9: Optional[float] = None
    away_starter_k9: Optional[float] = None
    home_starter_innings: Optional[float] = None
    away_starter_innings: Optional[float] = None
    home_recent_runs_per_game: Optional[float] = None
    away_recent_runs_per_game: Optional[float] = None
    home_recent_runs_allowed_per_game: Optional[float] = None
    away_recent_runs_allowed_per_game: Optional[float] = None
    home_recent_pythag: Optional[float] = None
    away_recent_pythag: Optional[float] = None
    home_recent_games: int = 0
    away_recent_games: int = 0
    # New in pass-3.5: team hitting depth + pitcher recent form + park factor
    home_recent_team_ops: Optional[float] = None
    away_recent_team_ops: Optional[float] = None
    home_recent_team_obp: Optional[float] = None
    away_recent_team_obp: Optional[float] = None
    home_recent_team_slg: Optional[float] = None
    away_recent_team_slg: Optional[float] = None
    home_starter_recent_era: Optional[float] = None
    away_starter_recent_era: Optional[float] = None
    home_starter_recent_whip: Optional[float] = None
    away_starter_recent_whip: Optional[float] = None
    home_starter_recent_innings: Optional[float] = None
    away_starter_recent_innings: Optional[float] = None
    park_factor: Optional[int] = None
    sources: tuple[str, ...] = field(default_factory=tuple)


NULL_ENRICHMENT = MatchEnrichment()


class BaseballEnricher:
    """Looks up MLB-side enrichment for Betman match records."""

    def __init__(
        self,
        cache_root: Path,
        verbose: bool = False,
    ) -> None:
        self.client = MLBStatsClient(cache_dir=cache_root / "mlb")
        self.verbose = verbose
        self._game_lookup_cache: dict[str, dict[tuple[int, int], dict]] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [enrich] {message}", flush=True)

    def prefetch_for(
        self,
        matches: Iterable[MatchRecord],
        max_workers: int = 8,
        progress_callback: Optional[Any] = None,
    ) -> None:
        """Pre-warm the cache for all matches in concurrent batches.

        Three stages: (1) unique schedule dates → builds an in-memory game
        index, (2) unique probable-pitcher (id, season) pairs from that index,
        (3) unique (team_id, season, week_anchor) team-window keys.
        """
        mlb_matches = [m for m in matches if m.league_name == "MLB" and m.game_datetime is not None]
        if not mlb_matches:
            return

        schedule_dates: set[str] = set()
        for match in mlb_matches:
            for iso in self._candidate_dates(match):
                schedule_dates.add(iso)

        self._concurrent_run(
            label="schedule",
            jobs=[(self._schedule_lookup, (iso,)) for iso in schedule_dates],
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

        pitcher_jobs: set[tuple[int, int]] = set()
        pitcher_recent_jobs: set[tuple[int, int, str]] = set()
        team_window_jobs: set[tuple[int, int, str]] = set()
        for match in mlb_matches:
            home_id = betman_to_mlb_id(match.home_id)
            away_id = betman_to_mlb_id(match.away_id)
            if not home_id or not away_id:
                continue
            game = self._find_game(home_id, away_id, list(self._candidate_dates(match)))
            if game is None:
                continue
            season = int(game.get("season") or match.game_datetime.year)
            end_iso = (game.get("officialDate") or match.game_datetime.astimezone(timezone.utc).date().isoformat())
            anchor = week_bucket_end(end_iso)

            for side in ("home", "away"):
                starter = _extract_probable(game, side)
                if starter and starter.get("id"):
                    pid = int(starter["id"])
                    pitcher_jobs.add((pid, season))
                    pitcher_recent_jobs.add((pid, season, anchor))

            team_window_jobs.add((home_id, season, anchor))
            team_window_jobs.add((away_id, season, anchor))

        self._concurrent_run(
            label="pitcher",
            jobs=[(self.client.pitcher_season_stats, (pid, season)) for pid, season in pitcher_jobs],
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

        def _pitcher_recent_call(pid: int, season: int, anchor: str) -> None:
            self.client.pitcher_recent_stats(pid, season, _next_day(anchor))

        self._concurrent_run(
            label="pitcher_recent",
            jobs=[(_pitcher_recent_call, (pid, season, anchor)) for pid, season, anchor in pitcher_recent_jobs],
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

        # Translate team window jobs back to a callable. The end_iso passed in
        # gets re-bucketed by team_recent_stats to the same anchor.
        def _team_call(team_id: int, season: int, anchor: str) -> None:
            self.client.team_recent_stats(team_id, season, _next_day(anchor))

        self._concurrent_run(
            label="team_window",
            jobs=[(_team_call, (tid, season, anchor)) for tid, season, anchor in team_window_jobs],
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

    def _concurrent_run(
        self,
        label: str,
        jobs: list[tuple[Any, tuple]],
        max_workers: int,
        progress_callback: Optional[Any] = None,
    ) -> None:
        if not jobs:
            return
        if progress_callback:
            progress_callback(f"{label}: {len(jobs)} unique keys")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fn, *args) for fn, args in jobs]
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    self._log(f"{label} job failed: {exc}")
                if progress_callback and (done == 1 or done % 50 == 0 or done == len(jobs)):
                    progress_callback(f"{label}: {done}/{len(jobs)}")

    def _candidate_dates(self, match: MatchRecord) -> list[str]:
        if match.game_datetime is None:
            return []
        local = match.game_datetime.astimezone()
        utc = match.game_datetime.astimezone(timezone.utc)
        seen: set[str] = set()
        out: list[str] = []
        for iso in (
            local.date().isoformat(),
            utc.date().isoformat(),
            (utc - timedelta(days=1)).date().isoformat(),
            (utc + timedelta(days=1)).date().isoformat(),
        ):
            if iso not in seen:
                seen.add(iso)
                out.append(iso)
        return out

    def lookup(self, match: MatchRecord) -> MatchEnrichment:
        # MLB-only adapter; KBO is a planned follow-up.
        if match.league_name != "MLB":
            return NULL_ENRICHMENT

        home_id = betman_to_mlb_id(match.home_id)
        away_id = betman_to_mlb_id(match.away_id)
        if not home_id or not away_id or match.game_datetime is None:
            return NULL_ENRICHMENT

        game_dt_local = match.game_datetime.astimezone()
        game_dt_utc = match.game_datetime.astimezone(timezone.utc)
        # MLB schedule API is keyed by ET-zone date; using UTC date is close
        # enough for non-pacific midnight games. Try local + UTC + UTC-1 day.
        game = self._find_game(home_id, away_id, [
            game_dt_local.date().isoformat(),
            game_dt_utc.date().isoformat(),
            (game_dt_utc - _one_day()).date().isoformat(),
            (game_dt_utc + _one_day()).date().isoformat(),
        ])
        if game is None:
            self._log(f"no MLB schedule match for {match.home_name}/{match.away_name} ({match.game_date_str})")
            return NULL_ENRICHMENT

        season = int(game.get("season") or game_dt_utc.year)
        start_iso = (game.get("officialDate") or game_dt_utc.date().isoformat())

        home_starter = _extract_probable(game, "home")
        away_starter = _extract_probable(game, "away")
        home_pitch = self._pitcher_stats(home_starter, season)
        away_pitch = self._pitcher_stats(away_starter, season)
        home_pitch_recent = self._pitcher_recent_form(home_starter, season, start_iso)
        away_pitch_recent = self._pitcher_recent_form(away_starter, season, start_iso)

        home_form = self._team_form(home_id, season, start_iso)
        away_form = self._team_form(away_id, season, start_iso)

        park_factor = park_factor_for_home_team(home_id)

        sources: list[str] = ["mlb_schedule"]
        if home_pitch or away_pitch:
            sources.append("mlb_pitcher_stats")
        if home_pitch_recent or away_pitch_recent:
            sources.append("mlb_pitcher_recent")
        if home_form or away_form:
            sources.append("mlb_team_window")

        return MatchEnrichment(
            home_starter_era=_get_float(home_pitch, "era"),
            away_starter_era=_get_float(away_pitch, "era"),
            home_starter_whip=_get_float(home_pitch, "whip"),
            away_starter_whip=_get_float(away_pitch, "whip"),
            home_starter_k9=_get_float(home_pitch, "strikeoutsPer9Inn"),
            away_starter_k9=_get_float(away_pitch, "strikeoutsPer9Inn"),
            home_starter_innings=_get_float(home_pitch, "inningsPitched"),
            away_starter_innings=_get_float(away_pitch, "inningsPitched"),
            home_recent_runs_per_game=(home_form or {}).get("rs_pg"),
            away_recent_runs_per_game=(away_form or {}).get("rs_pg"),
            home_recent_runs_allowed_per_game=(home_form or {}).get("ra_pg"),
            away_recent_runs_allowed_per_game=(away_form or {}).get("ra_pg"),
            home_recent_pythag=(home_form or {}).get("pythag"),
            away_recent_pythag=(away_form or {}).get("pythag"),
            home_recent_games=int((home_form or {}).get("games", 0)),
            away_recent_games=int((away_form or {}).get("games", 0)),
            home_recent_team_ops=(home_form or {}).get("ops"),
            away_recent_team_ops=(away_form or {}).get("ops"),
            home_recent_team_obp=(home_form or {}).get("obp"),
            away_recent_team_obp=(away_form or {}).get("obp"),
            home_recent_team_slg=(home_form or {}).get("slg"),
            away_recent_team_slg=(away_form or {}).get("slg"),
            home_starter_recent_era=_get_float(home_pitch_recent, "era"),
            away_starter_recent_era=_get_float(away_pitch_recent, "era"),
            home_starter_recent_whip=_get_float(home_pitch_recent, "whip"),
            away_starter_recent_whip=_get_float(away_pitch_recent, "whip"),
            home_starter_recent_innings=_get_float(home_pitch_recent, "inningsPitched"),
            away_starter_recent_innings=_get_float(away_pitch_recent, "inningsPitched"),
            park_factor=park_factor,
            sources=tuple(sources),
        )

    def _find_game(self, home_id: int, away_id: int, candidate_dates: list[str]) -> Optional[dict]:
        seen: set[str] = set()
        for iso in candidate_dates:
            if iso in seen:
                continue
            seen.add(iso)
            try:
                lookup = self._schedule_lookup(iso)
            except Exception as exc:  # noqa: BLE001
                self._log(f"schedule fetch failed for {iso}: {exc}")
                continue
            game = lookup.get((home_id, away_id))
            if game is not None:
                return game
        return None

    def _schedule_lookup(self, iso_date: str) -> dict[tuple[int, int], dict]:
        cached = self._game_lookup_cache.get(iso_date)
        if cached is not None:
            return cached
        payload = self.client.schedule_for_date(iso_date)
        lookup: dict[tuple[int, int], dict] = {}
        for date_block in payload.get("dates", []):
            for game in date_block.get("games", []):
                teams = game.get("teams", {}) or {}
                home_team = ((teams.get("home") or {}).get("team") or {}).get("id")
                away_team = ((teams.get("away") or {}).get("team") or {}).get("id")
                if home_team and away_team:
                    lookup[(int(home_team), int(away_team))] = game
        self._game_lookup_cache[iso_date] = lookup
        return lookup

    def _pitcher_stats(self, person: Optional[dict], season: int) -> Optional[dict[str, Any]]:
        if not person:
            return None
        person_id = person.get("id")
        if not person_id:
            return None
        try:
            payload = self.client.pitcher_season_stats(int(person_id), season)
        except Exception as exc:  # noqa: BLE001
            self._log(f"pitcher stats fetch failed (id={person_id}): {exc}")
            return None
        for stat_block in payload.get("stats", []):
            splits = stat_block.get("splits") or []
            if not splits:
                continue
            stat = (splits[0].get("stat") or {})
            return stat
        return None

    def _team_form(self, team_id: int, season: int, end_iso: str) -> Optional[dict[str, float]]:
        try:
            payload = self.client.team_recent_stats(team_id, season, end_iso)
        except Exception as exc:  # noqa: BLE001
            self._log(f"team window fetch failed (team={team_id}): {exc}")
            return None

        runs_scored = None
        runs_allowed = None
        games_played = 0
        ops = obp = slg = None
        for stat_block in payload.get("stats", []):
            splits = stat_block.get("splits") or []
            if not splits:
                continue
            stat = splits[0].get("stat") or {}
            group = ((stat_block.get("group") or {}).get("displayName") or "").lower()
            if group == "hitting":
                runs_scored = _safe_float(stat.get("runs"))
                games_played = max(games_played, _safe_int(stat.get("gamesPlayed")))
                ops = _safe_float(stat.get("ops"))
                obp = _safe_float(stat.get("obp"))
                slg = _safe_float(stat.get("slg"))
            elif group == "pitching":
                runs_allowed = _safe_float(stat.get("runs"))
                games_played = max(games_played, _safe_int(stat.get("gamesPlayed")))

        if not games_played:
            return None
        rs_pg = (runs_scored / games_played) if runs_scored is not None else None
        ra_pg = (runs_allowed / games_played) if runs_allowed is not None else None
        pythag = None
        if rs_pg is not None and ra_pg is not None and (rs_pg + ra_pg) > 0:
            rs_e = rs_pg ** PYTHAG_EXPONENT
            ra_e = ra_pg ** PYTHAG_EXPONENT
            pythag = rs_e / (rs_e + ra_e)
        return {
            "rs_pg": rs_pg,
            "ra_pg": ra_pg,
            "pythag": pythag,
            "games": games_played,
            "ops": ops,
            "obp": obp,
            "slg": slg,
        }

    def _pitcher_recent_form(
        self,
        person: Optional[dict],
        season: int,
        end_date_iso: str,
    ) -> Optional[dict[str, Any]]:
        if not person:
            return None
        person_id = person.get("id")
        if not person_id:
            return None
        try:
            payload = self.client.pitcher_recent_stats(int(person_id), season, end_date_iso)
        except Exception as exc:  # noqa: BLE001
            self._log(f"pitcher recent fetch failed (id={person_id}): {exc}")
            return None
        for stat_block in payload.get("stats", []):
            splits = stat_block.get("splits") or []
            if not splits:
                continue
            stat = splits[0].get("stat") or {}
            return stat
        return None


def _extract_probable(game: dict, side: str) -> Optional[dict]:
    teams = game.get("teams") or {}
    side_block = teams.get(side) or {}
    return side_block.get("probablePitcher")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_float(stat: Optional[dict], key: str) -> Optional[float]:
    if not stat:
        return None
    return _safe_float(stat.get(key))


def _one_day():
    return timedelta(days=1)


def _next_day(iso: str) -> str:
    """Add 1 day to an ISO date so the bucket-end logic still rounds back to it."""
    from datetime import date

    return (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
