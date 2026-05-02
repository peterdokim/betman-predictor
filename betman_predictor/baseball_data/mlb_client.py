"""Thin client for the public MLB Stats API.

Endpoints used:

- /api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher,team
- /api/v1/people/{personId}/stats?stats=season&group=pitching&season=YYYY
- /api/v1/teams/{teamId}/stats?stats=byDateRange&group=hitting,pitching&...

All responses are cached on disk under cache/baseball/mlb/. The schedule cache
is invalidated daily; per-pitcher and per-team windows are cached forever
because they're indexed by date and never change after the fact.

Team-window queries are bucketed to the most recent Sunday before the game so
that all matches in the same week share one cache entry — this dramatically
cuts cold-cache fetch counts during training.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://statsapi.mlb.com/api/v1"


def week_bucket_end(end_date_iso: str) -> str:
    """Return the most recent Sunday strictly before `end_date_iso`."""
    d = date.fromisoformat(end_date_iso)
    offset = (d.weekday() + 1) % 7  # Mon=0->1, Sun=6->0
    if offset == 0:
        offset = 7
    return (d - timedelta(days=offset)).isoformat()


class MLBStatsClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: int = 15,
        max_retries: int = 3,
        backoff_seconds: float = 0.7,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Per-cache-key locks so concurrent threads asking for the same URL
        # don't race the network — the second waiter just reads the file.
        self._inflight_locks: dict[str, threading.Lock] = {}
        self._inflight_master_lock = threading.Lock()

    def _key_lock(self, cache_path: Path) -> threading.Lock:
        key = str(cache_path)
        with self._inflight_master_lock:
            lock = self._inflight_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._inflight_locks[key] = lock
            return lock

    def _cache_path(self, *parts: str) -> Path:
        path = self.cache_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_cache(self, path: Path, max_age_seconds: int | None) -> Any | None:
        if not path.exists():
            return None
        if max_age_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > max_age_seconds:
                return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
        cache_path: Path,
        max_age_seconds: int | None,
        refresh: bool,
    ) -> Any:
        if not refresh:
            cached = self._load_cache(cache_path, max_age_seconds)
            if cached is not None:
                return cached

        # Serialize concurrent requests for the same key. The second arrival
        # finds the file already written by the first.
        lock = self._key_lock(cache_path)
        with lock:
            if not refresh:
                cached = self._load_cache(cache_path, max_age_seconds)
                if cached is not None:
                    return cached

            url = f"{BASE_URL}{endpoint}"
            last_error: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = requests.get(url, params=params, timeout=self.timeout_seconds)
                    response.raise_for_status()
                    payload = response.json()
                    self._save_cache(cache_path, payload)
                    return payload
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt == self.max_retries:
                        break
                    time.sleep(self.backoff_seconds * attempt)
            raise RuntimeError(f"MLB Stats API failed for {endpoint}") from last_error

    def schedule_for_date(self, iso_date: str, refresh: bool = False) -> Any:
        cache_path = self._cache_path("schedule", f"{iso_date}.json")
        return self._get(
            endpoint="/schedule",
            params={
                "sportId": 1,
                "date": iso_date,
                "hydrate": "probablePitcher,team",
            },
            cache_path=cache_path,
            max_age_seconds=21600,  # 6 hours
            refresh=refresh,
        )

    def pitcher_season_stats(
        self,
        person_id: int,
        season: int,
        refresh: bool = False,
    ) -> Any:
        cache_path = self._cache_path("pitcher", str(season), f"{person_id}.json")
        return self._get(
            endpoint=f"/people/{person_id}/stats",
            params={"stats": "season", "group": "pitching", "season": season},
            cache_path=cache_path,
            max_age_seconds=43200,  # 12 hours
            refresh=refresh,
        )

    def pitcher_recent_stats(
        self,
        person_id: int,
        season: int,
        end_date_iso: str,
        window_days: int = 30,
        refresh: bool = False,
    ) -> Any:
        """Cumulative pitcher stats over a recent window, week-bucketed for cache reuse.

        Captures last ~5-6 starts, which is more responsive to current form
        than season-to-date. Uses the same week anchor as team_recent_stats so
        many pitchers in the same training week share cache pressure.
        """
        bucket_end_iso = week_bucket_end(end_date_iso)
        bucket_end_dt = date.fromisoformat(bucket_end_iso)
        start_iso = (bucket_end_dt - timedelta(days=window_days)).isoformat()
        cache_path = self._cache_path(
            "pitcher_recent",
            str(season),
            f"{person_id}-w{bucket_end_iso}.json",
        )
        return self._get(
            endpoint=f"/people/{person_id}/stats",
            params={
                "stats": "byDateRange",
                "group": "pitching",
                "startDate": start_iso,
                "endDate": bucket_end_iso,
                "season": season,
            },
            cache_path=cache_path,
            max_age_seconds=None,
            refresh=refresh,
        )

    def team_recent_stats(
        self,
        team_id: int,
        season: int,
        end_date_iso: str,
        window_days: int = 21,
        refresh: bool = False,
    ) -> Any:
        """Cumulative team stats over [bucket_end - window_days, bucket_end].

        End-date is bucketed to the previous Sunday so all matches in the same
        Mon–Sun week share one cache entry. This trades a few days of recency
        for ~7× fewer unique API calls during training. Window_days=21 captures
        ~15-18 games on a typical MLB schedule.
        """
        bucket_end_iso = week_bucket_end(end_date_iso)
        bucket_end_dt = date.fromisoformat(bucket_end_iso)
        start_iso = (bucket_end_dt - timedelta(days=window_days)).isoformat()
        cache_path = self._cache_path(
            "team_window",
            str(season),
            f"{team_id}-w{bucket_end_iso}.json",
        )
        return self._get(
            endpoint=f"/teams/{team_id}/stats",
            params={
                "stats": "byDateRange",
                "group": "hitting,pitching",
                "startDate": start_iso,
                "endDate": bucket_end_iso,
                "season": season,
            },
            cache_path=cache_path,
            max_age_seconds=None,
            refresh=refresh,
        )
