from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from betman_predictor.models import MarketDefinition, RoundReference


class BetmanClient:
    """Minimal client for the public Betman endpoints used by this project."""

    BASE_URL = "https://www.betman.co.kr"

    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: int = 20,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_cache(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        referer: str,
        cache_path: Path | None = None,
        max_age_seconds: int | None = None,
        refresh: bool = False,
    ) -> Any:
        if cache_path and not refresh:
            cached = self._load_cache(cache_path, max_age_seconds)
            if cached is not None:
                return cached

        body = dict(payload)
        body.setdefault("_sbmInfo", {"debugMode": "false"})

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": self.BASE_URL,
            "Referer": referer,
        }

        last_error: Exception | None = None
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    data=json.dumps(body),
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type.lower():
                    snippet = response.text[:500].strip()
                    raise RuntimeError(
                        f"Expected JSON from {endpoint}, received {content_type or 'unknown'}: {snippet}"
                    )
                data = response.json()
                if cache_path:
                    self._save_cache(cache_path, data)
                return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff_seconds * attempt)

        raise RuntimeError(f"Betman request failed for {endpoint}") from last_error

    def get_buyable_games(self, refresh: bool = False) -> list[dict[str, Any]]:
        cache_path = self._cache_path("api", "buyable-games.json")
        payload = {}
        data = self._post_json(
            endpoint="/buyPsblGame/inqCacheBuyAbleGameInfoList.do",
            payload=payload,
            referer=f"{self.BASE_URL}/main/mainPage/gamebuy/buyableGameList.do",
            cache_path=cache_path,
            max_age_seconds=900,
            refresh=refresh,
        )
        proto_games = data.get("protoGames", [])
        toto_games = data.get("totoGames", [])
        return list(proto_games) + list(toto_games)

    def get_closed_rounds(
        self,
        gm_id: str,
        limit: int = 120,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        page_size = min(max(limit, 20), 100)
        collected: list[dict[str, Any]] = []
        start = 0
        draw = 1

        while len(collected) < limit:
            cache_path = self._cache_path("api", "closed-rounds", f"{gm_id}-{start}-{page_size}.json")
            payload = {"gmId": gm_id, "draw": draw, "start": start, "length": page_size}
            data = self._post_json(
                endpoint="/buyPsblGame/closedList.do",
                payload=payload,
                referer=f"{self.BASE_URL}/main/mainPage/gamebuy/closedGameList.do",
                cache_path=cache_path,
                max_age_seconds=21600,
                refresh=refresh,
            )
            rows = data.get("schedules", {}).get("data", [])
            if not rows:
                break
            collected.extend(rows)
            total = data.get("schedules", {}).get("recordsTotal", len(collected))
            start += page_size
            draw += 1
            if start >= total:
                break

        return collected[:limit]

    def get_round_detail(
        self,
        gm_id: str,
        gm_ts: int | str,
        game_year: str = "",
        refresh: bool = False,
    ) -> dict[str, Any]:
        gm_ts_str = str(gm_ts)
        cache_path = self._cache_path("api", "round-details", f"{gm_id}-{gm_ts_str}.json")
        payload = {"gmId": gm_id, "gmTs": gm_ts_str, "gameYear": game_year}
        return self._post_json(
            endpoint="/buyPsblGame/gameInfoInq.do",
            payload=payload,
            referer=f"{self.BASE_URL}/main/mainPage/gamebuy/gameSlip.do?gmId={gm_id}&gmTs={gm_ts_str}",
            cache_path=cache_path,
            max_age_seconds=21600,
            refresh=refresh,
        )

    def discover_target_round(self, market: MarketDefinition, refresh: bool = False) -> RoundReference:
        live_games = [row for row in self.get_buyable_games(refresh=refresh) if row.get("gmId") == market.gm_id]
        if live_games:
            live_games.sort(key=lambda row: int(row.get("gmTs", 0)), reverse=True)
            row = live_games[0]
            return RoundReference(
                gm_id=market.gm_id,
                gm_ts=int(row["gmTs"]),
                round_no=int(row.get("gmOsidTs") or 0) or None,
                round_year=int(row.get("gmOsidTsYear") or 0) or None,
                sale_status=row.get("saleStatus"),
                source="buyable",
                game_name=(row.get("gameMaster") or {}).get("gameName"),
            )

        recent_closed = self.get_closed_rounds(market.gm_id, limit=12, refresh=refresh)
        for row in recent_closed:
            sale_status = row.get("saleStatus")
            if sale_status in {"SaleProgress", "SaleComplete"}:
                return RoundReference(
                    gm_id=market.gm_id,
                    gm_ts=int(row["gmTs"]),
                    round_no=int(row.get("gmOsidTs") or 0) or None,
                    round_year=int(row.get("gmOsidTsYear") or 0) or None,
                    sale_status=sale_status,
                    source="closed-list",
                    game_name=(row.get("gameMaster") or {}).get("gameName"),
                )

        if not recent_closed:
            raise RuntimeError(f"Could not find any rounds for {market.name_ko}.")

        fallback = recent_closed[0]
        return RoundReference(
            gm_id=market.gm_id,
            gm_ts=int(fallback["gmTs"]),
            round_no=int(fallback.get("gmOsidTs") or 0) or None,
            round_year=int(fallback.get("gmOsidTsYear") or 0) or None,
            sale_status=fallback.get("saleStatus"),
            source="closed-list-fallback",
            game_name=(fallback.get("gameMaster") or {}).get("gameName"),
        )

