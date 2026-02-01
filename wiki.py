"""
wikipop.py

Build season-aligned popularity metrics using Wikipedia Pageviews (Wikimedia REST API).

Inputs:
  - 2026_MCM_Problem_C_Data.csv   (DWTS dataset; must contain columns: season, celebrity_name)
  - dates.csv                    (must contain columns: season, week, date)

Outputs:
  - wiki_popularity_by_season.csv  (one row per (season, celebrity))
  - cache/                         (resolved titles + pageviews JSON)

What it measures:
  - Calendar-year popularity: daily pageviews from Jan 1 to Dec 31 of the year the season started.
  - Season-window popularity: daily pageviews from season start to season end (based on dates.csv).
  - Baseline popularity: 28 days before season start (mean pageviews).
  - Lift: season_mean - baseline_mean.

Notes:
  - Wikipedia page titles may not match celebrity_name exactly; this script resolves titles via the
    MediaWiki search API and skips disambiguation pages when possible.
  - Some names will still require overrides (e.g. common names). Add to NAME_OVERRIDES.

Install:
  pip install pandas requests numpy
"""

from __future__ import annotations

import os
import json
import time
import random
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


DATA_PATH = "data/main.csv"
DATES_PATH = "data/dates.csv"
OUT_PATH = "wiki_popularity_by_season.csv"

CACHE_DIR = "cache"
TITLE_CACHE_PATH = os.path.join(CACHE_DIR, "wiki_title_cache.json")

# If you only want to run a few seasons while testing:
SEASONS_TO_RUN: Optional[set[int]] = None   # None = all seasons; or e.g., {27, 28}

# Baseline window length (pre-season)
BASELINE_DAYS = 28

# Polite pacing + retry
SLEEP_MIN = 0.05
SLEEP_MAX = 0.15
MAX_RETRIES = 6

HEADERS = {
    "User-Agent": "MCM2026-DWTS-WikiPageviews/1.0 (academic project)"
}

# Manual title overrides for ambiguous names (you will likely add a few)
# Use Wikipedia titles with spaces replaced by underscores.
NAME_OVERRIDES: Dict[str, str] = {
    # "Mario": "Mario_Lopez",
    # "Nick Carter": "Nick_Carter_(musician)",
    # "Monica": "Monica_(singer)",
}


@dataclass
class SeasonWindow:
    season: int
    start: pd.Timestamp
    end: pd.Timestamp


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_title_cache() -> Dict[str, str]:
    ensure_cache_dir()
    if os.path.exists(TITLE_CACHE_PATH):
        try:
            with open(TITLE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_title_cache(cache: Dict[str, str]) -> None:
    ensure_cache_dir()
    with open(TITLE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def jitter_sleep() -> None:
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def request_with_backoff(url: str, params: dict) -> Optional[dict]:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            jitter_sleep()
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                sleep_for = delay + random.uniform(0, delay * 0.25)
                print(f"[HTTP {r.status_code}] Backing off {sleep_for:.1f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(sleep_for)
                delay = min(delay * 2, 60)
                continue
            # Other errors: treat as failure for this query
            return None
        except requests.RequestException:
            sleep_for = delay + random.uniform(0, delay * 0.25)
            time.sleep(sleep_for)
            delay = min(delay * 2, 60)
    return None


def get_season_windows(dates_df: pd.DataFrame) -> List[SeasonWindow]:
    d = dates_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    start = d.loc[d["week"] == 1, ["season", "date"]].rename(columns={"date": "start"})
    end = d.groupby("season", as_index=False)["date"].max().rename(columns={"date": "end"})
    merged = start.merge(end, on="season", how="left").sort_values("season")
    return [SeasonWindow(int(r["season"]), r["start"], r["end"]) for _, r in merged.iterrows()]


def get_cast_by_season(data_df: pd.DataFrame) -> Dict[int, List[str]]:
    cast = (
        data_df.groupby("season")["celebrity_name"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    return {int(season): [n.strip() for n in names if str(n).strip()] for season, names in cast.items()}


def is_disambiguation(title: str) -> bool:
    """
    Check if a Wikipedia page is a disambiguation page via pageprops.
    """
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "ppprop": "disambiguation",
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
    }
    js = request_with_backoff(api, params)
    if not js:
        return False
    pages = js.get("query", {}).get("pages", [])
    if not pages:
        return False
    pageprops = pages[0].get("pageprops", {})
    return "disambiguation" in pageprops


def resolve_wikipedia_title(name: str, title_cache: Dict[str, str]) -> Optional[str]:
    """
    Resolve a celebrity name to a Wikipedia article title.
    Uses overrides first, otherwise MediaWiki search results (skipping disambiguation pages).
    Caches results by input name.
    """
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]

    if name in title_cache:
        return title_cache[name]

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "srlimit": 6,
        "format": "json",
        "formatversion": 2,
    }
    js = request_with_backoff(api, params)
    if not js:
        return None

    results = js.get("query", {}).get("search", [])
    if not results:
        return None

    # Pick first non-disambiguation title
    for r in results:
        title = r.get("title")
        if not title:
            continue
        if not is_disambiguation(title):
            title_cache[name] = title.replace(" ", "_")
            return title_cache[name]

    # Fallback: even if disambiguation, take top result
    title = results[0].get("title")
    if title:
        title_cache[name] = title.replace(" ", "_")
        return title_cache[name]
    return None


def cache_path_for_views(title: str, granularity: str, start_yyyymmdd: str, end_yyyymmdd: str) -> str:
    safe_title = title.replace("/", "_")
    return os.path.join(CACHE_DIR, f"views__{safe_title}__{granularity}__{start_yyyymmdd}_{end_yyyymmdd}.json")


def fetch_pageviews_daily(title: str, start_dt: date, end_dt: date) -> List[dict]:
    """
    Fetch daily pageviews for a single article title from Wikimedia REST API.

    Returns list of items, each with:
      - timestamp (YYYYMMDD00)
      - views (int)
    """
    ensure_cache_dir()
    encoded = urllib.parse.quote(title, safe="")
    start_s = start_dt.strftime("%Y%m%d")
    end_s = end_dt.strftime("%Y%m%d")
    cp = cache_path_for_views(title, "daily", start_s, end_s)

    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f).get("items", [])
        except Exception:
            pass

    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{encoded}/daily/{start_s}/{end_s}"
    )
    js = request_with_backoff(url, params={})
    if not js:
        return []

    with open(cp, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False)
    return js.get("items", [])


def summarize_views(items: List[dict]) -> Tuple[float, float, float, float]:
    """
    Return (total, mean, peak, std) from a list of pageview items.
    """
    if not items:
        return (np.nan, np.nan, np.nan, np.nan)
    views = [int(x.get("views", 0)) for x in items]
    if len(views) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    arr = np.array(views, dtype=float)
    return (float(arr.sum()), float(arr.mean()), float(arr.max()), float(arr.std(ddof=0)))


def main() -> None:
    ensure_cache_dir()

    data_df = pd.read_csv(DATA_PATH)
    dates_df = pd.read_csv(DATES_PATH)

    season_windows = get_season_windows(dates_df)
    cast_by_season = get_cast_by_season(data_df)

    title_cache = load_title_cache()

    rows = []
    for w in season_windows:
        season = w.season
        if SEASONS_TO_RUN is not None and season not in SEASONS_TO_RUN:
            continue
        if season not in cast_by_season:
            continue

        start_date = w.start.date()
        end_date = w.end.date()
        year = start_date.year

        # calendar year of season start
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)

        cast = cast_by_season[season]
        print(f"Season {season}: {len(cast)} contestants | season {start_date}..{end_date} | year {year}")

        for name in cast:
            title = resolve_wikipedia_title(name, title_cache)
            if not title:
                rows.append({
                    "season": season,
                    "celebrity_name": name,
                    "wiki_title": None,
                    "year": year,
                    "year_total": np.nan,
                    "year_mean": np.nan,
                    "year_peak": np.nan,
                    "year_std": np.nan,
                    "season_total": np.nan,
                    "season_mean": np.nan,
                    "season_peak": np.nan,
                    "season_std": np.nan,
                    "baseline_mean": np.nan,
                    "lift": np.nan,
                    "status": "no_title",
                })
                continue

            # Year views
            year_items = fetch_pageviews_daily(title, year_start, year_end)
            year_total, year_mean, year_peak, year_std = summarize_views(year_items)

            # Season views
            season_items = fetch_pageviews_daily(title, start_date, end_date)
            season_total, season_mean, season_peak, season_std = summarize_views(season_items)

            # Baseline (pre-season)
            baseline_start = start_date - timedelta(days=BASELINE_DAYS)
            baseline_end = start_date - timedelta(days=1)
            baseline_items = fetch_pageviews_daily(title, baseline_start, baseline_end)
            _, baseline_mean, _, _ = summarize_views(baseline_items)

            lift = season_mean - baseline_mean if (not np.isnan(season_mean) and not np.isnan(baseline_mean)) else np.nan

            rows.append({
                "season": season,
                "celebrity_name": name,
                "wiki_title": title,
                "year": year,
                "year_total": year_total,
                "year_mean": year_mean,
                "year_peak": year_peak,
                "year_std": year_std,
                "season_total": season_total,
                "season_mean": season_mean,
                "season_peak": season_peak,
                "season_std": season_std,
                "baseline_mean": baseline_mean,
                "lift": lift,
                "status": "ok",
            })

    # Save outputs
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    save_title_cache(title_cache)
    print(f"\nWrote {OUT_PATH} with {len(out_df)} rows")
    print(f"Title cache saved to {TITLE_CACHE_PATH}")


if __name__ == "__main__":
    main()
