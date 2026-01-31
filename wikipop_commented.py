"""
wikipop_commented.py

This script builds *season-aligned popularity metrics* using **Wikipedia pageviews**.

WHY Wikipedia pageviews?
- It's a public, documented API (Wikimedia REST API), so it is stable and reproducible.
- It provides a daily time series (views/day) you can align to DWTS season dates.

WHAT THIS SCRIPT OUTPUTS
- A CSV file: `wiki_popularity_by_season.csv`
  One row per (season, celebrity_name), with summary statistics computed from pageviews.

WHAT IT MEASURES FOR EACH CONTESTANT
1) Calendar-year popularity (Jan 1 → Dec 31 of the year the season starts)
   - year_total, year_mean, year_peak, year_std
2) Season-window popularity (season start → season end from dates.csv)
   - season_total, season_mean, season_peak, season_std
3) Baseline popularity (the BASELINE_DAYS days right before the season starts)
   - baseline_mean
4) Lift (how much attention increased during the season compared to baseline)
   - lift = season_mean - baseline_mean

INPUT FILES
- 2026_MCM_Problem_C_Data.csv
    Needs columns:
      - season
      - celebrity_name
- dates.csv
    Needs columns:
      - season
      - week
      - date

CACHING (IMPORTANT FOR SPEED)
This script caches:
- Resolved Wikipedia titles (name -> title) in: cache/wiki_title_cache.json
- Pageview API responses in JSON files like:
    cache/views__{title}__daily__{start}_{end}.json
So rerunning is fast and doesn't re-download data you already fetched.

RUNNING
    pip install pandas requests numpy
    python wikipop_commented.py

TIP: Start with a couple seasons (e.g. {27, 28}) to validate titles and output.
"""

from __future__ import annotations

import os
import json
import time
import random
import urllib.parse
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# -----------------------------
# File paths (adjust if needed)
# -----------------------------
DATA_PATH = "2026_MCM_Problem_C_Data.csv"      # Your main DWTS dataset (season + celebrity_name)
DATES_PATH = "dates.csv"                       # Maps (season, week) -> date
OUT_PATH = "wiki_popularity_by_season.csv"     # Output CSV we write

# -----------------------------
# Cache settings
# -----------------------------
CACHE_DIR = "cache"
TITLE_CACHE_PATH = os.path.join(CACHE_DIR, "wiki_title_cache.json")


# ----------------------------------------------------------
# Season selection (set to None to run all seasons)
# ----------------------------------------------------------
# If you only want to run a few seasons while testing, set a set of ints:
#   SEASONS_TO_RUN = {27, 28}
# If you want ALL seasons, set:
#   SEASONS_TO_RUN = None
SEASONS_TO_RUN: Optional[set[int]] = {29}


# ----------------------------------------------------------
# Baseline window configuration
# ----------------------------------------------------------
# Baseline is the "pre-season" period used to compute a reference mean.
# For each contestant:
#   baseline window = [season_start - BASELINE_DAYS, season_start - 1 day]
BASELINE_DAYS = 28


# ----------------------------------------------------------
# Polite pacing and retries
# ----------------------------------------------------------
# Wikipedia is generally permissive, but it's polite to sleep a bit.
# If you do many contestants/seasons, this keeps you from hammering the API.
SLEEP_MIN = 0.05
SLEEP_MAX = 0.15

# Max retries for transient failures (429 Too Many Requests / 503 Service Unavailable).
MAX_RETRIES = 6

# Wikipedia asks for a descriptive User-Agent.
HEADERS = {
    "User-Agent": "MCM2026-DWTS-WikiPageviews/1.0 (academic project)"
}


# ----------------------------------------------------------
# Manual overrides for ambiguous names
# ----------------------------------------------------------
# Title resolution is the hardest part. Many names resolve cleanly via search,
# but common names ("Monica", "Jordan", "Mario") may resolve wrong.
#
# Add manual overrides in this dict:
#   key   = celebrity_name from your dataset
#   value = Wikipedia article title with spaces -> underscores
NAME_OVERRIDES: Dict[str, str] = {
    # "Mario": "Mario_Lopez",
    # "Nick Carter": "Nick_Carter_(musician)",
    # "Monica": "Monica_(singer)",
}


# -----------------------------
# Small helper data structure
# -----------------------------
@dataclass
class SeasonWindow:
    """
    Represents the date range of a DWTS season according to dates.csv.
    - season: season number (int)
    - start:  timestamp for week 1 date (premiere week)
    - end:    timestamp for last available date in that season (finale week)
    """
    season: int
    start: pd.Timestamp
    end: pd.Timestamp


# -----------------------------
# Cache folder utilities
# -----------------------------
def ensure_cache_dir() -> None:
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_title_cache() -> Dict[str, str]:
    """
    Load the name->wiki_title cache from disk.
    This prevents repeating Wikipedia search calls on reruns.
    """
    ensure_cache_dir()
    if os.path.exists(TITLE_CACHE_PATH):
        try:
            with open(TITLE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # If cache file is corrupted, just ignore it and rebuild.
            return {}
    return {}


def save_title_cache(cache: Dict[str, str]) -> None:
    """Persist the title cache back to disk."""
    ensure_cache_dir()
    with open(TITLE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# -----------------------------
# Networking utilities
# -----------------------------
def jitter_sleep() -> None:
    """
    Sleep a tiny random amount to spread out requests.
    (Helps avoid looking like a bot that requests in perfectly regular intervals.)
    """
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def request_with_backoff(url: str, params: dict) -> Optional[dict]:
    """
    Make an HTTP GET request with retry + exponential backoff.

    Why this exists:
    - Some requests fail transiently (network issues, temporary server overload).
    - The Wikimedia API can return 429 or 503 occasionally.
    - We retry a few times rather than crashing.

    Returns:
    - Parsed JSON dict on success (HTTP 200)
    - None on permanent failure
    """
    delay = 1.0  # initial backoff seconds
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            jitter_sleep()
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)

            if r.status_code == 200:
                return r.json()

            # Retryable errors
            if r.status_code in (429, 503):
                sleep_for = delay + random.uniform(0, delay * 0.25)
                print(f"[HTTP {r.status_code}] Backing off {sleep_for:.1f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(sleep_for)
                delay = min(delay * 2, 60)
                continue

            # Other errors: treat as permanent failure for this call
            return None

        except requests.RequestException:
            # Network error: retry with backoff
            sleep_for = delay + random.uniform(0, delay * 0.25)
            time.sleep(sleep_for)
            delay = min(delay * 2, 60)

    # Failed after MAX_RETRIES attempts
    return None


# -----------------------------
# Input parsing helpers
# -----------------------------
def get_season_windows(dates_df: pd.DataFrame) -> List[SeasonWindow]:
    """
    From dates.csv, compute each season's (start_date, end_date).

    Assumption:
    - The row with week == 1 is the season start date.
    - The season end date is the *maximum date* for that season.

    If dates.csv is accurate, this creates correct airing windows.
    """
    d = dates_df.copy()
    d["date"] = pd.to_datetime(d["date"])

    # Start date: the date in week 1 for each season
    start = d.loc[d["week"] == 1, ["season", "date"]].rename(columns={"date": "start"})

    # End date: the last date for each season
    end = d.groupby("season", as_index=False)["date"].max().rename(columns={"date": "end"})

    merged = start.merge(end, on="season", how="left").sort_values("season")

    # Return a list of SeasonWindow dataclass objects (nice typed container)
    return [SeasonWindow(int(r["season"]), r["start"], r["end"]) for _, r in merged.iterrows()]


def get_cast_by_season(data_df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    From the DWTS dataset, build:
        season -> [unique celebrity names]

    We take unique names because the dataset likely has multiple rows per contestant
    across weeks, dances, etc.
    """
    cast = (
        data_df.groupby("season")["celebrity_name"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )

    # Clean whitespace and ensure season keys are ints
    return {
        int(season): [n.strip() for n in names if str(n).strip()]
        for season, names in cast.items()
    }


# -----------------------------
# Wikipedia title resolution
# -----------------------------
def is_disambiguation(title: str) -> bool:
    """
    Check if a Wikipedia page title is a disambiguation page.

    Why:
    - Wikipedia search may return a disambiguation page (e.g., "Jordan").
    - Pageviews for a disambiguation page are not what you want for a person.

    How:
    - Uses MediaWiki API pageprops with ppprop=disambiguation.
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
    Resolve a human name (celebrity_name) into a Wikipedia article title.

    Strategy:
    1) If name is in NAME_OVERRIDES, use that (manual fix for ambiguous names).
    2) Else, if name is cached, reuse cached title.
    3) Else, run Wikipedia search API and select:
         - the first *non-disambiguation* page from the top results.

    Returns:
    - A title string with underscores (e.g. "Mario_Lopez"), or
    - None if resolution fails.
    """
    # (1) Manual override
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]

    # (2) Cached title
    if name in title_cache:
        return title_cache[name]

    # (3) Wikipedia search
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "srlimit": 6,            # look at the top few results
        "format": "json",
        "formatversion": 2,
    }
    js = request_with_backoff(api, params)
    if not js:
        return None

    results = js.get("query", {}).get("search", [])
    if not results:
        return None

    # Choose first search result that isn't a disambiguation page
    for r in results:
        title = r.get("title")
        if not title:
            continue
        if not is_disambiguation(title):
            # Store with underscores because the pageviews REST endpoint expects URL-safe titles
            title_cache[name] = title.replace(" ", "_")
            return title_cache[name]

    # If everything was disambiguation (rare), fall back to the top result anyway
    title = results[0].get("title")
    if title:
        title_cache[name] = title.replace(" ", "_")
        return title_cache[name]

    return None


# -----------------------------
# Wikipedia pageviews download
# -----------------------------
def cache_path_for_views(title: str, granularity: str, start_yyyymmdd: str, end_yyyymmdd: str) -> str:
    """
    Build a deterministic cache filename for the pageviews request.
    This ensures the same (title, window) always maps to the same cache file.
    """
    safe_title = title.replace("/", "_")  # avoid creating nested directories
    return os.path.join(
        CACHE_DIR,
        f"views__{safe_title}__{granularity}__{start_yyyymmdd}_{end_yyyymmdd}.json"
    )


def fetch_pageviews_daily(title: str, start_dt: date, end_dt: date) -> List[dict]:
    """
    Fetch daily pageviews for ONE Wikipedia article between start_dt and end_dt.

    Uses the Wikimedia REST API endpoint:
      /metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}

    In our URL:
      project     = en.wikipedia
      access      = all-access
      agent       = user        (exclude spiders/bots)
      article     = {title}
      granularity = daily
      start/end   = YYYYMMDD
    """
    ensure_cache_dir()

    # URL-encode the title because parentheses, commas, etc. can appear in titles
    encoded = urllib.parse.quote(title, safe="")

    start_s = start_dt.strftime("%Y%m%d")
    end_s = end_dt.strftime("%Y%m%d")

    # Cache file for this exact request
    cp = cache_path_for_views(title, "daily", start_s, end_s)

    # If cached, load and return immediately
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f).get("items", [])
        except Exception:
            # Cache corrupted -> ignore and refetch
            pass

    # Build REST API URL
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{encoded}/daily/{start_s}/{end_s}"
    )

    # Fetch JSON from API
    js = request_with_backoff(url, params={})
    if not js:
        return []

    # Save whole JSON response to cache so we can reuse later
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False)

    return js.get("items", [])


# -----------------------------
# Feature engineering (summaries)
# -----------------------------
def summarize_views(items: List[dict]) -> Tuple[float, float, float, float]:
    """
    Convert daily pageview 'items' into numeric summary statistics.

    Returns:
      (total, mean, peak, std)
    """
    if not items:
        return (np.nan, np.nan, np.nan, np.nan)

    views = [int(x.get("views", 0)) for x in items]
    if len(views) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    arr = np.array(views, dtype=float)
    return (float(arr.sum()), float(arr.mean()), float(arr.max()), float(arr.std(ddof=0)))


# -----------------------------
# Main program
# -----------------------------
def main() -> None:
    """Entry point: reads inputs, fetches data, writes output CSV."""
    ensure_cache_dir()

    # Load DWTS dataset and dates mapping
    data_df = pd.read_csv(DATA_PATH)
    dates_df = pd.read_csv(DATES_PATH)

    # Compute season windows (start/end dates) and cast lists (names per season)
    season_windows = get_season_windows(dates_df)
    cast_by_season = get_cast_by_season(data_df)

    # Load cached name->title mappings
    title_cache = load_title_cache()

    rows = []  # rows that will become the output CSV

    # Loop through each season window
    for w in season_windows:
        season = w.season

        # Skip seasons not in SEASONS_TO_RUN (if a filter is set)
        if SEASONS_TO_RUN is not None and season not in SEASONS_TO_RUN:
            continue

        # Skip seasons that have no contestants in the main dataset
        if season not in cast_by_season:
            continue

        # Convert timestamps to python date objects
        start_date = w.start.date()
        end_date = w.end.date()

        # Define calendar-year window using year of season start date
        year = start_date.year
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)

        cast = cast_by_season[season]
        print(f"Season {season}: {len(cast)} contestants | season {start_date}..{end_date} | year {year}")

        # Loop through each contestant in the season
        for name in cast:
            # Resolve the Wikipedia title for this name (with caching and disambiguation logic)
            title = resolve_wikipedia_title(name, title_cache)

            # If title can't be found, write a placeholder row so you can diagnose later
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

            # 1) Calendar-year views
            year_items = fetch_pageviews_daily(title, year_start, year_end)
            year_total, year_mean, year_peak, year_std = summarize_views(year_items)

            # 2) Season-window views
            season_items = fetch_pageviews_daily(title, start_date, end_date)
            season_total, season_mean, season_peak, season_std = summarize_views(season_items)

            # 3) Baseline views (pre-season)
            baseline_start = start_date - timedelta(days=BASELINE_DAYS)
            baseline_end = start_date - timedelta(days=1)
            baseline_items = fetch_pageviews_daily(title, baseline_start, baseline_end)
            _, baseline_mean, _, _ = summarize_views(baseline_items)

            # Lift = change in average daily views from baseline to season
            lift = (
                season_mean - baseline_mean
                if (not np.isnan(season_mean) and not np.isnan(baseline_mean))
                else np.nan
            )

            # Write the final row for this contestant-season
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

    # Save outputs to disk
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    # Save title cache so reruns are faster and consistent
    save_title_cache(title_cache)

    print(f"\nWrote {OUT_PATH} with {len(out_df)} rows")
    print(f"Title cache saved to {TITLE_CACHE_PATH}")


if __name__ == "__main__":
    main()
