"""
wikipop_halves_placement_judgescore.py

Adds to the Wikipedia pageviews pipeline:
1) Half-season pageview metrics (first half vs second half)
2) Placement column (from `placement`)
3) Average season judge score column (computed from week*_judge*_score columns)
   - Added as the LAST column in the output CSV (after placement).

Output CSV:
  wiki_popularity_by_season.csv

Inputs:
  2026_MCM_Problem_C_Data.csv  (must include: season, celebrity_name, placement, week*_judge*_score columns)
  dates.csv                   (must include: season, week, date)

Install:
  pip install pandas requests numpy
Run:
  python wikipop_halves_placement_judgescore.py
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

import re


# -----------------------------
# File paths (adjust if needed)
# -----------------------------
DATA_PATH = "2026_MCM_Problem_C_Data.csv"
DATES_PATH = "dates.csv"
OUT_PATH = "wiki_popularity_by_season.csv"

# -----------------------------
# Cache settings
# -----------------------------
CACHE_DIR = "cache"
TITLE_CACHE_PATH = os.path.join(CACHE_DIR, "wiki_title_cache.json")

# ------------------------------------
# Season selection (None = all seasons)
# ------------------------------------
SEASONS_TO_RUN: Optional[set[int]] = None  # e.g., {27, 28}

# -----------------------------
# Baseline window configuration
# -----------------------------
BASELINE_DAYS = 28

# -----------------------------
# Polite pacing and retries
# -----------------------------
SLEEP_MIN = 0.05
SLEEP_MAX = 0.15
MAX_RETRIES = 6

HEADERS = {
    "User-Agent": "MCM2026-DWTS-WikiPageviews/1.0 (academic project)"
}

# -----------------------------
# Manual overrides for ambiguous names
# -----------------------------
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


# -----------------------------
# Cache folder utilities
# -----------------------------
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


# -----------------------------
# Networking utilities
# -----------------------------
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

            return None
        except requests.RequestException:
            sleep_for = delay + random.uniform(0, delay * 0.25)
            time.sleep(sleep_for)
            delay = min(delay * 2, 60)
    return None


# -----------------------------
# Input parsing helpers
# -----------------------------
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


def get_placement_by_season(data_df: pd.DataFrame) -> Dict[tuple[int, str], Optional[int]]:
    """
    Build (season, celebrity_name) -> placement using mode across rows (in case of duplicates).
    """
    placements: Dict[tuple[int, str], Optional[int]] = {}
    for (season, name), g in data_df.groupby(["season", "celebrity_name"], dropna=False):
        vals = g["placement"].dropna().tolist()
        if not vals:
            placements[(int(season), str(name))] = None
            continue
        s = pd.Series(vals)
        mode = s.mode()
        chosen = mode.iloc[0] if not mode.empty else vals[0]
        try:
            placements[(int(season), str(name))] = int(chosen)
        except Exception:
            placements[(int(season), str(name))] = None
    return placements


import re
import numpy as np
import pandas as pd

def get_avg_judge_score_by_season(data_df: pd.DataFrame) -> dict[tuple[int, str], float | None]:
    score_cols = [c for c in data_df.columns if re.fullmatch(r"week\d+_judge\d+_score", c)]
    if not score_cols:
        raise ValueError("No judge score columns found matching pattern week<k>_judge<j>_score")

    avg_map: dict[tuple[int, str], float | None] = {}

    for (season, name), g in data_df.groupby(["season", "celebrity_name"], dropna=False):
        # Convert to numeric; non-numeric becomes NaN
        scores = g[score_cols].apply(pd.to_numeric, errors="coerce")

        # KEY FIX: treat 0 as "missing because eliminated" (or otherwise absent)
        scores = scores.mask(scores == 0, np.nan)

        # Flatten and ignore NaNs
        flat = scores.to_numpy().ravel()
        flat = flat[~np.isnan(flat)]

        avg_map[(int(season), str(name))] = float(np.mean(flat)) if flat.size else None

    return avg_map



# -----------------------------
# Wikipedia title resolution
# -----------------------------
def is_disambiguation(title: str) -> bool:
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

    for r in results:
        title = r.get("title")
        if not title:
            continue
        if not is_disambiguation(title):
            title_cache[name] = title.replace(" ", "_")
            return title_cache[name]

    title = results[0].get("title")
    if title:
        title_cache[name] = title.replace(" ", "_")
        return title_cache[name]
    return None


# -----------------------------
# Wikipedia pageviews download
# -----------------------------
def cache_path_for_views(title: str, granularity: str, start_yyyymmdd: str, end_yyyymmdd: str) -> str:
    safe_title = title.replace("/", "_")
    return os.path.join(CACHE_DIR, f"views__{safe_title}__{granularity}__{start_yyyymmdd}_{end_yyyymmdd}.json")


def fetch_pageviews_daily(title: str, start_dt: date, end_dt: date) -> List[dict]:
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


# -----------------------------
# Feature engineering
# -----------------------------
def summarize_views(items: List[dict]) -> Tuple[float, float, float, float]:
    if not items:
        return (np.nan, np.nan, np.nan, np.nan)

    views = [int(x.get("views", 0)) for x in items]
    if not views:
        return (np.nan, np.nan, np.nan, np.nan)

    arr = np.array(views, dtype=float)
    return (float(arr.sum()), float(arr.mean()), float(arr.max()), float(arr.std(ddof=0)))


def split_season_into_halves(start_date: date, end_date: date) -> Tuple[Tuple[date, date], Tuple[date, date]]:
    """
    Split inclusive date range [start_date, end_date] into two halves by days.

    Let N = number of days in range (inclusive).
    - first half length = ceil(N/2)
    - second half length = floor(N/2)
    """
    n_days = (end_date - start_date).days + 1
    first_len = (n_days + 1) // 2  # ceil(N/2)
    first_start = start_date
    first_end = start_date + timedelta(days=first_len - 1)
    second_start = first_end + timedelta(days=1)
    second_end = end_date
    return (first_start, first_end), (second_start, second_end)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_cache_dir()

    data_df = pd.read_csv(DATA_PATH)
    dates_df = pd.read_csv(DATES_PATH)

    season_windows = get_season_windows(dates_df)
    cast_by_season = get_cast_by_season(data_df)

    # Precompute mappings once
    placement_map = get_placement_by_season(data_df)
    avg_judge_score_map = get_avg_judge_score_by_season(data_df)

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
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)

        (h1_start, h1_end), (h2_start, h2_end) = split_season_into_halves(start_date, end_date)

        cast = cast_by_season[season]
        print(
            f"Season {season}: {len(cast)} contestants | "
            f"season {start_date}..{end_date} | "
            f"half1 {h1_start}..{h1_end} | half2 {h2_start}..{h2_end}"
        )

        for name in cast:
            title = resolve_wikipedia_title(name, title_cache)

            placement = placement_map.get((season, name))
            avg_judge_score = avg_judge_score_map.get((season, name))

            if not title:
                rows.append({
                    "season": season,
                    "celebrity_name": name,
                    "wiki_title": None,
                    "year": year,
                    "year_total": np.nan, "year_mean": np.nan, "year_peak": np.nan, "year_std": np.nan,
                    "season_total": np.nan, "season_mean": np.nan, "season_peak": np.nan, "season_std": np.nan,
                    "half1_total": np.nan, "half1_mean": np.nan, "half1_peak": np.nan, "half1_std": np.nan,
                    "half2_total": np.nan, "half2_mean": np.nan, "half2_peak": np.nan, "half2_std": np.nan,
                    "baseline_mean": np.nan,
                    "lift": np.nan,
                    "status": "no_title",
                    "placement": placement,
                    "avg_season_judge_score": avg_judge_score,
                })
                continue

            # Calendar-year views
            year_items = fetch_pageviews_daily(title, year_start, year_end)
            year_total, year_mean, year_peak, year_std = summarize_views(year_items)

            # Full season views
            season_items = fetch_pageviews_daily(title, start_date, end_date)
            season_total, season_mean, season_peak, season_std = summarize_views(season_items)

            # Half 1 views
            half1_items = fetch_pageviews_daily(title, h1_start, h1_end)
            half1_total, half1_mean, half1_peak, half1_std = summarize_views(half1_items)

            # Half 2 views
            half2_items = fetch_pageviews_daily(title, h2_start, h2_end) if h2_start <= h2_end else []
            half2_total, half2_mean, half2_peak, half2_std = summarize_views(half2_items)

            # Baseline mean (pre-season)
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
                "year_total": year_total, "year_mean": year_mean, "year_peak": year_peak, "year_std": year_std,
                "season_total": season_total, "season_mean": season_mean, "season_peak": season_peak, "season_std": season_std,
                "half1_total": half1_total, "half1_mean": half1_mean, "half1_peak": half1_peak, "half1_std": half1_std,
                "half2_total": half2_total, "half2_mean": half2_mean, "half2_peak": half2_peak, "half2_std": half2_std,
                "baseline_mean": baseline_mean,
                "lift": lift,
                "status": "ok",
                "placement": placement,
                "avg_season_judge_score": avg_judge_score,
            })

    out_df = pd.DataFrame(rows)

    # Reorder columns so:
    #   placement is second-to-last
    #   avg_season_judge_score is last
    if "placement" in out_df.columns and "avg_season_judge_score" in out_df.columns:
        cols = [c for c in out_df.columns if c not in ("placement", "avg_season_judge_score")]
        cols = cols + ["placement", "avg_season_judge_score"]
        out_df = out_df[cols]

    out_df.to_csv(OUT_PATH, index=False)
    save_title_cache(title_cache)

    print(f"\nWrote {OUT_PATH} with {len(out_df)} rows")
    print(f"Title cache saved to {TITLE_CACHE_PATH}")


if __name__ == "__main__":
    main()
