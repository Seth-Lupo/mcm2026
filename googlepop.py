# dwts_pytrends_anchor.py
#
# Anchor-based Google Trends batching for DWTS contestants.
# - Compares 4 contestants + 1 anchor per query
# - Stitches across batches via contestant/anchor ratios
#
# Requires:
#   pip install pytrends pandas numpy
#
# Inputs (edit paths as needed):
#   2026_MCM_Problem_C_Data.csv  (from COMAP)
#   dates.csv                   (season/week -> date mapping)
#
# Outputs:
#   trends_popularity_by_season.csv

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from pytrends.request import TrendReq


DATA_PATH = "2026_MCM_Problem_C_Data.csv"
DATES_PATH = "dates.csv"

OUT_PATH = "trends_popularity_by_season.csv"
CACHE_DIR = "pytrends_cache"

ANCHOR_TERM = "Dancing with the Stars"
GEO = "US"               # Use "US" for DWTS (U.S. version). Could also use "" for worldwide.
TIMEZONE = 360           # minutes offset; pytrends expects minutes. 360 ~ US Central, but not critical.

# Control rate limiting (pytrends often gets 429 if you go too fast)
SLEEP_MIN = 4.0
SLEEP_MAX = 9.0

# Optional baseline window (pre-season)
USE_BASELINE = True
BASELINE_WEEKS = 8  # weeks prior to season start

# Some celebrity names collide with other meanings.
# Add manual disambiguation where needed:
# e.g. "Mario" -> "Mario (celebrity)" won't work; instead use "Mario Lopez"
# You can maintain a mapping once you notice ambiguous names.
NAME_OVERRIDES: Dict[str, str] = {
    # "Apolo Anton Ohno": "Apolo Ohno",
    # "Joey McIntyre": "Joey McIntyre NKOTB",
}

SEASONS_TO_RUN = {27, 28}


@dataclass
class SeasonWindow:
    season: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def normalize_query_name(name: str) -> str:
    name = str(name).strip()
    return NAME_OVERRIDES.get(name, name)


def get_season_windows(dates_df: pd.DataFrame) -> List[SeasonWindow]:
    d = dates_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    # season start = week 1 date; end = max date in that season
    start = d.loc[d["week"] == 1, ["season", "date"]].rename(columns={"date": "start"})
    end = d.groupby("season", as_index=False)["date"].max().rename(columns={"date": "end"})
    merged = start.merge(end, on="season", how="left").sort_values("season")
    windows = [
        SeasonWindow(int(row["season"]), row["start"], row["end"])
        for _, row in merged.iterrows()
    ]
    return windows


def get_cast_by_season(data_df: pd.DataFrame) -> Dict[int, List[str]]:
    cast = (
        data_df.groupby("season")["celebrity_name"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    # Normalize names + drop empties
    out = {}
    for season, names in cast.items():
        clean = [normalize_query_name(n) for n in names if str(n).strip()]
        out[int(season)] = clean
    return out


def chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def cache_key(season: int, kw_list: List[str], timeframe: str, geo: str) -> str:
    safe = "_".join([str(season)] + [k.replace(" ", "_").replace("/", "_") for k in kw_list])
    safe = safe[:180]  # avoid huge filenames
    return os.path.join(CACHE_DIR, f"{safe}__{timeframe}__{geo}.csv")


def fetch_interest_over_time(pytrends: TrendReq, kw_list: List[str], timeframe: str, geo: str) -> pd.DataFrame:
    """
    Fetch interest_over_time for kw_list.
    Returns DataFrame indexed by date with columns for each term + isPartial.
    """
    pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo=geo)
    iot = pytrends.interest_over_time()
    if iot is None or iot.empty:
        return pd.DataFrame()
    return iot


def load_or_fetch(pytrends: TrendReq, season: int, kw_list: List[str], timeframe: str, geo: str) -> pd.DataFrame:
    ensure_cache_dir()
    ck = cache_key(season, kw_list, timeframe, geo)
    if os.path.exists(ck):
        try:
            return pd.read_csv(ck, parse_dates=["date"]).set_index("date")
        except Exception:
            pass  # fall through and refetch if cache corrupted

    # Rate limiting / politeness sleep
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    print("done sleeping!")
    iot = fetch_interest_over_time(pytrends, kw_list=kw_list, timeframe=timeframe, geo=geo)
    if not iot.empty:
        out = iot.copy()
        out.index.name = "date"
        out.reset_index().to_csv(ck, index=False)
    return iot


def compute_ratios(iot: pd.DataFrame, contestants: List[str], anchor: str) -> pd.DataFrame:
    """
    Convert interest series to contestant/anchor ratios for each date.
    Handles zeros by returning NaN where anchor is 0.
    """
    df = iot.copy()
    if df.empty:
        return df
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    # Ensure anchor exists
    if anchor not in df.columns:
        return pd.DataFrame()

    anchor_series = df[anchor].replace({0: np.nan}).astype(float)
    ratios = {}
    for c in contestants:
        if c in df.columns:
            ratios[c] = df[c].astype(float) / anchor_series
    return pd.DataFrame(ratios, index=df.index)


def summarize_series(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {"mean": np.nan, "max": np.nan, "std": np.nan}
    return {"mean": float(s.mean()), "max": float(s.max()), "std": float(s.std(ddof=0))}


def main() -> None:
    data_df = pd.read_csv(DATA_PATH)
    dates_df = pd.read_csv(DATES_PATH)

    season_windows = get_season_windows(dates_df)
    cast_by_season = get_cast_by_season(data_df)

    pytrends = TrendReq(hl="en-US", tz=TIMEZONE)

    rows = []
    for w in season_windows:
        season = w.season
        if SEASONS_TO_RUN is not None and season not in SEASONS_TO_RUN:
                continue

        contestants = cast_by_season[season]
        if not contestants:
            continue

        # Season timeframe
        start = w.start_date.strftime("%Y-%m-%d")
        end = w.end_date.strftime("%Y-%m-%d")
        season_timeframe = f"{start} {end}"

        # Optional baseline timeframe (pre-season)
        if USE_BASELINE:
            baseline_start = (w.start_date - pd.Timedelta(weeks=BASELINE_WEEKS)).strftime("%Y-%m-%d")
            baseline_end = (w.start_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            baseline_timeframe = f"{baseline_start} {baseline_end}"
        else:
            baseline_timeframe = None

        # Batch into 4 contestants + anchor
        for batch in chunks(contestants, 4):
            kw_list = batch + [ANCHOR_TERM]

            # Fetch season window interest
            iot_season = load_or_fetch(pytrends, season=season, kw_list=kw_list, timeframe=season_timeframe, geo=GEO)
            ratios_season = compute_ratios(iot_season, contestants=batch, anchor=ANCHOR_TERM)

            # Fetch baseline window interest (optional)
            if baseline_timeframe:
                iot_base = load_or_fetch(pytrends, season=season, kw_list=kw_list, timeframe=baseline_timeframe, geo=GEO)
                ratios_base = compute_ratios(iot_base, contestants=batch, anchor=ANCHOR_TERM)
            else:
                ratios_base = None

            # Summarize per contestant
            for c in batch:
                season_stats = summarize_series(ratios_season[c]) if (not ratios_season.empty and c in ratios_season.columns) else {"mean": np.nan, "max": np.nan, "std": np.nan}

                if ratios_base is not None and (not ratios_base.empty) and c in ratios_base.columns:
                    base_stats = summarize_series(ratios_base[c])
                else:
                    base_stats = {"mean": np.nan, "max": np.nan, "std": np.nan}

                # Lift is the difference in mean ratio season vs baseline
                lift = season_stats["mean"] - base_stats["mean"] if (not np.isnan(season_stats["mean"]) and not np.isnan(base_stats["mean"])) else np.nan

                rows.append({
                    "season": season,
                    "celebrity_name": c,
                    "season_start": start,
                    "season_end": end,
                    "geo": GEO,
                    "anchor": ANCHOR_TERM,
                    "season_ratio_mean": season_stats["mean"],
                    "season_ratio_max": season_stats["max"],
                    "season_ratio_std": season_stats["std"],
                    "baseline_ratio_mean": base_stats["mean"],
                    "baseline_ratio_max": base_stats["max"],
                    "baseline_ratio_std": base_stats["std"],
                    "lift_mean_ratio": lift,
                })

    out_df = pd.DataFrame(rows)

    # If a celebrity appears multiple times due to name overrides or duplicates,
    # you can aggregate; for now, keep first non-null per season/name.
    out_df = out_df.sort_values(["season", "celebrity_name"]).reset_index(drop=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
