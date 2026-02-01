#!/usr/bin/env python3
"""
Build (person, season, week) -> reddit mention counts dataset.

Scans reddit posts/comments and counts mentions of each contestant
during their season's weeks.

Mention scoring:
- Top-level post/comment (comment_level=1): counts as 1
- Nested replies (comment_level>1): each counts as 1

Output: data/reddit_mentions.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from rapidfuzz import fuzz
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_dates() -> pd.DataFrame:
    """Load season/week dates."""
    df = pd.read_csv(DATA_DIR / "dates.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_contestants() -> pd.DataFrame:
    """Load contestant info."""
    df = pd.read_csv(DATA_DIR / "main.csv")
    return df[["season", "celebrity_name"]].drop_duplicates()


def load_reddit() -> pd.DataFrame:
    """Load reddit data."""
    log.info("Loading reddit data...")
    df = pd.read_csv(DATA_DIR / "reddit.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["comment_level"] = pd.to_numeric(df["comment_level"], errors="coerce").fillna(1).astype(int)
    df["karma"] = pd.to_numeric(df["karma"], errors="coerce").fillna(0).astype(int)
    log.info(f"Loaded {len(df):,} reddit rows")
    return df


def build_week_intervals(dates_df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[datetime, datetime]]:
    """
    Build mapping of (season, week) -> (start_date, end_date).

    Week N runs from date[N] to date[N+1] - 1 day.
    """
    intervals = {}

    for season in dates_df["season"].unique():
        sdf = dates_df[dates_df["season"] == season].sort_values("week")
        valid = sdf[sdf["date"].notna()]

        if valid.empty:
            continue

        weeks = valid["week"].tolist()
        dates = valid["date"].tolist()

        for i, (week, start) in enumerate(zip(weeks, dates)):
            if i + 1 < len(dates):
                end = dates[i + 1] - timedelta(days=1)
            else:
                # Last week: give it 7 days
                end = start + timedelta(days=6)

            intervals[(season, week)] = (start, end)

    return intervals


def get_name_variants(name: str) -> List[str]:
    """Generate search variants for a name."""
    variants = [name.lower()]

    parts = name.split()
    if len(parts) >= 2:
        # First name, last name
        variants.append(parts[0].lower())
        variants.append(parts[-1].lower())
        # First + Last (no space)
        variants.append(f"{parts[0]}{parts[-1]}".lower())

    # Handle special cases
    name_lower = name.lower()

    # Common nicknames/shortenings
    if "bobby" in name_lower:
        variants.append("bobby")
    if "bristol" in name_lower:
        variants.append("bristol")
    if "spicer" in name_lower or "spicer" in name_lower:
        variants.append("spicer")
        variants.append("sean spicer")

    return list(set(variants))


def mentions_person(text: str, name: str, variants: List[str], threshold: int = 85) -> bool:
    """
    Check if text mentions a person.

    Uses fuzzy matching for full name and exact matching for name parts.
    """
    if pd.isna(text) or not text:
        return False

    text_lower = text.lower()

    # Check full name with fuzzy matching
    if fuzz.partial_ratio(name.lower(), text_lower) >= threshold:
        return True

    # Check variants (exact substring match)
    for variant in variants:
        # Only match if variant is substantial (>3 chars) or is the full name
        if len(variant) > 3 and variant in text_lower:
            # Avoid false positives: check word boundaries
            pattern = r'\b' + re.escape(variant) + r'\b'
            if re.search(pattern, text_lower):
                return True

    return False


def build_mentions_dataset(
    reddit_df: pd.DataFrame,
    contestants_df: pd.DataFrame,
    week_intervals: Dict[Tuple[int, int], Tuple[datetime, datetime]],
) -> pd.DataFrame:
    """
    Build the (person, season, week) -> mentions dataset.
    """
    log.info("Building mentions dataset...")

    # Precompute name variants
    name_variants = {}
    for _, row in contestants_df.iterrows():
        name = row["celebrity_name"]
        name_variants[name] = get_name_variants(name)

    # Group reddit by date for faster filtering
    reddit_df = reddit_df.sort_values("date")

    results = []
    total_combos = 0

    for season in sorted(contestants_df["season"].unique()):
        season_contestants = contestants_df[contestants_df["season"] == season]["celebrity_name"].tolist()
        season_weeks = [(s, w) for (s, w) in week_intervals.keys() if s == season]

        if not season_weeks:
            continue

        log.info(f"Processing season {season}: {len(season_contestants)} contestants, {len(season_weeks)} weeks")

        for week_key in sorted(season_weeks):
            _, week = week_key
            start_date, end_date = week_intervals[week_key]

            # Filter reddit to this week
            mask = (reddit_df["date"] >= start_date) & (reddit_df["date"] <= end_date)
            week_reddit = reddit_df[mask]

            if week_reddit.empty:
                # No reddit data for this week
                for name in season_contestants:
                    results.append({
                        "season": season,
                        "week": week,
                        "celebrity_name": name,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "mentions": 0,
                        "karma": 0,
                        "top_level_mentions": 0,
                        "top_level_karma": 0,
                        "reply_mentions": 0,
                        "reply_karma": 0,
                        "reddit_posts": 0,
                    })
                continue

            # Combine title + text for searching
            week_reddit = week_reddit.copy()
            week_reddit["full_text"] = week_reddit["title"].fillna("") + " " + week_reddit["text"].fillna("")

            for name in season_contestants:
                variants = name_variants[name]

                # Count mentions and karma
                top_level = 0
                top_level_karma = 0
                replies = 0
                reply_karma = 0

                for _, reddit_row in week_reddit.iterrows():
                    if mentions_person(reddit_row["full_text"], name, variants):
                        k = reddit_row["karma"]
                        if reddit_row["comment_level"] == 1:
                            top_level += 1
                            top_level_karma += k
                        else:
                            replies += 1
                            reply_karma += k

                results.append({
                    "season": season,
                    "week": week,
                    "celebrity_name": name,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "mentions": top_level + replies,
                    "karma": top_level_karma + reply_karma,
                    "top_level_mentions": top_level,
                    "top_level_karma": top_level_karma,
                    "reply_mentions": replies,
                    "reply_karma": reply_karma,
                    "reddit_posts": len(week_reddit),
                })
                total_combos += 1

        log.info(f"  Processed {total_combos} (person, week) combinations so far")

    return pd.DataFrame(results)


def main():
    """Build and save reddit mentions dataset."""
    log.info("Loading data...")
    dates_df = load_dates()
    contestants_df = load_contestants()
    reddit_df = load_reddit()

    log.info("Building week intervals...")
    week_intervals = build_week_intervals(dates_df)
    log.info(f"Found {len(week_intervals)} (season, week) intervals")

    # Filter to seasons where we have reddit data (reddit started ~2005, popular later)
    reddit_min_date = reddit_df["date"].min()
    reddit_max_date = reddit_df["date"].max()
    log.info(f"Reddit date range: {reddit_min_date} to {reddit_max_date}")

    # Filter intervals to reddit date range
    valid_intervals = {
        k: v for k, v in week_intervals.items()
        if v[0] >= reddit_min_date - timedelta(days=30)
    }
    log.info(f"Filtered to {len(valid_intervals)} intervals within reddit date range")

    # Filter contestants to those seasons
    valid_seasons = set(k[0] for k in valid_intervals.keys())
    contestants_df = contestants_df[contestants_df["season"].isin(valid_seasons)]
    log.info(f"Processing {len(contestants_df)} contestants across {len(valid_seasons)} seasons")

    # Build the dataset
    mentions_df = build_mentions_dataset(reddit_df, contestants_df, valid_intervals)

    # Save
    out_path = DATA_DIR / "reddit_mentions.csv"
    mentions_df.to_csv(out_path, index=False)
    log.info(f"Saved {len(mentions_df)} rows to {out_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total (person, week) rows: {len(mentions_df)}")
    print(f"Rows with mentions > 0: {(mentions_df['mentions'] > 0).sum()}")
    print(f"Total mentions: {mentions_df['mentions'].sum():,}")
    print(f"Total karma: {mentions_df['karma'].sum():,}")
    print(f"\nBy season:")
    by_season = mentions_df.groupby("season").agg({"mentions": "sum", "karma": "sum"}).sort_index()
    for season, row in by_season.iterrows():
        print(f"  S{season}: {row['mentions']:,} mentions, {row['karma']:,} karma")


if __name__ == "__main__":
    main()
