#!/usr/bin/env python3
"""
Scrape per-episode viewership (Viewers in millions) from Wikipedia season pages
for Dancing with the Stars (American TV series), seasons 1-34.

Outputs:
  - dwts_episode_viewership.txt  (tab-separated)
  - dwts_episode_viewership.csv

Notes:
  - Wikipedia season pages commonly include a "Ratings" table with "Viewers (millions)".
    Example: season 1 has a per-episode ratings table including viewers in millions.
  - Table schemas vary by season (some have "No.", some have "Show"/"Episode", some include specials).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import requests


WIKI_SEASON_URL = "https://en.wikipedia.org/wiki/Dancing_with_the_Stars_%28American_TV_series%29_season_{}"
OUT_TXT = "dwts_episode_viewership.txt"
OUT_CSV = "dwts_episode_viewership.csv"


@dataclass
class EpisodeRow:
    season: int
    episode_label: str
    air_date: str
    viewers_millions: float


def _normalize_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def _find_viewers_col(cols: List[str]) -> Optional[str]:
    """
    Return the best-matching column name for viewers in millions.
    We accept various forms like:
      - 'Viewers (millions)'
      - 'Viewers' (sometimes already numeric)
    """
    normalized = {_normalize_colname(c): c for c in cols}

    # Prefer explicit "viewers" + "million"
    for n, orig in normalized.items():
        if "viewer" in n and "million" in n:
            return orig

    # Fall back to "viewers"
    for n, orig in normalized.items():
        if n == "viewers" or n.startswith("viewers"):
            return orig

    return None


def _find_airdate_col(cols: List[str]) -> Optional[str]:
    normalized = {_normalize_colname(c): c for c in cols}
    for n, orig in normalized.items():
        if n in ("air date", "original air date", "date", "first aired"):
            return orig
    # some seasons use "Airdate"
    for n, orig in normalized.items():
        if "air" in n and "date" in n:
            return orig
    return None


def _find_episode_id_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to identify columns that define the episode label.
    Common patterns:
      - 'No.' + 'Title'
      - 'Episode' + 'Show'
      - 'No.' alone
      - 'Title' alone
    Returns: (no_col, title_col, episode_col)
    """
    normalized = {_normalize_colname(c): c for c in cols}

    no_col = None
    title_col = None
    episode_col = None

    for n, orig in normalized.items():
        if n in ("no.", "no", "#", "overall", "episode no.", "episode number"):
            no_col = orig
        if n in ("title", "episode", "episode title"):
            # careful: some tables have a column literally named "Episode"
            # but it's like "Performance Show: Week 1"
            title_col = orig
        if n == "episode":
            episode_col = orig

    # If we picked the same column for title_col and episode_col, that's OK.
    return no_col, title_col, episode_col


def _coerce_viewers(val) -> Optional[float]:
    """
    Convert a cell like '13.48[10]' or '13.48' to float.
    Return None if it can't be parsed.
    """
    if pd.isna(val):
        return None
    s = str(val)

    # Remove bracketed refs like [10]
    s = re.sub(r"\[[^\]]*\]", "", s)

    # Remove footnotes/extra text
    s = s.strip()

    # Some pages have "TBA" or "—"
    if s in ("", "—", "-", "tba", "n/a", "na"):
        return None

    # Extract first float-looking number
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _pick_ratings_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Identify the ratings table by presence of a viewers column.
    If multiple candidates exist, pick the one with the most rows.
    """
    candidates = []
    for df in tables:
        viewers_col = _find_viewers_col(list(df.columns))
        if viewers_col is None:
            continue
        # Must have at least a few rows to be meaningful
        if len(df) < 2:
            continue
        candidates.append(df)

    if not candidates:
        return None

    # Pick the largest table (most rows)
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def scrape_season(season_num: int, session: requests.Session) -> List[EpisodeRow]:
    url = WIKI_SEASON_URL.format(season_num)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    # pandas read_html can parse Wikipedia tables directly from HTML text
    tables = pd.read_html(resp.text)
    ratings = _pick_ratings_table(tables)
    if ratings is None:
        raise RuntimeError(f"No ratings table found for season {season_num} at {url}")

    viewers_col = _find_viewers_col(list(ratings.columns))
    if viewers_col is None:
        raise RuntimeError(f"Ratings table found but no viewers column detected for season {season_num}")

    airdate_col = _find_airdate_col(list(ratings.columns))
    no_col, title_col, episode_col = _find_episode_id_cols(list(ratings.columns))

    rows: List[EpisodeRow] = []
    for i, r in ratings.iterrows():
        viewers = _coerce_viewers(r.get(viewers_col))
        if viewers is None:
            # Keep rows with missing viewers out (common in incomplete current-season tables)
            continue

        # Build an episode label
        pieces = []

        # Many seasons have "No." as numeric episode index; some have "Special"
        if no_col and no_col in ratings.columns:
            no_val = r.get(no_col)
            if not pd.isna(no_val):
                pieces.append(str(no_val).strip())

        # Title / Episode naming can vary (e.g., "Episode 101" vs "Performance Show: Week 1")
        label_val = None
        if title_col and title_col in ratings.columns:
            label_val = r.get(title_col)
        elif episode_col and episode_col in ratings.columns:
            label_val = r.get(episode_col)

        if label_val is not None and not pd.isna(label_val):
            pieces.append(str(label_val).strip())

        episode_label = " | ".join([p for p in pieces if p]) or f"Row {i+1}"

        air_date = ""
        if airdate_col and airdate_col in ratings.columns:
            av = r.get(airdate_col)
            if av is not None and not pd.isna(av):
                air_date = str(av).strip()

        rows.append(EpisodeRow(
            season=season_num,
            episode_label=episode_label,
            air_date=air_date,
            viewers_millions=viewers
        ))

    return rows


def main() -> None:
    session = requests.Session()
    session.headers.update({
        # Be polite; Wikipedia sometimes blocks empty/robotic UAs
        "User-Agent": "DWTSViewershipScraper/1.0 (educational; contact: none)"
    })

    all_rows: List[EpisodeRow] = []

    for s in range(1, 35):
        try:
            print(f"Scraping season {s}...")
            season_rows = scrape_season(s, session)
            print(f"  found {len(season_rows)} episode rows with viewers")
            all_rows.extend(season_rows)
        except Exception as e:
            print(f"  WARNING: season {s} failed: {e}")
        # small delay to be courteous
        time.sleep(0.75)

    # Sort for stable output
    all_rows.sort(key=lambda x: (x.season, x.episode_label))

    # Write TXT (tab-separated)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Season\tEpisodeLabel\tAirDate\tViewersMillions\n")
        for r in all_rows:
            f.write(f"{r.season}\t{r.episode_label}\t{r.air_date}\t{r.viewers_millions:.2f}\n")

    # Write CSV
    df = pd.DataFrame([{
        "Season": r.season,
        "EpisodeLabel": r.episode_label,
        "AirDate": r.air_date,
        "ViewersMillions": r.viewers_millions
    } for r in all_rows])
    df.to_csv(OUT_CSV, index=False)

    print(f"\nDone.\nWrote: {OUT_TXT}\nWrote: {OUT_CSV}\nTotal rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
