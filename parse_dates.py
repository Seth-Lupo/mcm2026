import pandas as pd
import requests
from io import StringIO
import re
import html
from datetime import datetime

CSV_URL = "https://epguides.com/common/exportToCSVmaze.asp?maze=71"

def fetch_csv():
    """Fetch CSV from epguides (wrapped in HTML pre tag)."""
    resp = requests.get(CSV_URL)
    resp.raise_for_status()
    text = resp.text
    start = text.find('<pre>') + 5
    end = text.find('</pre>')
    csv_text = text[start:end].strip()
    return pd.read_csv(StringIO(csv_text))

def parse_date(date_str):
    """Parse date string like '01 Jun 05' to datetime."""
    try:
        return datetime.strptime(date_str.strip(), "%d %b %y")
    except:
        return None

def is_results_show(title):
    """Check if episode is a results show (not a performance)."""
    t = title.lower()
    if 'result' in t:
        return True
    if 'recap' in t:
        return True
    if 'winner' in t and 'announced' in t:
        return True
    return False

def is_special(title):
    """Check if episode is a special (not regular competition)."""
    t = title.lower()
    specials = ['dance off', 'all-time', 'holiday', 'holidays', 'christmas']
    return any(s in t for s in specials)

def get_week_from_title(title):
    """Extract week number from title if present."""
    t = title.lower()

    # "Week X" or "Week #X"
    m = re.search(r'week\s*#?\s*(\d+)', t)
    if m:
        return int(m.group(1))

    # "Round X" (numeric)
    m = re.search(r'round\s+(\d+)', t)
    if m:
        return int(m.group(1))

    # "Round One/Two/etc"
    words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    m = re.search(r'round\s+(\w+)', t)
    if m and m.group(1) in words:
        return words[m.group(1)]

    return None

def process_season(season_df):
    """Process a single season's episodes to get week dates."""
    # Sort by episode number
    eps = season_df.sort_values('episode').reset_index(drop=True)

    results = {}
    week_counter = 0

    for _, row in eps.iterrows():
        title = html.unescape(row['title'])

        # Skip results shows and specials
        if is_results_show(title) or is_special(title):
            continue

        # Try to get week from title
        week = get_week_from_title(title)

        if week is None:
            # Increment counter for performance shows
            week_counter += 1
            week = week_counter
        else:
            week_counter = week

        # Only keep weeks 1-11, first occurrence
        if 1 <= week <= 11 and week not in results:
            results[week] = {
                'date': parse_date(row['airdate']),
                'title': title
            }

    return results

def main():
    print("Fetching CSV from epguides...")
    raw_df = fetch_csv()
    print(f"Fetched {len(raw_df)} episodes")

    # Build the matrix
    rows = []
    for season in range(1, 35):
        season_df = raw_df[raw_df['season'] == season]

        if len(season_df) == 0:
            # No episodes for this season
            week_data = {}
        else:
            week_data = process_season(season_df)

        for week in range(1, 12):
            if week in week_data:
                d = week_data[week]
                date_str = d['date'].strftime('%Y-%m-%d') if d['date'] else ''
                rows.append({
                    'season': season,
                    'week': week,
                    'date': date_str,
                    'title': d['title']
                })
            else:
                rows.append({
                    'season': season,
                    'week': week,
                    'date': '',
                    'title': ''
                })

    df = pd.DataFrame(rows)
    df.to_csv('data/dates.csv', index=False, na_rep='')
    print(f"Saved to data/dates.csv ({len(df)} rows)")

    # Show sample
    print("\nSample (seasons 1-3):")
    sample = df[(df['season'] <= 3) & (df['date'] != '')]
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
