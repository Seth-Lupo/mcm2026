import requests
import csv
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

CSV_PATH = "data/reddit.csv"
SUBREDDIT = "dancingwiththestars"
csv_lock = threading.Lock()

# Reuse session for connection pooling
session = requests.Session()

def clean_text(text):
    """Remove newlines and normalize whitespace."""
    if not text:
        return ""
    return " ".join(text.split())

def init_csv():
    """Initialize CSV with headers if it doesn't exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["id", "parent_id", "post_id", "date", "title", "text", "comment_level", "karma"])

def append_to_csv(rows):
    """Append rows to CSV (thread-safe)."""
    with csv_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerows(rows)

def download_posts():
    """Download all posts from the subreddit."""
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    before = None
    total = 0

    print("Downloading posts...")

    while True:
        params = {
            "subreddit": SUBREDDIT,
            "limit": 100,
        }
        if before:
            params["before"] = int(before)

        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as e:
            print(f"\nError: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        rows = []
        for post in data:
            post_id = post.get("id", "")
            date = datetime.fromtimestamp(post["created_utc"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            title = clean_text(post.get("title", ""))
            text = clean_text(post.get("selftext", ""))
            karma = post.get("score", 0)
            rows.append([post_id, "", post_id, date, title, text, 0, karma])

        append_to_csv(rows)
        total += len(rows)
        before = data[-1]["created_utc"]

        print(f"Posts: {total}", end="\r")
        time.sleep(0.05)

    print(f"\nPosts complete: {total}")
    return total

def download_comments():
    """Download all comments from the subreddit."""
    url = "https://arctic-shift.photon-reddit.com/api/comments/search"
    before = None
    total = 0

    print("Downloading comments...")

    while True:
        params = {
            "subreddit": SUBREDDIT,
            "limit": 100,
        }
        if before:
            params["before"] = int(before)

        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as e:
            print(f"\nError: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        rows = []
        for comment in data:
            comment_id = comment.get("id", "")
            # parent_id format: "t3_xxx" (post) or "t1_xxx" (comment) - strip prefix
            raw_parent = comment.get("parent_id", "")
            parent_id = raw_parent[3:] if raw_parent else ""
            # link_id is the root post - format: "t3_xxx" - strip prefix
            raw_link = comment.get("link_id", "")
            post_id = raw_link[3:] if raw_link else ""

            date = datetime.fromtimestamp(comment["created_utc"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            title = ""  # Comments don't have titles
            text = clean_text(comment.get("body", ""))

            # Determine depth from parent_id type
            if raw_parent.startswith("t3_"):
                comment_level = 1  # Direct reply to post
            else:
                depth = comment.get("depth", None)
                if depth is not None:
                    comment_level = depth + 1
                else:
                    comment_level = 2  # Assume nested if parent is a comment

            karma = comment.get("score", 0)
            rows.append([comment_id, parent_id, post_id, date, title, text, comment_level, karma])

        append_to_csv(rows)
        total += len(rows)
        before = data[-1]["created_utc"]

        print(f"Comments: {total}", end="\r")
        time.sleep(0.05)

    print(f"\nComments complete: {total}")
    return total

def main():
    init_csv()

    # Run posts and comments downloads in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        post_future = executor.submit(download_posts)
        comment_future = executor.submit(download_comments)

        post_count = post_future.result()
        comment_count = comment_future.result()

    print(f"\n=== DONE ===")
    print(f"Total posts: {post_count}")
    print(f"Total comments: {comment_count}")
    print(f"Total rows: {post_count + comment_count}")
    print(f"Saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
