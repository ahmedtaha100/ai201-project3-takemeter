#!/usr/bin/env python3
"""
Collect public r/nba comments via Reddit's public JSON endpoints — no API key, no PRAW.

WHY THIS IS A SEPARATE STEP YOU RUN: Reddit blocks data-center / cloud egress IPs (HTTP 403)
regardless of User-Agent, so this could not be run from the build environment. Run it once
from a normal residential connection (your Mac) and it works with a polite delay.

    python src/collect.py --target 400 --out data/raw_comments.csv

It pulls from a MIX of sources so the eventual label distribution can be balanced:
  - r/nba/comments.json            (recent flat comments across the sub — bulk, reaction-heavy)
  - top posts' comment threads     (Post Game Threads + Daily/Discussion threads -> hot takes
                                    and analysis), to make sure `analysis` clears 20%.

It filters out [deleted]/[removed], bot/automod, very short or link-only comments, dedupes,
and writes a CSV of raw (unlabeled) comments. Labeling happens in src/label.py.

stdlib only — nothing to install.
"""
import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.request

UA = "takemeter-research/0.1 (CodePath AI201 student project; contact: you@example.com)"
SLEEP = 2.5  # be polite to Reddit's unauthenticated rate limit


def _get(url: str, retries: int = 3):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code == 429:
                time.sleep(SLEEP * (attempt + 2))
                continue
            if attempt == retries - 1:
                print(f"  ! request failed ({type(e).__name__} {code}): {url}", file=sys.stderr)
                return None
            time.sleep(SLEEP)
    return None


BOT_MARKERS = ("i am a bot", "^(i am a bot", "performed automatically", "AutoModerator")
JUNK = {"[deleted]", "[removed]", ""}


def clean(body: str) -> str:
    body = html.unescape(body or "")
    body = re.sub(r"https?://\S+", "", body)          # strip URLs
    body = re.sub(r"/?u/\w+|/?r/\w+", "", body)        # strip user/sub mentions
    body = re.sub(r"&gt;.*", "", body)                 # strip quote lines
    body = re.sub(r"\s+", " ", body).strip()
    return body


def usable(body: str) -> bool:
    if body in JUNK:
        return False
    low = body.lower()
    if any(m.lower() in low for m in BOT_MARKERS):
        return False
    words = body.split()
    if len(words) < 3 or len(body) < 12:    # too short to carry discourse
        return False
    if len(body) > 1200:                     # essays are rare; keep it comment-sized
        return False
    if not re.search(r"[a-zA-Z]", body):
        return False
    return True


def collect_flat(subreddit: str, n: int):
    """Recent comments across the whole subreddit, paginated."""
    out, after, calls = {}, None, 0
    while len(out) < n and calls < 40:
        url = f"https://www.reddit.com/r/{subreddit}/comments.json?limit=100&raw_json=1"
        if after:
            url += f"&after={after}"
        data = _get(url)
        calls += 1
        if not data:
            break
        children = data.get("data", {}).get("children", [])
        if not children:
            break
        for c in children:
            d = c.get("data", {})
            body = clean(d.get("body", ""))
            if usable(body):
                out[d.get("id")] = {
                    "id": d.get("id"), "text": body, "score": d.get("score", 0),
                    "permalink": d.get("permalink", ""), "thread": d.get("link_title", ""),
                    "source": "flat_comments",
                }
        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(SLEEP)
    return out


def collect_threads(subreddit: str, n: int):
    """Top-level comments from discussion-style threads (analysis/hot-take rich)."""
    out = {}
    # Pull hot + top posts, prefer Post Game / Daily / Discussion threads.
    for sort, extra in (("hot", ""), ("top", "&t=week")):
        listing = _get(
            f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit=50&raw_json=1{extra}"
        )
        time.sleep(SLEEP)
        if not listing:
            continue
        posts = listing.get("data", {}).get("children", [])
        wanted = [
            p["data"] for p in posts
            if any(k in (p["data"].get("title", "").lower())
                   for k in ("post game", "game thread", "daily", "discussion", "thoughts", "[serious]"))
        ] or [p["data"] for p in posts][:8]
        for post in wanted:
            if len(out) >= n:
                break
            permalink = post.get("permalink", "")
            data = _get(f"https://www.reddit.com{permalink}.json?limit=100&raw_json=1&sort=top")
            time.sleep(SLEEP)
            if not data or len(data) < 2:
                continue
            for c in data[1].get("data", {}).get("children", []):
                d = c.get("data", {})
                if d.get("kind") == "more":
                    continue
                body = clean(d.get("body", ""))
                if usable(body):
                    out[d.get("id")] = {
                        "id": d.get("id"), "text": body, "score": d.get("score", 0),
                        "permalink": d.get("permalink", ""), "thread": post.get("title", ""),
                        "source": "thread",
                    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subreddit", default="nba")
    ap.add_argument("--target", type=int, default=400, help="raw comments to collect (label/filter down to >=250)")
    ap.add_argument("--out", default="data/raw_comments.csv")
    args = ap.parse_args()

    print(f"Collecting from r/{args.subreddit} (target {args.target} raw)...")
    merged = {}
    merged.update(collect_threads(args.subreddit, args.target // 2))
    print(f"  thread comments: {len(merged)}")
    merged.update(collect_flat(args.subreddit, args.target - len(merged)))
    print(f"  + flat comments: {len(merged)} total unique")

    rows = sorted(merged.values(), key=lambda r: -r["score"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "score", "permalink", "thread", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} raw comments -> {args.out}")
    if len(rows) < 300:
        print("  NOTE: <300 raw; re-run later (more threads will have posted) to build a buffer "
              "for balancing to >=250 labeled with each class >=20%.")


if __name__ == "__main__":
    main()
