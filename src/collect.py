#!/usr/bin/env python3
"""
Collect real r/sports comments for TakeMeter.

Reddit's public JSON endpoints now return HTTP 403 to non-authenticated clients
(an anti-bot wall) even from a residential IP, so live scraping is not reliable.
Instead this pulls real r/sports comments from a public Reddit archive on the
Hugging Face Hub (HuggingFaceGECLM/REDDIT_comments, PushShift dumps 2006-2023),
which is the same underlying source PushShift always provided.

    python src/collect.py --target 900 --out data/raw_comments.csv

It streams the r/sports split, cleans each comment (strips URLs, user/sub mentions,
quote lines, bot/automod posts, deleted/removed, too-short and link-only), dedupes,
and keeps a MIX of comment lengths so the reasoned `analysis` class is reachable and
the bulk `reaction` class is represented. Output columns match the rest of the
pipeline: id,text,score,permalink,thread,source. Labeling happens in src/label.py.

Needs: datasets, pandas (already in requirements.txt).
"""
import argparse
import csv
import html
import re
import sys

DATASET = "HuggingFaceGECLM/REDDIT_comments"
SPLIT = "sports"

BOT_MARKERS = ("i am a bot", "performed automatically", "automoderator", "^(i am a bot")
JUNK = {"[deleted]", "[removed]", ""}


def clean(body: str) -> str:
    body = html.unescape(body or "")
    body = re.sub(r"https?://\S+", "", body)        # strip URLs
    body = re.sub(r"/?u/\w+|/?r/\w+", "", body)      # strip user/sub mentions
    body = re.sub(r"^&gt;.*$", "", body, flags=re.M)  # strip quoted lines
    body = re.sub(r"\s+", " ", body).strip()
    return body


def usable(body: str) -> bool:
    if body in JUNK:
        return False
    low = body.lower()
    if any(m in low for m in BOT_MARKERS):
        return False
    words = body.split()
    if len(words) < 4 or len(body) < 16:   # too short to carry discourse
        return False
    if len(body) > 1200:                    # keep it comment-sized
        return False
    if not re.search(r"[a-zA-Z]", body):
        return False
    # drop pure one-liner link/image leftovers and quotes
    if low.startswith(("edit:", "deleted", "this comment")):
        return False
    return True


def bucket(body: str) -> str:
    n = len(body)
    if n <= 80:
        return "short"   # reaction-leaning
    if n <= 260:
        return "mid"
    return "long"        # analysis-leaning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=900, help="raw comments to collect")
    ap.add_argument("--out", default="data/raw_comments.csv")
    ap.add_argument("--scan-limit", type=int, default=120000, help="max stream rows to scan")
    args = ap.parse_args()

    from datasets import load_dataset

    # Length-balanced quotas so `analysis` (long, reasoned) is reachable, not drowned
    # out by the much more common short reactions.
    quota = {"short": int(args.target * 0.30),
             "mid":   int(args.target * 0.40),
             "long":  args.target - int(args.target * 0.30) - int(args.target * 0.40)}
    got = {"short": 0, "mid": 0, "long": 0}

    print(f"Streaming r/{SPLIT} from {DATASET} (target {args.target}, quotas {quota})...")
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)

    seen_text = set()
    rows = {}
    scanned = 0
    for r in ds:
        scanned += 1
        if scanned > args.scan_limit:
            break
        body = clean(r.get("body", ""))
        if not usable(body):
            continue
        key = body.lower()
        if key in seen_text:
            continue
        b = bucket(body)
        if got[b] >= quota[b]:
            if all(got[k] >= quota[k] for k in quota):
                break
            continue
        seen_text.add(key)
        cid = r.get("id") or f"row{scanned}"
        rows[cid] = {
            "id": cid,
            "text": body,
            "score": r.get("score", ""),
            "permalink": r.get("permalink", ""),
            "thread": r.get("link_id", ""),
            "source": f"hf:{DATASET}#{SPLIT}",
        }
        got[b] += 1
        if sum(got.values()) % 100 == 0:
            print(f"  scanned {scanned:>6}  collected {sum(got.values())}  ({got})")

    out_rows = list(rows.values())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "score", "permalink", "thread", "source"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} raw comments -> {args.out}  (scanned {scanned} rows; buckets {got})")
    if len(out_rows) < 300:
        print("  NOTE: <300 raw collected; raise --scan-limit.", file=sys.stderr)


if __name__ == "__main__":
    main()
