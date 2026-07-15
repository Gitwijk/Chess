"""Fetch Lichess account status for a sample of players → ground-truth labels.

Queries the public Lichess API (POST /api/users, max 300 ids per request) and
records for each player:
  - tos_violation: banned for Terms-of-Service violation (cheating, boosting…)
  - disabled:      account closed (reason unknown — excluded from training)
  - title:         GM/IM/…/BOT (BOT accounts are known engine players)

Labels for the cheat detector:
  positive  = tos_violation
  negative  = normal account
  excluded  = disabled without tos_violation flag

Usage:
    python src/fetch_player_status.py                # sample 6000 players
    python src/fetch_player_status.py --sample 2000
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
COUNTS_PATH = _BASE / "data" / "processed" / "player_counts.parquet"
OUT_PATH = _BASE / "data" / "processed" / "players.parquet"

API_URL = "https://lichess.org/api/users"
BATCH = 300
SLEEP_BETWEEN = 3.0     # be polite to the free API
RETRY_429_WAIT = 65.0   # Lichess asks for a full minute after a 429


def fetch_batch(usernames: list[str]) -> list[dict]:
    body = ",".join(usernames).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "text/plain",
                 "User-Agent": "chess-ml research script"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  429 rate-limited, waiting {RETRY_429_WAIT}s...", flush=True)
                time.sleep(RETRY_429_WAIT)
            else:
                print(f"  HTTP {e.code}, retry {attempt + 1}...", flush=True)
                time.sleep(10)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error ({e}), retry {attempt + 1}...", flush=True)
            time.sleep(10)
    raise RuntimeError(f"Failed to fetch batch after retries (first user: {usernames[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=6000,
                    help="Number of players to sample (default 6000)")
    ap.add_argument("--min-games", type=int, default=20)
    args = ap.parse_args()

    counts = pd.read_parquet(COUNTS_PATH)
    pool = counts[counts["n_games"] >= args.min_games]
    print(f"Pool: {len(pool):,} players with >= {args.min_games} games")

    rng = random.Random(42)
    # Top-500 most active guaranteed in (lots of their games available),
    # rest sampled randomly from the pool.
    top = list(pool.index[:500])
    rest = rng.sample(list(pool.index[500:]), min(args.sample - len(top), len(pool) - 500))
    players = top + rest
    print(f"Querying {len(players):,} players in batches of {BATCH}...")

    # Resume support: skip players already fetched
    done: dict[str, dict] = {}
    if OUT_PATH.exists():
        prev = pd.read_parquet(OUT_PATH)
        done = {r["username_queried"]: r for _, r in prev.iterrows()}
        print(f"Resuming: {len(done):,} already fetched")

    todo = [p for p in players if p not in done]
    rows: list[dict] = list(done.values())

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        users = fetch_batch(batch)
        found = {u["id"].lower(): u for u in users}
        # API matches case-insensitively on id
        for name in batch:
            u = found.get(name.lower())
            if u is None:
                rows.append({"username_queried": name, "found": False,
                             "tos_violation": False, "disabled": False, "title": None})
                continue
            rows.append({
                "username_queried": name,
                "found": True,
                "tos_violation": bool(u.get("tosViolation", False)),
                "disabled": bool(u.get("disabled", False)),
                "title": u.get("title"),
            })
        n_tos = sum(r["tos_violation"] for r in rows)
        n_dis = sum(r["disabled"] for r in rows)
        print(f"  {len(rows):,}/{len(players):,} fetched  "
              f"(tos_violation={n_tos}, disabled={n_dis})", flush=True)

        df = pd.DataFrame(rows)
        tmp = OUT_PATH.parent / (OUT_PATH.name + ".tmp")
        df.to_parquet(tmp)
        tmp.rename(OUT_PATH)

        if i + BATCH < len(todo):
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df):,} players → {OUT_PATH}")
    print(f"  tos_violation : {df['tos_violation'].sum():,}")
    print(f"  disabled      : {df['disabled'].sum():,}")
    print(f"  bots          : {(df['title'] == 'BOT').sum():,}")
    print(f"  not found     : {(~df['found']).sum():,}")


if __name__ == "__main__":
    main()
