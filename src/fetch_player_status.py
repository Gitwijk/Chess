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
SLEEP_BETWEEN = 6.0     # be polite to the free API
RETRY_429_WAIT = 65.0   # Lichess asks for a full minute after a 429
MAX_ATTEMPTS = 8        # 4 was not enough: repeated 429s escalate the penalty


def fetch_batch(usernames: list[str]) -> list[dict]:
    """Fetch one batch. Returns None if the batch could not be fetched, so the
    caller can save progress and stop cleanly instead of losing the run."""
    body = ",".join(usernames).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "text/plain",
                 "User-Agent": "chess-ml research script"},
        method="POST",
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Escalate: repeated violations extend the cool-off period.
                wait = RETRY_429_WAIT * (attempt + 1)
                print(f"  429 rate-limited, waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_ATTEMPTS})...", flush=True)
                time.sleep(wait)
            else:
                print(f"  HTTP {e.code}, retry {attempt + 1}...", flush=True)
                time.sleep(10 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error ({e}), retry {attempt + 1}...", flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=6000,
                    help="Number of players to sample (default 6000)")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--counts", type=Path, default=COUNTS_PATH,
                    help="Player-counts parquet (default data/processed/player_counts.parquet; "
                         "use player_counts_elo.parquet for the Elo-banded corpus)")
    ap.add_argument("--out", type=Path, default=OUT_PATH,
                    help="Output parquet (default data/processed/players.parquet)")
    ap.add_argument("--stratify-elo", action="store_true",
                    help="Sample evenly across 200-point Elo bands (needs a "
                         "mean_elo column, as produced by scan_players.py)")
    args = ap.parse_args()

    out_path = args.out if args.out.is_absolute() else _BASE / args.out
    counts = pd.read_parquet(args.counts)
    pool = counts[counts["n_games"] >= args.min_games]
    print(f"Pool: {len(pool):,} players with >= {args.min_games} games")

    rng = random.Random(42)
    if args.stratify_elo and "mean_elo" in pool.columns:
        # Even coverage per rating band. Without this the sample follows the
        # population, which is dominated by 1200-1800 and would leave too few
        # labelled players at the extremes to evaluate per-band performance.
        bands = (pool["mean_elo"] // 200 * 200).astype(int)
        groups = {b: list(idx) for b, idx in pool.groupby(bands).groups.items()}
        per_band = max(1, args.sample // len(groups))
        players = []
        for b in sorted(groups):
            g = groups[b]
            players.extend(rng.sample(g, min(per_band, len(g))))
        rng.shuffle(players)
        print(f"Stratified over {len(groups)} Elo bands, "
              f"up to {per_band:,} players each")
    else:
        # Top-500 most active guaranteed in (lots of their games available),
        # rest sampled randomly from the pool.
        top = list(pool.index[:500])
        rest = rng.sample(list(pool.index[500:]),
                          min(args.sample - len(top), len(pool) - 500))
        players = top + rest
    print(f"Querying {len(players):,} players in batches of {BATCH}...")

    # Resume support: skip players already fetched
    done: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        # to_dict("records") — iterrows() yields Series, and mixing those with
        # the plain dicts appended below makes pd.DataFrame(rows) fail.
        done = {r["username_queried"]: r for r in prev.to_dict("records")}
        print(f"Resuming: {len(done):,} already fetched")

    todo = [p for p in players if p not in done]
    rows: list[dict] = list(done.values())

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        users = fetch_batch(batch)
        if users is None:
            print(f"\nGiving up after {MAX_ATTEMPTS} attempts. "
                  f"{len(rows):,} players saved to {out_path} — "
                  f"re-run the same command later to resume.", flush=True)
            break
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
        tmp = out_path.parent / (out_path.name + ".tmp")
        df.to_parquet(tmp)
        tmp.rename(out_path)

        if i + BATCH < len(todo):
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df):,} players → {out_path}")
    print(f"  tos_violation : {df['tos_violation'].sum():,}")
    print(f"  disabled      : {df['disabled'].sum():,}")
    print(f"  bots          : {(df['title'] == 'BOT').sum():,}")
    print(f"  not found     : {(~df['found']).sum():,}")


if __name__ == "__main__":
    main()
