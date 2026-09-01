"""Build an Elo-stratified player list from a PGN corpus (headers only).

parse_pgn.py reads full movetext because it needs NumMoves, which is far too
slow for the ~84M-game Elo-banded corpus. This scans headers only
(chess.pgn.read_headers) and records, per player, how many games they appear in
and their median Elo — enough to feed fetch_player_status.py.

Sampling is stratified: `--files-per-band` files are taken from each Elo band so
weak and strong players are both represented, instead of whichever band happens
to have the most files.

Usage:
    python src/scan_players.py --pgn-dir "/Volumes/My Passport Pro/ChessBase/split_by_elo"
    python src/scan_players.py --pgn-dir <dir> --files-per-band 2 --workers 4
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import chess.pgn
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
OUT_PATH = _BASE / "data" / "processed" / "player_counts_elo.parquet"

BAND_RE = re.compile(r"^(elo_\d+_tot_\d+)")


def scan_file(pgn_path: Path) -> tuple[Counter, dict]:
    """Return (games per player, summed Elo per player) for one file."""
    counts: Counter = Counter()
    elo_sum: dict[str, int] = defaultdict(int)
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            h = chess.pgn.read_headers(f)
            if h is None:
                break
            if "Bullet" in h.get("Event", ""):
                continue   # timing signal is noise at ~1s/move
            for side, elo_key in (("White", "WhiteElo"), ("Black", "BlackElo")):
                name = h.get(side, "")
                if not name or name == "?":
                    continue
                counts[name] += 1
                try:
                    elo_sum[name] += int(h.get(elo_key, 0) or 0)
                except ValueError:
                    pass
    print(f"  {pgn_path.name}: {sum(counts.values()):,} player-games", flush=True)
    return counts, dict(elo_sum)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn-dir", type=Path, required=True)
    ap.add_argument("--files-per-band", type=int, default=1,
                    help="Files sampled from each Elo band (default 1)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    by_band: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(args.pgn_dir.glob("*.pgn")):
        m = BAND_RE.match(p.stem)
        by_band[m.group(1) if m else "other"].append(p)

    files = [p for band in sorted(by_band)
             for p in by_band[band][:args.files_per_band]]
    if not files:
        raise SystemExit(f"No PGNs found in {args.pgn_dir}")
    print(f"{len(by_band)} Elo bands, scanning {len(files)} files "
          f"({args.files_per_band} per band) with {args.workers} workers...")

    with Pool(args.workers) as pool:
        results = pool.map(scan_file, files)

    total: Counter = Counter()
    elo_sum: Counter = Counter()
    for c, e in results:
        total.update(c)
        elo_sum.update(e)

    df = pd.DataFrame({
        "n_games": pd.Series(total),
        "mean_elo": pd.Series({k: elo_sum[k] / total[k] for k in total}),
    })
    df.index.name = "player"
    df = df[df["n_games"] >= 5].sort_values("n_games", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / (args.out.name + ".tmp")
    df.to_parquet(tmp)
    tmp.rename(args.out)

    print(f"\n{len(total):,} players seen, {len(df):,} with >=5 games -> {args.out}")
    bands = pd.cut(df["mean_elo"], bins=range(800, 3600, 200))
    print("\nplayers per Elo band (>=5 games):")
    for b, n in bands.value_counts().sort_index().items():
        if n:
            print(f"  {b}: {n:,}")


if __name__ == "__main__":
    main()
