"""Parse Lichess Elite PGN files into a single parquet file of per-game metadata."""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess.pgn
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pgn_source import add_source_args, find_pgn_files  # noqa: E402

_BASE = Path(__file__).resolve().parent.parent
OUT_PATH = _BASE / "data" / "processed" / "games.parquet"
PARTS_DIR = _BASE / "data" / "processed" / "_parts"

FIELDS = [
    "Event", "White", "Black", "Result", "WhiteElo", "BlackElo",
    "ECO", "Opening", "TimeControl", "UTCDate", "UTCTime", "Termination",
]


def iter_games(pgn_path: Path):
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def game_to_row(game: chess.pgn.Game, source_file: str) -> dict:
    headers = game.headers
    row = {field: headers.get(field) for field in FIELDS}
    row["WhiteElo"] = pd.to_numeric(row["WhiteElo"], errors="coerce")
    row["BlackElo"] = pd.to_numeric(row["BlackElo"], errors="coerce")
    row["NumMoves"] = sum(1 for _ in game.mainline_moves())
    row["SourceFile"] = source_file
    return row


def parse_file(pgn_path: Path) -> Path:
    part_path = PARTS_DIR / f"{pgn_path.stem}.parquet"
    if part_path.exists():
        print(f"  skip {pgn_path.name} (already parsed)", flush=True)
        return part_path

    for attempt in range(5):
        try:
            rows = [game_to_row(game, pgn_path.name) for game in iter_games(pgn_path)]
            break
        except OSError as e:
            print(f"  retry {pgn_path.name} (attempt {attempt + 1}): {e}", flush=True)
            time.sleep(5)
    else:
        raise OSError(f"Failed to read {pgn_path.name} after 5 attempts")

    # Drop forfeit/aborted games with no moves.
    rows = [row for row in rows if row["NumMoves"] > 0]

    df = pd.DataFrame(rows)
    tmp_path = part_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(part_path)
    print(f"  done {pgn_path.name}: {len(rows):,} games", flush=True)
    return part_path


# File discovery (including the stem filter) now lives in pgn_source.py.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--out", type=Path, default=OUT_PATH,
                    help="Output parquet path (default data/processed/games.parquet)")
    args = ap.parse_args()

    pgn_files = find_pgn_files(args.pgn_dir, args.pattern)
    out_path = args.out if args.out.is_absolute() else _BASE / args.out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = 6
    print(f"Parsing {len(pgn_files)} files using {n_workers} workers...")
    with Pool(n_workers) as pool:
        part_paths = pool.map(parse_file, pgn_files)

    df = pd.concat((pd.read_parquet(p) for p in part_paths), ignore_index=True)
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df):,} games to {out_path}")


if __name__ == "__main__":
    main()
