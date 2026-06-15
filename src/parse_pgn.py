"""Parse Lichess Elite PGN files into a single parquet file of per-game metadata."""

import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess.pgn
import pandas as pd

DRIVE_DIR = Path("/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database")
LOCAL_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "games.parquet"
PARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "_parts"

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

    df = pd.DataFrame(rows)
    tmp_path = part_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(part_path)
    print(f"  done {pgn_path.name}: {len(rows):,} games", flush=True)
    return part_path


STEM_RE = re.compile(r"^lichess_elite_\d{4}-\d{2}$")


def main():
    # Prefer local copies (more reliable than the network drive); fall back to the drive.
    # Skip stray duplicates (e.g. "lichess_elite_2021-06 2.pgn").
    local_files = {p.stem: p for p in LOCAL_RAW_DIR.glob("lichess_elite_*.pgn") if STEM_RE.match(p.stem)}
    drive_files = {}
    if DRIVE_DIR.exists():
        drive_files = {
            p.stem: p for p in DRIVE_DIR.glob("lichess_elite_*.pgn")
            if STEM_RE.match(p.stem) and p.stem not in local_files
        }

    pgn_files = sorted((local_files | drive_files).values())
    if not pgn_files:
        sys.exit("No PGN files found in data/raw/ or the Google Drive folder")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = 6
    print(f"Parsing {len(pgn_files)} files using {n_workers} workers...")
    with Pool(n_workers) as pool:
        part_paths = pool.map(parse_file, pgn_files)

    df = pd.concat((pd.read_parquet(p) for p in part_paths), ignore_index=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} games to {OUT_PATH}")


if __name__ == "__main__":
    main()
