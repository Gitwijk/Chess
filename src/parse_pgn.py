"""Parse Lichess Elite PGN files into a single parquet file of per-game metadata."""

import sys
from pathlib import Path

import chess.pgn
import pandas as pd

RAW_DIR = Path("/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database")
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "games.parquet"

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


def main():
    pgn_files = sorted(RAW_DIR.glob("lichess_elite_*.pgn"))
    if not pgn_files:
        sys.exit(f"No PGN files found in {RAW_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, pgn_path in enumerate(pgn_files, 1):
        print(f"[{i}/{len(pgn_files)}] {pgn_path.name}")
        for game in iter_games(pgn_path):
            rows.append(game_to_row(game, pgn_path.name))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} games to {OUT_PATH}")


if __name__ == "__main__":
    main()
