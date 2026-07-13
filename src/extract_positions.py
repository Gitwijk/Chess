"""Sample positions from games and encode them as (12, 8, 8) tensors with an outcome label.

For each sampled game, picks a random ply, encodes the resulting board from the
perspective of the side to move (mirroring if it's Black's turn), and labels it
with the game outcome from that side's perspective: 1.0 = win, 0.5 = draw, 0.0 = loss.
"""

import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd

DRIVE_DIR = Path("/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database")
LOCAL_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "positions"

SAMPLE_RATE = 0.02  # ~2% of ~25M games
POSITIONS_PER_GAME = 3  # distinct random plies sampled per game -> ~1.5M positions total
MIN_PLY, MAX_PLY = 10, 40
RESULT_TO_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

STEM_RE = re.compile(r"^lichess_elite_\d{4}-\d{2}$")

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode the board as a (12, 8, 8) tensor from the perspective of the side to move."""
    b = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((12, 8, 8), dtype=np.int8)
    for square, piece in b.piece_map().items():
        row, col = divmod(square, 8)
        planes[PIECE_TO_PLANE[(piece.piece_type, piece.color)], row, col] = 1
    return planes


def iter_games(pgn_path: Path):
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def process_file(pgn_path: Path) -> Path:
    out_path = OUT_DIR / f"{pgn_path.stem}.npz"
    if out_path.exists():
        print(f"  skip {pgn_path.name} (already done)", flush=True)
        return out_path

    rng = random.Random(hash(pgn_path.stem) & 0xFFFFFFFF)
    boards, labels = [], []

    for attempt in range(5):
        try:
            for game in iter_games(pgn_path):
                result = game.headers.get("Result")
                if result not in RESULT_TO_SCORE or rng.random() >= SAMPLE_RATE:
                    continue

                moves = list(game.mainline_moves())
                if len(moves) < MIN_PLY + 1:
                    continue

                max_ply = min(MAX_PLY, len(moves) - 1)
                n_plies = min(POSITIONS_PER_GAME, max_ply - MIN_PLY + 1)
                plies = sorted(rng.sample(range(MIN_PLY, max_ply + 1), n_plies))

                score = RESULT_TO_SCORE[result]
                board = game.board()
                for ply_idx, move in enumerate(moves[:max_ply], start=1):
                    board.push(move)
                    if ply_idx in plies:
                        s = score if board.turn == chess.WHITE else 1.0 - score
                        boards.append(encode_board(board))
                        labels.append(s)
            break
        except OSError as e:
            print(f"  retry {pgn_path.name} (attempt {attempt + 1}): {e}", flush=True)
            boards, labels = [], []
            time.sleep(5)
    else:
        raise OSError(f"Failed to read {pgn_path.name} after 5 attempts")

    boards_arr = np.stack(boards) if boards else np.empty((0, 12, 8, 8), dtype=np.int8)
    labels_arr = np.array(labels, dtype=np.float32)

    tmp_path = out_path.parent / (out_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, boards=boards_arr, labels=labels_arr)
    tmp_path.rename(out_path)
    print(f"  done {pgn_path.name}: {len(labels_arr):,} positions", flush=True)
    return out_path


def main():
    local_files = {p.stem: p for p in LOCAL_RAW_DIR.glob("lichess_elite_*.pgn") if STEM_RE.match(p.stem)}
    drive_files = {}
    if DRIVE_DIR.exists():
        drive_files = {
            p.stem: p for p in DRIVE_DIR.glob("lichess_elite_*.pgn")
            if STEM_RE.match(p.stem) and p.stem not in local_files
        }
    pgn_files = sorted((local_files | drive_files).values())
    if not pgn_files:
        raise SystemExit("No PGN files found in data/raw/ or the Google Drive folder")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = 6
    print(f"Extracting positions from {len(pgn_files)} files using {n_workers} workers "
          f"(sample rate {SAMPLE_RATE:.1%})...")
    with Pool(n_workers) as pool:
        part_paths = pool.map(process_file, pgn_files)

    total = sum(len(np.load(p)["labels"]) for p in part_paths)
    print(f"\nTotal positions extracted: {total:,}")


if __name__ == "__main__":
    main()
