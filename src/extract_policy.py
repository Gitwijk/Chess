"""Extract (board, move) training pairs from PGN files for policy head training.

For each sampled game, replays every move and records:
  - board: 17-plane (8x8) encoding from the side-to-move's perspective
  - move:  from_square * 64 + to_square (0-4095), mirrored for Black

Both board and move use the same mirroring convention as extract_evals.py,
so the same CNN backbone can serve both heads.

Run:
    python src/extract_policy.py
"""

import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.pgn
import numpy as np

DRIVE_DIR = Path("/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database")
LOCAL_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "policy"

SAMPLE_RATE = 0.02  # 2% of ~25M games -> ~500K games -> ~20M positions
STEM_RE = re.compile(r"^lichess_elite_\d{4}-\d{2}$")

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4, (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10, (chess.KING,  chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    b = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((17, 8, 8), dtype=np.int8)
    for square, piece in b.piece_map().items():
        row, col = divmod(square, 8)
        planes[PIECE_TO_PLANE[(piece.piece_type, piece.color)], row, col] = 1
    if b.has_kingside_castling_rights(chess.WHITE):  planes[12] = 1
    if b.has_queenside_castling_rights(chess.WHITE): planes[13] = 1
    if b.has_kingside_castling_rights(chess.BLACK):  planes[14] = 1
    if b.has_queenside_castling_rights(chess.BLACK): planes[15] = 1
    if b.ep_square is not None:
        row, col = divmod(b.ep_square, 8)
        planes[16, row, col] = 1
    return planes


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode move as from_square * 64 + to_square (0-4095).

    Mirrors both squares to the side-to-move coordinate frame, matching
    the board encoding convention. Under-promotions map to the same index
    as queen promotions (identical from/to squares); acceptable because
    under-promotions are extremely rare in elite play.
    """
    if board.turn == chess.WHITE:
        return move.from_square * 64 + move.to_square
    return chess.square_mirror(move.from_square) * 64 + chess.square_mirror(move.to_square)


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
    boards, moves = [], []

    for attempt in range(5):
        try:
            for game in iter_games(pgn_path):
                if rng.random() >= SAMPLE_RATE:
                    continue
                board = game.board()
                node = game
                while node.variations:
                    next_node = node.variations[0]
                    move = next_node.move
                    boards.append(encode_board(board))
                    moves.append(encode_move(board, move))
                    board.push(move)
                    node = next_node
            break
        except OSError as e:
            print(f"  retry {pgn_path.name} (attempt {attempt + 1}): {e}", flush=True)
            boards, moves = [], []
            time.sleep(5)
    else:
        raise OSError(f"Failed to read {pgn_path.name} after 5 attempts")

    boards_arr = np.stack(boards) if boards else np.empty((0, 17, 8, 8), dtype=np.int8)
    moves_arr = np.array(moves, dtype=np.int16)

    tmp_path = out_path.parent / (out_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, boards=boards_arr, moves=moves_arr)
    tmp_path.rename(out_path)
    print(f"  done {pgn_path.name}: {len(moves_arr):,} positions", flush=True)
    return out_path


def main():
    local_files = {p.stem: p for p in LOCAL_RAW_DIR.glob("lichess_elite_*.pgn")
                   if STEM_RE.match(p.stem)}
    drive_files = {}
    if DRIVE_DIR.exists():
        drive_files = {
            p.stem: p for p in DRIVE_DIR.glob("lichess_elite_*.pgn")
            if STEM_RE.match(p.stem) and p.stem not in local_files
        }
    pgn_files = sorted((local_files | drive_files).values())
    if not pgn_files:
        raise SystemExit("No PGN files found in data/raw/ or Google Drive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = 6
    print(f"Extracting policy data from {len(pgn_files)} files "
          f"using {n_workers} workers (sample rate {SAMPLE_RATE:.1%})...")
    with Pool(n_workers) as pool:
        part_paths = pool.map(process_file, pgn_files)

    total = sum(len(np.load(p)["moves"]) for p in part_paths)
    print(f"\nTotal positions extracted: {total:,}")


if __name__ == "__main__":
    main()
