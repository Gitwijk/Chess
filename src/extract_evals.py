"""Extract Stockfish-evaluated positions from the Lichess evaluation database.

Download the database first (~5 GB compressed):
    curl -O https://database.lichess.org/lichess_db_eval.jsonl.zst

Then run:
    python src/extract_evals.py data/lichess_db_eval.jsonl.zst

Outputs shards to data/processed/evals/. Resumable: existing shards are
skipped so you can restart after an interruption.

Label: sigmoid(cp / 400) from the side-to-move's perspective. Mate scores
are treated as ±1000 cp. This converts a Stockfish centipawn evaluation into
a win-probability in [0, 1] suitable for BCEWithLogitsLoss.
"""

import argparse
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np
import zstandard

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "evals"
SHARD_SIZE = 200_000
SAMPLE_RATE = 0.05   # 5% of ~200M positions ≈ 10M positions total
MATE_CP = 1000       # treat mate as ±1000 cp before sigmoid

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
    # planes 0–11: piece positions (side-to-move perspective)
    for square, piece in b.piece_map().items():
        row, col = divmod(square, 8)
        planes[PIECE_TO_PLANE[(piece.piece_type, piece.color)], row, col] = 1
    # planes 12–15: castling rights (side-to-move K/Q, opponent k/q)
    if b.has_kingside_castling_rights(chess.WHITE):  planes[12] = 1
    if b.has_queenside_castling_rights(chess.WHITE): planes[13] = 1
    if b.has_kingside_castling_rights(chess.BLACK):  planes[14] = 1
    if b.has_queenside_castling_rights(chess.BLACK): planes[15] = 1
    # plane 16: en passant target square
    if b.ep_square is not None:
        row, col = divmod(b.ep_square, 8)
        planes[16, row, col] = 1
    return planes


def best_cp(evals: list) -> float | None:
    """Centipawn score from the deepest available evaluation (White's POV)."""
    if not evals:
        return None
    best = max(evals, key=lambda e: e.get("depth", 0))
    pvs = best.get("pvs", [])
    if not pvs:
        return None
    pv = pvs[0]
    if "mate" in pv:
        return math.copysign(MATE_CP, pv["mate"])
    if "cp" in pv:
        return max(-MATE_CP, min(MATE_CP, float(pv["cp"])))
    return None


def cp_to_label(cp: float, side_to_move: chess.Color) -> float:
    """Win probability [0,1] for the side to move, derived from centipawns."""
    prob_white = 1.0 / (1.0 + math.exp(-cp / 400.0))
    return prob_white if side_to_move == chess.WHITE else 1.0 - prob_white


def save_shard(idx: int, boards: list, labels: list) -> Path:
    out_path = OUT_DIR / f"shard_{idx:06d}.npz"
    tmp_path = OUT_DIR / f"shard_{idx:06d}.npz.tmp"
    boards_arr = np.stack(boards).astype(np.int8)
    labels_arr = np.array(labels, dtype=np.float32)
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, boards=boards_arr, labels=labels_arr)
    tmp_path.rename(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Path to lichess_db_eval.jsonl.zst")
    ap.add_argument("--sample", type=float, default=SAMPLE_RATE,
                    help=f"Fraction of positions to sample (default {SAMPLE_RATE})")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"File not found: {in_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(OUT_DIR.glob("shard_*.npz"))
    done_shards = len(existing)
    skip_positions = done_shards * SHARD_SIZE
    shard_idx = done_shards
    print(f"Output dir : {OUT_DIR}")
    print(f"Sample rate: {args.sample:.1%}")
    print(f"Resuming   : shard {shard_idx} (skipping first {skip_positions:,} sampled positions)")

    rng = random.Random(42)
    boards_buf: list = []
    labels_buf: list = []
    n_seen = n_sampled = n_errors = 0
    t0 = t_last = time.time()

    dctx = zstandard.ZstdDecompressor()
    with open(in_path, "rb") as fh:
        stream = dctx.stream_reader(fh)
        text = io.TextIOWrapper(stream, encoding="utf-8")
        for raw_line in text:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            n_seen += 1

            if rng.random() >= args.sample:
                continue
            n_sampled += 1

            if n_sampled <= skip_positions:
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                n_errors += 1
                continue

            cp = best_cp(obj.get("evals", []))
            if cp is None:
                continue

            try:
                board = chess.Board(obj["fen"])
            except Exception:
                n_errors += 1
                continue

            boards_buf.append(encode_board(board))
            labels_buf.append(cp_to_label(cp, board.turn))

            if len(boards_buf) >= SHARD_SIZE:
                path = save_shard(shard_idx, boards_buf, labels_buf)
                print(f"  shard {shard_idx:06d} saved → {path.name}", flush=True)
                shard_idx += 1
                boards_buf, labels_buf = [], []

            now = time.time()
            if now - t_last >= 60:
                rate = n_seen / (now - t0)
                print(f"  {n_seen:,} lines  {n_sampled:,} sampled  "
                      f"{shard_idx} shards  {rate:,.0f} lines/s", flush=True)
                t_last = now

    if boards_buf:
        save_shard(shard_idx, boards_buf, labels_buf)
        shard_idx += 1

    total = sum(len(np.load(p)["labels"]) for p in sorted(OUT_DIR.glob("shard_*.npz")))
    print(f"\nDone. {shard_idx} shards, {total:,} positions, {n_errors} parse errors.")


if __name__ == "__main__":
    main()
