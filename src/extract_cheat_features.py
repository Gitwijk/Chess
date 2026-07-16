"""Extract per-player move-quality features for cheat detection.

For every game involving a labeled player (from players.parquet), replays the
game through the policy + value CNNs and measures, per move played:

  - rank of the played move among legal moves (by policy probability)
  - policy probability mass assigned to the played move
  - win-probability swing caused by the move (value net before vs. after)

Aggregated per (game, player) into one feature row. Output:
data/processed/cheat_features.parquet

Usage:
    python src/extract_cheat_features.py                  # all labeled players
    python src/extract_cheat_features.py --max-games 30   # cap per player
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from mcts import encode_board, encode_move_idx, load_models  # noqa: E402

DRIVE_DIR = Path("/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database")
LOCAL_RAW_DIR = _BASE / "data" / "raw"
PLAYERS_PATH = _BASE / "data" / "processed" / "players.parquet"

MIN_PLIES = 20          # skip very short games
SKIP_OPENING_PLIES = 10  # opening theory moves match engines trivially
STEM_RE = re.compile(r"^lichess_elite_\d{4}-\d{2}$")  # monthly files only, no merged dupes


def pgn_files() -> list[Path]:
    local = {p.stem: p for p in LOCAL_RAW_DIR.glob("lichess_elite_*.pgn")
             if STEM_RE.match(p.stem)}
    drive = {}
    if DRIVE_DIR.exists():
        drive = {p.stem: p for p in DRIVE_DIR.glob("lichess_elite_*.pgn")
                 if STEM_RE.match(p.stem) and p.stem not in local}
    return sorted((local | drive).values())


@torch.no_grad()
def game_features(game: chess.pgn.Game, policy_net, value_net, device) -> dict | None:
    """Replay one game; return per-side move-quality features."""
    moves = list(game.mainline_moves())
    if len(moves) < MIN_PLIES:
        return None

    # Encode every position (before each move) in one batch
    board = game.board()
    boards_np, move_idxs, legal_lists, turns = [], [], [], []
    for move in moves:
        boards_np.append(encode_board(board))
        move_idxs.append(encode_move_idx(board, move))
        legal_lists.append([encode_move_idx(board, m) for m in board.legal_moves])
        turns.append(board.turn)
        board.push(move)
    # Final position too (for the last move's value swing)
    boards_np.append(encode_board(board))

    batch = torch.from_numpy(np.stack(boards_np)).to(device)
    policy_logits = policy_net(batch[:-1]).cpu().numpy()          # (n_moves, 4096)
    values_stm = torch.sigmoid(value_net(batch)).cpu().numpy()    # (n_moves+1,) side-to-move POV

    # Convert values to White's POV so swings are comparable
    values_white = np.array([
        v if t == chess.WHITE else 1.0 - v
        for v, t in zip(values_stm, list(turns) + [board.turn])
    ])

    per_side: dict[bool, dict[str, list]] = {
        chess.WHITE: defaultdict(list), chess.BLACK: defaultdict(list)}

    for i, (move_idx, legal, turn) in enumerate(zip(move_idxs, legal_lists, turns)):
        if i < SKIP_OPENING_PLIES:
            continue
        logits = policy_logits[i]
        leg = np.array(legal)
        leg_logits = logits[leg]
        # softmax over legal moves only
        leg_logits = leg_logits - leg_logits.max()
        probs = np.exp(leg_logits)
        probs /= probs.sum()

        played_pos = int(np.where(leg == move_idx)[0][0])
        rank = int((probs > probs[played_pos]).sum())  # 0 = engine's top choice
        d = per_side[turn]
        d["rank"].append(rank)
        d["top1"].append(rank == 0)
        d["top3"].append(rank < 3)
        d["prob"].append(float(probs[played_pos]))
        # swing from mover's perspective: positive = position improved
        swing = values_white[i + 1] - values_white[i]
        d["swing"].append(swing if turn == chess.WHITE else -swing)

    def agg(d: dict[str, list]) -> dict | None:
        if len(d["rank"]) < 8:
            return None
        r = np.array(d["rank"]); p = np.array(d["prob"]); s = np.array(d["swing"])
        return {
            "n_moves": len(r),
            "top1_rate": float(np.mean(d["top1"])),
            "top3_rate": float(np.mean(d["top3"])),
            "mean_rank": float(r.mean()),
            "median_rank": float(np.median(r)),
            "mean_prob": float(p.mean()),
            "mean_swing": float(s.mean()),
            "worst_swing": float(s.min()),
            "blunder_rate": float(np.mean(s < -0.15)),
        }

    def seq(d: dict[str, list]) -> dict | None:
        if len(d["rank"]) < 8:
            return None
        return {
            "seq_rank": np.array(d["rank"], dtype=np.int16),
            "seq_prob": np.array(d["prob"], dtype=np.float32),
            "seq_swing": np.array(d["swing"], dtype=np.float32),
        }

    w, b = agg(per_side[chess.WHITE]), agg(per_side[chess.BLACK])
    if w is None and b is None:
        return None
    return {"white": w, "black": b,
            "white_seq": seq(per_side[chess.WHITE]),
            "black_seq": seq(per_side[chess.BLACK])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=30,
                    help="Max games per labeled player (default 30)")
    ap.add_argument("--max-total", type=int, default=0,
                    help="Stop after this many feature rows (0 = no cap)")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="Stop after scanning this many games (0 = no cap; for smoke tests)")
    ap.add_argument("--sequences", action="store_true",
                    help="Also write per-move sequences to <prefix>_sequences.parquet")
    ap.add_argument("--out-prefix", default="cheat_features",
                    help="Output filename prefix in data/processed/ (default cheat_features)")
    args = ap.parse_args()

    out_path = _BASE / "data" / "processed" / f"{args.out_prefix}.parquet"
    seq_path = _BASE / "data" / "processed" / f"{args.out_prefix}_sequences.parquet"

    players = pd.read_parquet(PLAYERS_PATH)
    players = players[players["found"]]
    # exclude closed accounts with unknown reason; keep tos_violation + clean
    players = players[players["tos_violation"] | ~players["disabled"]]
    labels = dict(zip(players["username_queried"],
                      players["tos_violation"]))
    titles = dict(zip(players["username_queried"], players["title"]))
    wanted = set(labels)
    print(f"{len(wanted):,} labeled players "
          f"({players['tos_violation'].sum():,} tos_violation)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy_net, value_net = load_models(device)
    print(f"Models loaded on {device}")

    games_per_player: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    seq_rows: list[dict] = []
    n_scanned = 0
    stop = False

    # Newest files first: labels are current account status, so recent games
    # are the most label-consistent (a 2015 game by a 2024-banned player may
    # well be clean).
    for pgn_path in reversed(pgn_files()):
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            while True:
                # Fast path: parse headers only; movetext is skipped unless
                # a labeled player is involved (~10× faster scanning).
                offset = f.tell()
                headers = chess.pgn.read_headers(f)
                if headers is None:
                    break
                n_scanned += 1
                white = headers.get("White", "")
                black = headers.get("Black", "")
                w_want = white in wanted and games_per_player[white] < args.max_games
                b_want = black in wanted and games_per_player[black] < args.max_games
                if not (w_want or b_want):
                    continue

                f.seek(offset)
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                feats = game_features(game, policy_net, value_net, device)
                if feats is None:
                    continue

                for name, side_feats, side_seq, is_white in (
                        (white, feats["white"], feats["white_seq"], True),
                        (black, feats["black"], feats["black_seq"], False)):
                    want = w_want if is_white else b_want
                    if not want or side_feats is None:
                        continue
                    games_per_player[name] += 1
                    meta = {
                        "player": name,
                        "label_tos": labels[name],
                        "title": titles.get(name),
                        "is_white": is_white,
                        "elo": int(game.headers.get(
                            "WhiteElo" if is_white else "BlackElo", 0) or 0),
                        "opp_elo": int(game.headers.get(
                            "BlackElo" if is_white else "WhiteElo", 0) or 0),
                        "source_file": pgn_path.stem,
                    }
                    rows.append({**meta, **side_feats})
                    if args.sequences and side_seq is not None:
                        seq_rows.append({**meta, **side_seq})

                if args.max_scan and n_scanned >= args.max_scan:
                    stop = True
                    break

        n_pos = sum(1 for r in rows if r["label_tos"])
        print(f"{pgn_path.stem}: {len(rows):,} rows "
              f"({n_pos:,} tos-labeled), {n_scanned:,} games scanned", flush=True)

        df = pd.DataFrame(rows)
        tmp = out_path.parent / (out_path.name + ".tmp")
        df.to_parquet(tmp)
        tmp.rename(out_path)
        if args.sequences:
            sdf = pd.DataFrame(seq_rows)
            tmp = seq_path.parent / (seq_path.name + ".tmp")
            sdf.to_parquet(tmp)
            tmp.rename(seq_path)

        if args.max_total and len(rows) >= args.max_total:
            print("Reached --max-total, stopping.")
            break
        if stop:
            print("Reached --max-scan, stopping.")
            break

    print(f"\nDone. {len(rows):,} feature rows → {out_path}")
    if args.sequences:
        print(f"      {len(seq_rows):,} sequence rows → {seq_path}")


if __name__ == "__main__":
    main()
