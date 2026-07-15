"""Play chess against the CNN + MCTS engine.

Usage:
    python src/play.py                   # you play White, engine plays Black
    python src/play.py --side black      # you play Black
    python src/play.py --side random     # randomly assigned
    python src/play.py --sims 800        # more simulations per move (stronger, slower)
    python src/play.py --fen "FEN..."    # start from a custom position

Enter moves in UCI notation: e2e4, g1f3, e1g1 (castling), e7e8q (promotion).
Type 'quit' to exit, 'board' to reprint the position, 'undo' to take back a move.
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from mcts import MCTS, load_models


def print_board(board: chess.Board, player_is_white: bool) -> None:
    """Print an ASCII board from the player's perspective."""
    symbols = {
        chess.PAWN:   ("♙", "♟"),
        chess.KNIGHT: ("♘", "♞"),
        chess.BISHOP: ("♗", "♝"),
        chess.ROOK:   ("♖", "♜"),
        chess.QUEEN:  ("♕", "♛"),
        chess.KING:   ("♔", "♚"),
    }
    ranks = range(8) if not player_is_white else range(7, -1, -1)
    files = range(8) if player_is_white else range(7, -1, -1)

    print()
    for rank in ranks:
        row = f"  {rank + 1} "
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += "·" if (rank + file) % 2 == 0 else "·"
            else:
                w_sym, b_sym = symbols[piece.piece_type]
                row += w_sym if piece.color == chess.WHITE else b_sym
            row += " "
        print(row)
    file_labels = "  abcdefgh" if player_is_white else "  hgfedcba"
    print(file_labels)
    print()


def get_player_move(board: chess.Board) -> chess.Move | None:
    """Prompt the player for a move; return None on 'quit'."""
    while True:
        try:
            raw = input("Your move (UCI): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw in ("quit", "q", "exit"):
            return None
        if raw in ("board", "b"):
            return "board"   # type: ignore[return-value]
        if raw in ("undo", "u"):
            return "undo"    # type: ignore[return-value]

        try:
            move = chess.Move.from_uci(raw)
        except ValueError:
            print(f"  Invalid UCI notation: {raw!r}  (example: e2e4, g1f3, e7e8q)")
            continue

        if move not in board.legal_moves:
            # Try adding queen promotion if user forgot promotion suffix
            promo = chess.Move(move.from_square, move.to_square, chess.QUEEN)
            if promo in board.legal_moves:
                print("  Assuming queen promotion.")
                return promo
            print(f"  Illegal move: {raw}")
            continue

        return move


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["white", "black", "random"], default="white")
    ap.add_argument("--sims", type=int, default=400,
                    help="MCTS simulations per engine move (default 400)")
    ap.add_argument("--fen", default=chess.STARTING_FEN,
                    help="Starting FEN (default: standard starting position)")
    ap.add_argument("--c-puct", type=float, default=1.4,
                    help="PUCT exploration constant (default 1.4)")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading models (device: {device})...", end=" ", flush=True)
    policy_net, value_net = load_models(device)
    engine = MCTS(policy_net, value_net, device, c_puct=args.c_puct)
    print("ready.\n")

    side = args.side
    if side == "random":
        side = random.choice(["white", "black"])
    player_is_white = (side == "white")
    print(f"You play {'White' if player_is_white else 'Black'}. "
          f"Engine uses {args.sims} simulations per move.\n")

    board = chess.Board(args.fen)
    move_history: list[chess.Move] = []
    print_board(board, player_is_white)

    while not board.is_game_over():
        player_turn = (board.turn == chess.WHITE) == player_is_white

        if player_turn:
            action = get_player_move(board)
            if action is None:
                print("Bye!")
                return
            if action == "board":
                print_board(board, player_is_white)
                continue
            if action == "undo":
                if len(move_history) >= 2:
                    board.pop(); board.pop()
                    move_history.pop(); move_history.pop()
                    print("  Took back two half-moves.")
                    print_board(board, player_is_white)
                else:
                    print("  Nothing to undo.")
                continue
            board.push(action)
            move_history.append(action)
            print_board(board, player_is_white)

        else:
            side_name = "White" if board.turn == chess.WHITE else "Black"
            print(f"Engine ({side_name}) thinking ({args.sims} sims)...", flush=True)
            move = engine.search(board, n_simulations=args.sims)
            board.push(move)
            move_history.append(move)
            print(f"Engine plays: {move.uci()}")
            print_board(board, player_is_white)

    # Game over
    result = board.result()
    outcome = board.outcome()
    if outcome and outcome.winner is not None:
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        reason = outcome.termination.name.replace("_", " ").lower()
        print(f"Game over: {winner} wins by {reason}. Result: {result}")
    else:
        term = outcome.termination.name.replace("_", " ").lower() if outcome else "unknown"
        print(f"Game over: Draw by {term}. Result: {result}")


if __name__ == "__main__":
    main()
