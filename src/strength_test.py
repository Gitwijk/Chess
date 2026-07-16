"""Estimate the MCTS engine's playing strength against Stockfish.

Plays matches against Stockfish with UCI_LimitStrength at several Elo levels
(alternating colors), reports score per level, and interpolates an Elo
estimate from the results:  elo_diff = 400 * log10(score / (1 - score)).

Game records are appended to logs/strength_games.pgn.

Usage:
    python src/strength_test.py                                 # defaults
    python src/strength_test.py --elo-list 1320,1600 --games 8 --sims 300
    python src/strength_test.py --games 2 --max-plies 20        # smoke test
"""

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from mcts import MCTS, load_models  # noqa: E402

PGN_OUT = _BASE / "logs" / "strength_games.pgn"
STOCKFISH_MIN_ELO = 1320   # UCI_Elo lower bound in Stockfish


def play_game(engine: MCTS, sf: chess.engine.SimpleEngine, sims: int,
              we_are_white: bool, max_plies: int, sf_movetime: float) -> tuple[float, chess.pgn.Game]:
    """Returns (our score: 1/0.5/0, game record)."""
    board = chess.Board()
    while not board.is_game_over() and board.ply() < max_plies:
        our_turn = (board.turn == chess.WHITE) == we_are_white
        if our_turn:
            move = engine.search(board, n_simulations=sims)
        else:
            move = sf.play(board, chess.engine.Limit(time=sf_movetime)).move
        board.push(move)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        score = 0.5   # draw or adjudicated at max_plies
    else:
        score = 1.0 if (outcome.winner == chess.WHITE) == we_are_white else 0.0

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "chess-ml MCTS" if we_are_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if we_are_white else "chess-ml MCTS"
    game.headers["Date"] = date.today().isoformat()
    game.headers["Result"] = board.result(claim_draw=True) if outcome else "1/2-1/2"
    return score, game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo-list", default="1320,1600,1900",
                    help="Comma-separated Stockfish UCI_Elo levels")
    ap.add_argument("--games", type=int, default=12, help="Games per level")
    ap.add_argument("--sims", type=int, default=300, help="MCTS simulations per move")
    ap.add_argument("--sf-movetime", type=float, default=0.05,
                    help="Stockfish seconds per move (default 0.05)")
    ap.add_argument("--max-plies", type=int, default=220,
                    help="Adjudicate as draw beyond this many plies")
    args = ap.parse_args()

    levels = [int(x) for x in args.elo_list.split(",")]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy_net, value_net = load_models(device)
    engine = MCTS(policy_net, value_net, device)
    print(f"MCTS engine ready ({args.sims} sims/move, device {device})")

    PGN_OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with chess.engine.SimpleEngine.popen_uci("stockfish") as sf:
        for elo in levels:
            sf.configure({"UCI_LimitStrength": True,
                          "UCI_Elo": max(elo, STOCKFISH_MIN_ELO)})
            score = 0.0
            wdl = [0, 0, 0]
            for g in range(args.games):
                we_are_white = g % 2 == 0
                s, game = play_game(engine, sf, args.sims, we_are_white,
                                    args.max_plies, args.sf_movetime)
                score += s
                wdl[0 if s == 1.0 else (1 if s == 0.5 else 2)] += 1
                game.headers["Event"] = f"strength test vs SF elo {elo}"
                with open(PGN_OUT, "a") as f:
                    print(game, file=f, end="\n\n")
                print(f"  elo {elo}  game {g + 1}/{args.games}: "
                      f"{'W' if s == 1 else ('D' if s == 0.5 else 'L')} "
                      f"(running {score}/{g + 1})", flush=True)

            frac = score / args.games
            results.append((elo, frac, wdl))
            print(f"Level {elo}: {wdl[0]}W {wdl[1]}D {wdl[2]}L  score {frac:.2f}")

    print("\n=== Summary ===")
    for elo, frac, wdl in results:
        if 0.0 < frac < 1.0:
            diff = 400 * math.log10(frac / (1 - frac))
            est = f"→ engine ≈ {elo + diff:+.0f} Elo (diff {diff:+.0f})"
        else:
            est = "→ score too extreme for an estimate"
        print(f"vs SF {elo}: {wdl[0]}W {wdl[1]}D {wdl[2]}L  score {frac:.2f}  {est}")
    print(f"\nGame records: {PGN_OUT}")


if __name__ == "__main__":
    main()
