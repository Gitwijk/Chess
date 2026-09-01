"""Estimate the MCTS engine's playing strength against Stockfish.

Plays matches against Stockfish with UCI_LimitStrength at several Elo levels
and reports, per level, the score with a 95% confidence interval and the
implied engine Elo.

Reliability features (the v1 harness was too noisy to verify improvements):
  - Opening book: each game starts from a different short opening line, played
    twice with colors reversed, so a single pet line cannot dominate the score.
  - Paired sampling: game 2i and 2i+1 use the SAME opening, colors swapped.
  - Longer default Stockfish time (0.3 s/move) — SF's limited-strength
    calibration is unreliable at very short time controls.
  - 95% CI on the score, propagated into an Elo range.

Game records are appended to logs/strength_games.pgn.

Usage:
    python src/strength_test.py                                  # defaults
    python src/strength_test.py --elo-list 2000,2300 --games 40
    python src/strength_test.py --games 2 --max-plies 20         # smoke test
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

# Short, mainstream opening lines (UCI). Each is played twice, colors reversed.
OPENING_BOOK = [
    [],                                            # start position
    ["e2e4", "e7e5"],                              # Open Game
    ["e2e4", "c7c5"],                              # Sicilian
    ["e2e4", "e7e6"],                              # French
    ["e2e4", "c7c6"],                              # Caro-Kann
    ["d2d4", "d7d5"],                              # Closed Game
    ["d2d4", "g8f6", "c2c4", "e7e6"],              # Indian
    ["d2d4", "g8f6", "c2c4", "g7g6"],              # King's Indian
    ["g1f3", "d7d5"],                              # Réti
    ["c2c4", "e7e5"],                              # English
]


def elo_diff_from_score(score: float) -> float | None:
    """Standard logistic Elo difference; None if the score is 0 or 1."""
    if not 0.0 < score < 1.0:
        return None
    return 400 * math.log10(score / (1 - score))


def play_game(engine: MCTS, sf: chess.engine.SimpleEngine, sims: int,
              we_are_white: bool, max_plies: int, sf_movetime: float,
              opening: list[str]) -> tuple[float, chess.pgn.Game]:
    """Returns (our score: 1/0.5/0, game record)."""
    engine.reset()   # never carry a retained tree across games
    board = chess.Board()
    for uci in opening:
        board.push(chess.Move.from_uci(uci))

    while not board.is_game_over() and board.ply() < max_plies:
        our_turn = (board.turn == chess.WHITE) == we_are_white
        if our_turn:
            move = engine.search(board, n_simulations=sims)
        else:
            move = sf.play(board, chess.engine.Limit(time=sf_movetime)).move
        board.push(move)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        score = 0.5   # draw, or adjudicated at max_plies
    else:
        score = 1.0 if (outcome.winner == chess.WHITE) == we_are_white else 0.0

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "chess-ml MCTS" if we_are_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if we_are_white else "chess-ml MCTS"
    game.headers["Date"] = date.today().isoformat()
    game.headers["Result"] = board.result(claim_draw=True) if outcome else "1/2-1/2"
    return score, game


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elo-list", default="1900,2200,2500",
                    help="Comma-separated Stockfish UCI_Elo levels")
    ap.add_argument("--games", type=int, default=30,
                    help="Games per level (rounded up to an even number)")
    ap.add_argument("--sims", type=int, default=300, help="MCTS simulations per move")
    ap.add_argument("--batch", type=int, default=32,
                    help="MCTS leaves evaluated per forward pass "
                         "(1 = unbatched; higher is faster)")
    ap.add_argument("--sf-movetime", type=float, default=0.3,
                    help="Stockfish seconds per move (default 0.3)")
    ap.add_argument("--max-plies", type=int, default=220,
                    help="Adjudicate as draw beyond this many plies")
    ap.add_argument("--label", default="",
                    help="Tag written into the PGN Event header (e.g. 'large-value-net')")
    ap.add_argument("--value-model", type=Path, default=None,
                    help="Value checkpoint to use (default: prefer *_large.pt). "
                         "Set explicitly to A/B two checkpoints under identical conditions.")
    ap.add_argument("--policy-model", type=Path, default=None,
                    help="Policy checkpoint to use (default: prefer *_large.pt)")
    args = ap.parse_args()

    levels = [int(x) for x in args.elo_list.split(",")]
    n_games = args.games + (args.games % 2)   # keep color-paired

    def _resolve(p: Path | None) -> Path | None:
        if p is None:
            return None
        return p if p.is_absolute() else _BASE / p

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy_net, value_net = load_models(device,
                                        policy_path=_resolve(args.policy_model),
                                        value_path=_resolve(args.value_model))
    engine = MCTS(policy_net, value_net, device, batch_size=args.batch)
    print(f"MCTS engine ready ({args.sims} sims/move, device {device})")
    print(f"{n_games} games/level, SF {args.sf_movetime}s/move, "
          f"{len(OPENING_BOOK)} book lines\n")

    PGN_OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with chess.engine.SimpleEngine.popen_uci("stockfish") as sf:
        for elo in levels:
            sf.configure({"UCI_LimitStrength": True,
                          "UCI_Elo": max(elo, STOCKFISH_MIN_ELO)})
            score = 0.0
            wdl = [0, 0, 0]
            for g in range(n_games):
                # Pair games: same opening, colors reversed.
                opening = OPENING_BOOK[(g // 2) % len(OPENING_BOOK)]
                we_are_white = g % 2 == 0
                s, game = play_game(engine, sf, args.sims, we_are_white,
                                    args.max_plies, args.sf_movetime, opening)
                score += s
                wdl[0 if s == 1.0 else (1 if s == 0.5 else 2)] += 1
                game.headers["Event"] = (f"strength test vs SF elo {elo}"
                                         + (f" [{args.label}]" if args.label else ""))
                with open(PGN_OUT, "a") as f:
                    print(game, file=f, end="\n\n")
                print(f"  elo {elo}  game {g + 1}/{n_games}: "
                      f"{'W' if s == 1 else ('D' if s == 0.5 else 'L')} "
                      f"(running {score}/{g + 1})", flush=True)

            frac = score / n_games
            # 95% CI on the mean score (normal approx; each game in {0, .5, 1}).
            se = math.sqrt(max(frac * (1 - frac), 1e-9) / n_games)
            lo, hi = max(0.0, frac - 1.96 * se), min(1.0, frac + 1.96 * se)
            results.append((elo, frac, wdl, lo, hi))
            print(f"Level {elo}: {wdl[0]}W {wdl[1]}D {wdl[2]}L  "
                  f"score {frac:.3f} [{lo:.3f}, {hi:.3f}]\n")

    print("=== Summary ===")
    for elo, frac, wdl, lo, hi in results:
        diff = elo_diff_from_score(frac)
        if diff is None:
            est = "score too extreme for an estimate"
        else:
            d_lo, d_hi = elo_diff_from_score(lo), elo_diff_from_score(hi)
            rng = ""
            if d_lo is not None and d_hi is not None:
                rng = f"  [{elo + d_lo:.0f}, {elo + d_hi:.0f}]"
            est = f"engine ~ {elo + diff:.0f} Elo{rng}"
        print(f"vs SF {elo}: {wdl[0]}W {wdl[1]}D {wdl[2]}L  "
              f"score {frac:.3f}  -> {est}")

    usable = [(e, f) for e, f, _, _, _ in results if 0.0 < f < 1.0]
    if usable:
        ests = [e + elo_diff_from_score(f) for e, f in usable]
        print(f"\nCombined estimate (mean of {len(ests)} usable levels): "
              f"{sum(ests) / len(ests):.0f} Elo")
    print(f"\nGame records: {PGN_OUT}")


if __name__ == "__main__":
    main()
