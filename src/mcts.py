"""Monte Carlo Tree Search using the value + policy CNN models.

Conventions:
  - Board tensor encoding and move index (0-4095) match extract_policy.py.
  - Q at each node is stored from the node's own side-to-move perspective.
  - PUCT selection uses (1 - child.Q) to convert to parent's perspective.
  - Value head logit → sigmoid → win probability for side to move.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Board / move encoding (must match extract_policy.py exactly)
# ---------------------------------------------------------------------------

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,  (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,  (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,  (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,  (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,  (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10, (chess.KING,   chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    b = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((17, 8, 8), dtype=np.float32)
    for square, piece in b.piece_map().items():
        row, col = divmod(square, 8)
        planes[PIECE_TO_PLANE[(piece.piece_type, piece.color)], row, col] = 1.0
    if b.has_kingside_castling_rights(chess.WHITE):  planes[12] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE): planes[13] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):  planes[14] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK): planes[15] = 1.0
    if b.ep_square is not None:
        row, col = divmod(b.ep_square, 8)
        planes[16, row, col] = 1.0
    return planes


def encode_move_idx(board: chess.Board, move: chess.Move) -> int:
    if board.turn == chess.WHITE:
        return move.from_square * 64 + move.to_square
    return chess.square_mirror(move.from_square) * 64 + chess.square_mirror(move.to_square)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)   # chess.Move → MCTSNode

    @property
    def Q(self) -> float:
        """Mean value from this node's own side-to-move perspective."""
        return self.value_sum / self.visit_count if self.visit_count else 0.5

    def puct(self, parent_visits: int, c_puct: float) -> float:
        # Use (1 - Q) to convert child's perspective → parent's perspective.
        exploitation = 1.0 - self.Q
        exploration = c_puct * self.prior * math.sqrt(max(1, parent_visits)) / (1 + self.visit_count)
        return exploitation + exploration


# ---------------------------------------------------------------------------
# MCTS engine
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 device: torch.device, c_puct: float = 1.4):
        self.policy = policy_net.eval()
        self.value  = value_net.eval()
        self.device = device
        self.c_puct = c_puct

    def _tensor(self, board: chess.Board) -> torch.Tensor:
        arr = encode_board(board)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _expand(self, node: MCTSNode, board: chess.Board) -> None:
        """Run policy head; assign priors to all legal child nodes."""
        legal = list(board.legal_moves)
        if not legal:
            return
        with torch.no_grad():
            logits = self.policy(self._tensor(board))[0].cpu().numpy()

        indices = np.array([encode_move_idx(board, m) for m in legal])
        leg_logits = logits[indices]
        leg_logits -= leg_logits.max()          # numerical stability
        priors = np.exp(leg_logits)
        priors /= priors.sum()

        for move, prior in zip(legal, priors):
            node.children[move] = MCTSNode(prior=float(prior))

    def _leaf_value(self, board: chess.Board) -> float:
        """Win probability for the side to move at this board."""
        if board.is_checkmate():
            return 0.0   # side to move is in checkmate → 0
        if board.is_game_over():
            return 0.5   # draw
        with torch.no_grad():
            logit = self.value(self._tensor(board)).item()
        return torch.sigmoid(torch.tensor(logit)).item()

    def search(self, board: chess.Board, n_simulations: int = 400) -> chess.Move:
        """Run MCTS and return the best move."""
        root = MCTSNode()
        self._expand(root, board)
        root.visit_count = 1

        for _ in range(n_simulations):
            node = root
            sim_board = board.copy(stack=False)
            path = [node]

            # --- Selection ---
            while node.children and node.visit_count > 1:
                best_move = max(
                    node.children,
                    key=lambda m: node.children[m].puct(node.visit_count, self.c_puct),
                )
                node = node.children[best_move]
                sim_board.push(best_move)
                path.append(node)

            # --- Expansion (if not terminal and not yet expanded) ---
            if not node.children and not sim_board.is_game_over():
                self._expand(node, sim_board)

            # --- Evaluation ---
            value = self._leaf_value(sim_board)

            # --- Backpropagation ---
            # path[0]=root ... path[-1]=leaf
            # Flip value at each alternating level: leaf gets `value`,
            # parent gets `1-value` (opponent's perspective), etc.
            for i, n in enumerate(reversed(path)):
                n.visit_count += 1
                n.value_sum += value if i % 2 == 0 else (1.0 - value)

        if not root.children:
            raise ValueError("No legal moves from this position")

        return max(root.children, key=lambda m: root.children[m].visit_count)

    def move_stats(self, root: Optional[MCTSNode]) -> str:
        """Human-readable top-5 moves by visit count (for debugging)."""
        if root is None or not root.children:
            return "(no stats)"
        ranked = sorted(root.children.items(),
                        key=lambda kv: kv[1].visit_count, reverse=True)[:5]
        lines = []
        for move, node in ranked:
            lines.append(f"  {move.uci():6s}  visits={node.visit_count:5d}  Q={node.Q:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_models(device: torch.device):
    """Load both CNN models from the standard model paths."""
    _base = Path(__file__).resolve().parent.parent

    # Import model classes from training scripts
    import sys
    sys.path.insert(0, str(_base / "src"))
    from train_cnn import PositionEvalCNN
    from train_policy import PolicyCNN

    value_path  = _base / "models" / "position_eval_cnn.pt"
    policy_path = _base / "models" / "policy_cnn.pt"

    if not value_path.exists():
        raise FileNotFoundError(f"Value model not found: {value_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy model not found: {policy_path}")

    value_net = PositionEvalCNN()
    value_net.load_state_dict(
        torch.load(value_path, map_location=device, weights_only=True))
    value_net = value_net.to(device).eval()

    policy_net = PolicyCNN()
    policy_net.load_state_dict(
        torch.load(policy_path, map_location=device, weights_only=True))
    policy_net = policy_net.to(device).eval()

    return policy_net, value_net
