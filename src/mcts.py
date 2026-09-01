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
    is_expanded: bool = False   # distinguishes "never evaluated" from "terminal"

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
    """Batched PUCT search with virtual loss and tree reuse.

    Leaves are collected in groups and evaluated in ONE forward pass per net:
    at batch 1 a position costs ~1.35 ms, in a batch of 200 ~0.15 ms, so the
    batching is worth ~9x on the network side. Virtual loss keeps the parallel
    descents from all piling onto the same branch.

    The tree also survives between moves: after our move and the opponent's
    reply, the subtree below that line already carries visits, and re-rooting
    into it is free extra search. Per-node Q is stored in that node's own
    side-to-move perspective, which is what makes re-rooting sound.
    """

    # Pretend a node in flight already lost, so other descents in the same
    # batch avoid it. A node's Q is its own perspective, and the parent selects
    # on (1 - childQ), so "looks bad to the parent" means pushing childQ to 1.
    VIRTUAL_LOSS = 1.0

    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 device: torch.device, c_puct: float = 1.4,
                 batch_size: int = 16):
        self.policy = policy_net.eval()
        self.value  = value_net.eval()
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self._root: Optional[MCTSNode] = None
        self._root_stack: Optional[list] = None
        self._root_origin: Optional[str] = None   # EPD of the game's start
        self.last_reused = 0        # visits inherited by the last search

    # -- tree reuse ---------------------------------------------------------

    def reset(self) -> None:
        """Drop the retained tree (call between games)."""
        self._root, self._root_stack, self._root_origin = None, None, None

    def _get_root(self, board: chess.Board) -> MCTSNode:
        """Re-root into the retained tree when this board continues it.

        A matching move prefix alone is not sufficient: two games started from
        different FENs can share one. Comparing the starting position as well
        makes "same game, further along" exact.
        """
        stack = list(board.move_stack)
        origin = board.root().epd()
        if (self._root is not None and self._root_stack is not None
                and origin == self._root_origin
                and len(stack) >= len(self._root_stack)
                and stack[:len(self._root_stack)] == self._root_stack):
            node = self._root
            for move in stack[len(self._root_stack):]:
                node = node.children.get(move)
                if node is None:
                    break
            if node is not None and node.is_expanded:
                self.last_reused = node.visit_count
                self._root, self._root_stack = node, stack
                return node
        self.last_reused = 0
        self._root, self._root_stack, self._root_origin = MCTSNode(), stack, origin
        return self._root

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, boards: list[chess.Board]):
        """One batched forward pass per net for a list of positions."""
        batch = torch.from_numpy(
            np.stack([encode_board(b) for b in boards])).to(self.device)
        logits = self.policy(batch).cpu().numpy()
        values = torch.sigmoid(self.value(batch)).cpu().numpy()
        return logits, values

    def _expand_from_logits(self, node: MCTSNode, board: chess.Board,
                            logits: np.ndarray) -> None:
        """Assign priors to all legal children from a precomputed policy row."""
        legal = list(board.legal_moves)
        if legal:
            idx = np.array([encode_move_idx(board, m) for m in legal])
            leg = logits[idx]
            leg = leg - leg.max()               # numerical stability
            priors = np.exp(leg)
            priors /= priors.sum()
            for move, prior in zip(legal, priors):
                node.children[move] = MCTSNode(prior=float(prior))
        node.is_expanded = True

    # -- search -------------------------------------------------------------

    def _descend(self, root: MCTSNode, board: chess.Board):
        """Walk to a leaf, applying virtual loss along the way."""
        node, path = root, [root]
        sim_board = board.copy(stack=False)
        while node.is_expanded and node.children:
            move = max(node.children,
                       key=lambda m: node.children[m].puct(node.visit_count, self.c_puct))
            node = node.children[move]
            sim_board.push(move)
            path.append(node)
        for nd in path:
            nd.visit_count += 1
            nd.value_sum += self.VIRTUAL_LOSS
        return path, sim_board, node

    @staticmethod
    def _backprop(path: list[MCTSNode], value: float, virtual_loss: float) -> None:
        """Undo the virtual loss, then apply the real value.

        path[0]=root … path[-1]=leaf. The value flips at each level: the leaf
        gets `value` in its own perspective, its parent gets `1 - value`.
        """
        for nd in path:
            nd.visit_count -= 1
            nd.value_sum -= virtual_loss
        for i, nd in enumerate(reversed(path)):
            nd.visit_count += 1
            nd.value_sum += value if i % 2 == 0 else (1.0 - value)

    def search(self, board: chess.Board, n_simulations: int = 400) -> chess.Move:
        """Run MCTS and return the most-visited move."""
        root = self._get_root(board)

        if not root.is_expanded:
            logits, values = self._evaluate([board])
            self._expand_from_logits(root, board, logits[0])
            root.visit_count += 1
            root.value_sum += float(values[0])

        # n_simulations is NEW work per move; inherited visits are a bonus on
        # top. Subtracting them instead would let a large reused tree reduce
        # the search to nothing and play on stale statistics.
        remaining = n_simulations
        while remaining > 0:
            n = min(self.batch_size, remaining)
            leaves = [self._descend(root, board) for _ in range(n)]

            # Terminal positions need no network call — keep them out of the batch.
            pending = [(p, b, nd) for p, b, nd in leaves if not b.is_game_over()]
            logits = values = None
            if pending:
                logits, values = self._evaluate([b for _, b, _ in pending])

            k = 0
            for path, sim_board, node in leaves:
                if sim_board.is_game_over():
                    # Checkmate is a loss for the side to move; anything else a draw.
                    value = 0.0 if sim_board.is_checkmate() else 0.5
                    node.is_expanded = True
                else:
                    if not node.is_expanded:
                        self._expand_from_logits(node, sim_board, logits[k])
                    value = float(values[k])
                    k += 1
                self._backprop(path, value, self.VIRTUAL_LOSS)
            remaining -= n

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

def _trunk_arch_from_state(state: dict) -> tuple[int, int]:
    """Infer (channels, n_blocks) from a stem+body state dict."""
    n_blocks = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("body."))
    channels = state["stem.3.weight"].shape[0]
    return channels, n_blocks


def load_models(device: torch.device, policy_path: Optional[Path] = None,
                value_path: Optional[Path] = None):
    """Load both CNN models. Prefers the *_large.pt checkpoints when present;
    each architecture is inferred from the checkpoint itself."""
    _base = Path(__file__).resolve().parent.parent

    # Import model classes from training scripts
    import sys
    sys.path.insert(0, str(_base / "src"))
    from train_cnn import PositionEvalCNN
    from train_policy import PolicyCNN

    def _pick(explicit: Optional[Path], large: str, small: str) -> Path:
        if explicit is not None:
            return explicit
        large_path = _base / "models" / large
        return large_path if large_path.exists() else _base / "models" / small

    value_path = _pick(value_path, "position_eval_cnn_large.pt", "position_eval_cnn.pt")
    policy_path = _pick(policy_path, "policy_cnn_large.pt", "policy_cnn.pt")

    if not value_path.exists():
        raise FileNotFoundError(f"Value model not found: {value_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy model not found: {policy_path}")

    value_state = torch.load(value_path, map_location=device, weights_only=True)
    v_ch, v_blocks = _trunk_arch_from_state(value_state)
    value_net = PositionEvalCNN(v_ch, v_blocks)
    value_net.load_state_dict(value_state)
    value_net = value_net.to(device).eval()
    print(f"Value : {value_path.name} (channels={v_ch}, blocks={v_blocks})")

    policy_state = torch.load(policy_path, map_location=device, weights_only=True)
    p_ch, p_blocks = _trunk_arch_from_state(policy_state)
    policy_ch = policy_state["policy_head.0.weight"].shape[0]
    policy_net = PolicyCNN(p_ch, p_blocks, policy_ch)
    policy_net.load_state_dict(policy_state)
    policy_net = policy_net.to(device).eval()
    print(f"Policy: {policy_path.name} (channels={p_ch}, blocks={p_blocks})")

    return policy_net, value_net
