"""Train a CNN policy head to predict moves from chess positions.

Loads (board, move) pairs from data/processed/policy/, where each move is
encoded as from_square * 64 + to_square (0–4095). Trains with CrossEntropyLoss
and a per-step cosine learning-rate schedule.

Usage:
    python src/train_policy.py                                # small net, fresh
    python src/train_policy.py --finetune                     # warm-start backbone from value model
    python src/train_policy.py --resume                       # continue from saved policy model
    python src/train_policy.py --channels 192 --blocks 6 --policy-ch 64 \
        --max-positions 28000000 --batch 1024 --lr 2e-4 --epochs 14 \
        --out models/policy_cnn_large.pt                      # scaled-up net
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

_BASE = Path(__file__).resolve().parent.parent
DATA_DIR = _BASE / "data" / "processed" / "policy"
DEFAULT_OUT = _BASE / "models" / "policy_cnn.pt"
VALUE_MODEL_PATH = _BASE / "models" / "position_eval_cnn.pt"

# 20M positions × 1088 bytes (int8) ≈ 21 GB — safe on 64 GB systems.
DEFAULT_MAX_POSITIONS = 20_000_000


def load_dataset(max_positions: int = DEFAULT_MAX_POSITIONS):
    """Load up to max_positions into a PREALLOCATED array.

    Preallocating avoids np.concatenate's 2× RAM peak, which matters at
    28M+ positions (~30 GB); the concatenate approach would spike to ~60 GB.
    Shard order is shuffled so the subset spans all years.
    """
    paths = sorted(DATA_DIR.glob("*.npz"))
    rng = np.random.default_rng(42)
    order = rng.permutation(len(paths))

    boards = np.empty((max_positions, 17, 8, 8), dtype=np.int8)
    moves = np.empty(max_positions, dtype=np.int16)
    pos = 0
    for i in order:
        if pos >= max_positions:
            break
        data = np.load(paths[i])
        n = len(data["moves"])
        if n == 0:
            continue
        take = min(n, max_positions - pos)
        if take < n:
            idx = np.sort(rng.choice(n, size=take, replace=False))
            boards[pos:pos + take] = data["boards"][idx]
            moves[pos:pos + take] = data["moves"][idx]
        else:
            boards[pos:pos + take] = data["boards"]
            moves[pos:pos + take] = data["moves"]
        pos += take

    if pos == 0:
        raise SystemExit(f"No data found in {DATA_DIR}")
    # If the dataset is smaller than the cap, hand back views (keeps the
    # oversized buffer alive, but that only happens when data is scarce).
    boards, moves = boards[:pos], moves[:pos]
    print(f"Loaded {pos:,} positions (cap={max_positions:,})")
    return boards, moves


class PolicyDataset(Dataset):
    def __init__(self, boards: np.ndarray, moves: np.ndarray):
        self.boards = torch.from_numpy(boards)
        # int16 → int64: CrossEntropyLoss requires LongTensor
        self.moves = torch.from_numpy(moves.astype(np.int64))

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        return self.boards[idx].float(), self.moves[idx]


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class PolicyCNN(nn.Module):
    """Stem keeps the fixed Sequential layout (conv,bn,relu,conv,bn,relu) so
    the architecture can be inferred from a state dict (see mcts.load_models)."""

    def __init__(self, channels: int = 128, n_blocks: int = 3, policy_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(17, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2), nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        # Policy head: 1×1 conv preserves spatial structure → flatten → 4096 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_ch, kernel_size=1),
            nn.BatchNorm2d(policy_ch), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_ch * 8 * 8, 4096),
        )

    def forward(self, x):
        return self.policy_head(self.body(self.stem(x)))


def topk_counts(logits: torch.Tensor, targets: torch.Tensor, k: int):
    _, top_k = logits.topk(k, dim=1)
    correct = (top_k == targets.unsqueeze(1)).any(dim=1).sum().item()
    return correct, len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true",
                    help="Warm-start backbone (stem+body) from value model weights "
                         "(only sensible at the default 128/3 architecture)")
    ap.add_argument("--resume", action="store_true",
                    help="Load model from --out and continue training")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None,
                    help="Peak LR (default 1e-4 fresh, 5e-5 on --resume)")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--policy-ch", type=int, default=32)
    ap.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS,
                    help=f"Cap on positions to load (default {DEFAULT_MAX_POSITIONS:,})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Model output path (default models/policy_cnn.pt)")
    args = ap.parse_args()

    out_path = args.out if args.out.is_absolute() else _BASE / args.out

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from {DATA_DIR.name}/...")
    boards, moves = load_dataset(args.max_positions)

    dataset = PolicyDataset(boards, moves)
    n_val = int(0.05 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = PolicyCNN(args.channels, args.blocks, args.policy_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PolicyCNN(channels={args.channels}, blocks={args.blocks}, "
          f"policy_ch={args.policy_ch}): {n_params:,} params")

    lr = args.lr
    if args.resume and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
        lr = lr or 5e-5
        print(f"Resumed from {out_path} (lr={lr})")
    elif args.finetune and VALUE_MODEL_PATH.exists():
        value_state = torch.load(VALUE_MODEL_PATH, map_location=device, weights_only=True)
        backbone = {k: v for k, v in value_state.items()
                    if k.startswith("stem.") or k.startswith("body.")}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        print(f"Loaded {len(backbone)} backbone weights from value model "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        lr = lr or 5e-5
    else:
        lr = lr or 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=lr * 0.05)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 4
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for boards_b, moves_b in train_loader:
            boards_b, moves_b = boards_b.to(device), moves_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(boards_b), moves_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(moves_b)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        c1 = c3 = c5 = n_total = 0
        with torch.no_grad():
            for boards_b, moves_b in val_loader:
                boards_b, moves_b = boards_b.to(device), moves_b.to(device)
                logits = model(boards_b)
                val_loss += criterion(logits, moves_b).item() * len(moves_b)
                k1, n = topk_counts(logits, moves_b, 1)
                k3, _ = topk_counts(logits, moves_b, 3)
                k5, _ = topk_counts(logits, moves_b, 5)
                c1 += k1; c3 += k3; c5 += k5; n_total += n
        val_loss /= len(val_ds)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"top1={c1/n_total:.4f}  top3={c3/n_total:.4f}  top5={c5/n_total:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No val_loss improvement for {patience} epochs, stopping.")
                break

    tmp_dir = out_path.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / out_path.name
    torch.save(best_state, tmp_path)
    tmp_path.rename(out_path)
    print(f"\nSaved to {out_path} (best val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
