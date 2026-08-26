"""Train a CNN to evaluate chess positions (predict win probability for the side to move).

DATA_DIR can point to either:
  data/processed/evals/     — Stockfish-annotated positions (recommended)
  data/processed/positions/ — game-outcome labels from PGN parsing

Usage:
    python src/train_cnn.py                                   # small net (128ch/3b)
    python src/train_cnn.py --resume                          # continue training
    python src/train_cnn.py --channels 192 --blocks 6 --batch 1024 \
        --lr 2e-4 --epochs 14 --out models/position_eval_cnn_large.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

_BASE = Path(__file__).resolve().parent.parent
DATA_DIR = _BASE / "data" / "processed" / "evals"
DEFAULT_OUT = _BASE / "models" / "position_eval_cnn.pt"

# 28M positions x 1088 bytes (int8) ~= 30 GB — safe on 64 GB systems.
DEFAULT_MAX_POSITIONS = 28_000_000


def load_dataset(max_positions: int = DEFAULT_MAX_POSITIONS):
    """Load up to max_positions into a PREALLOCATED array.

    Preallocating avoids np.concatenate's 2x RAM peak, which would spike to
    ~60 GB at this dataset size. Shard order is shuffled so the subset is not
    biased toward whichever shards happen to come first.
    """
    paths = sorted(DATA_DIR.glob("*.npz"))
    rng = np.random.default_rng(42)
    order = rng.permutation(len(paths))

    boards = np.empty((max_positions, 17, 8, 8), dtype=np.int8)
    labels = np.empty(max_positions, dtype=np.float32)
    pos = 0
    for i in order:
        if pos >= max_positions:
            break
        data = np.load(paths[i])
        n = len(data["labels"])
        if n == 0:
            continue
        take = min(n, max_positions - pos)
        if take < n:
            idx = np.sort(rng.choice(n, size=take, replace=False))
            boards[pos:pos + take] = data["boards"][idx]
            labels[pos:pos + take] = data["labels"][idx]
        else:
            boards[pos:pos + take] = data["boards"]
            labels[pos:pos + take] = data["labels"]
        pos += take

    if pos == 0:
        raise SystemExit(f"No data found in {DATA_DIR}")
    print(f"Loaded {pos:,} positions (cap={max_positions:,})")
    return boards[:pos], labels[:pos]


class PositionDataset(Dataset):
    def __init__(self, boards: np.ndarray, labels: np.ndarray):
        self.boards = torch.from_numpy(boards)  # kept as int8 to save ~4x RAM
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.boards[idx].float(), self.labels[idx]


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class PositionEvalCNN(nn.Module):
    """Stem keeps the fixed Sequential layout (conv,bn,relu,conv,bn,relu) so the
    architecture can be inferred from a state dict (see mcts.load_models)."""

    def __init__(self, channels: int = 128, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(17, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2), nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 64), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Load model from --out and continue training")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None,
                    help="Peak LR (default 1e-4 fresh, 5e-5 on --resume)")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out_path = args.out if args.out.is_absolute() else _BASE / args.out

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from {DATA_DIR.name}/...")
    boards, labels = load_dataset(args.max_positions)

    dataset = PositionDataset(boards, labels)
    n_val = int(0.05 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = PositionEvalCNN(args.channels, args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PositionEvalCNN(channels={args.channels}, blocks={args.blocks}): "
          f"{n_params:,} params")

    lr = args.lr
    if args.resume and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
        lr = lr or 5e-5
        print(f"Resumed from {out_path} (lr={lr})")
    else:
        lr = lr or 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=lr * 0.05)
    criterion = nn.BCEWithLogitsLoss()

    max_epochs = args.epochs
    patience = 4
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for boards_batch, labels_batch in train_loader:
            boards_batch, labels_batch = boards_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(boards_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(labels_batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        correct_sign = 0
        n_decisive = 0
        with torch.no_grad():
            for boards_batch, labels_batch in val_loader:
                boards_batch, labels_batch = boards_batch.to(device), labels_batch.to(device)
                logits = model(boards_batch)
                val_loss += criterion(logits, labels_batch).item() * len(labels_batch)
                preds = torch.sigmoid(logits)
                # "Correct" if the predicted favorite matches the actual winner (ignoring draws).
                decisive = labels_batch != 0.5
                correct_sign += ((preds[decisive] > 0.5) == (labels_batch[decisive] > 0.5)).sum().item()
                n_decisive += decisive.sum().item()
        val_loss /= len(val_ds)
        acc = correct_sign / n_decisive if n_decisive else float("nan")

        print(f"Epoch {epoch}/{max_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_winner_acc={acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            # Checkpoint immediately: a killed run resumes from the best epoch.
            tmp_dir = out_path.parent / "_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            ckpt = tmp_dir / out_path.name
            torch.save(best_state, ckpt)
            ckpt.rename(out_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No val_loss improvement for {patience} epochs, stopping early.")
                break

    # Work around a sandbox quirk: writing directly into a directory this process
    # didn't itself create can fail with "Operation not permitted", even though the
    # directory is writable. Writing into a freshly-created subdir and renaming up
    # avoids it.
    tmp_dir = out_path.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / out_path.name
    torch.save(best_state, tmp_path)
    tmp_path.rename(out_path)
    print(f"\nSaved model to {out_path} (best val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
