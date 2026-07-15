"""Train a CNN policy head to predict moves from chess positions.

Loads (board, move) pairs from data/processed/policy/, where each move is
encoded as from_square * 64 + to_square (0–4095). Trains with CrossEntropyLoss.

Usage:
    python src/train_policy.py                    # fresh training
    python src/train_policy.py --finetune         # warm-start backbone from value model
    python src/train_policy.py --resume           # continue from saved policy model
    python src/train_policy.py --epochs 30
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

_BASE = Path(__file__).resolve().parent.parent
DATA_DIR = _BASE / "data" / "processed" / "policy"
MODEL_PATH = _BASE / "models" / "policy_cnn.pt"
VALUE_MODEL_PATH = _BASE / "models" / "position_eval_cnn.pt"


def load_dataset():
    boards, moves = [], []
    for npz_path in sorted(DATA_DIR.glob("*.npz")):
        data = np.load(npz_path)
        if len(data["moves"]) == 0:
            continue
        boards.append(data["boards"])
        moves.append(data["moves"])
    if not boards:
        raise SystemExit(f"No data found in {DATA_DIR}")
    return np.concatenate(boards), np.concatenate(moves)


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
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(17, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.body = nn.Sequential(ResBlock(128), ResBlock(128), ResBlock(128))
        # Policy head: 1×1 conv preserves spatial structure → flatten → 4096 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),
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
                    help="Warm-start backbone (stem+body) from value model weights")
    ap.add_argument("--resume", action="store_true",
                    help="Load saved policy model and continue at half the LR")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from {DATA_DIR.name}/...")
    boards, moves = load_dataset()
    print(f"{len(moves):,} positions loaded")

    dataset = PolicyDataset(boards, moves)
    n_val = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)

    model = PolicyCNN().to(device)

    if args.resume and MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        lr = 5e-5
        print(f"Resumed from {MODEL_PATH} (lr={lr})")
    elif args.finetune and VALUE_MODEL_PATH.exists():
        value_state = torch.load(VALUE_MODEL_PATH, map_location=device, weights_only=True)
        backbone = {k: v for k, v in value_state.items()
                    if k.startswith("stem.") or k.startswith("body.")}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        print(f"Loaded {len(backbone)} backbone weights from value model "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        lr = 5e-5
    else:
        lr = 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 5
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
              f"top1={c1/n_total:.4f}  top3={c3/n_total:.4f}  top5={c5/n_total:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No val_loss improvement for {patience} epochs, stopping.")
                break

    tmp_dir = MODEL_PATH.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / MODEL_PATH.name
    torch.save(best_state, tmp_path)
    tmp_path.rename(MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH} (best val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
