"""Train a small CNN to evaluate chess positions (predict outcome from the side-to-move's perspective)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

POSITIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "positions"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "position_eval_cnn.pt"


def load_dataset():
    boards, labels = [], []
    for npz_path in sorted(POSITIONS_DIR.glob("*.npz")):
        data = np.load(npz_path)
        if len(data["labels"]) == 0:
            continue
        boards.append(data["boards"])
        labels.append(data["labels"])
    return np.concatenate(boards), np.concatenate(labels)


class PositionDataset(Dataset):
    def __init__(self, boards: np.ndarray, labels: np.ndarray):
        self.boards = torch.from_numpy(boards).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.boards[idx], self.labels[idx]


class PositionEvalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x).squeeze(-1)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    boards, labels = load_dataset()
    print(f"{len(labels):,} positions loaded")

    dataset = PositionDataset(boards, labels)
    n_val = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = PositionEvalCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    max_epochs = 20
    patience = 3
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
              f"val_winner_acc={acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No val_loss improvement for {patience} epochs, stopping early.")
                break

    # Work around a sandbox quirk: writing directly into a directory this process
    # didn't itself create can fail with "Operation not permitted", even though the
    # directory is writable. Writing into a freshly-created subdir and renaming up
    # avoids it.
    tmp_dir = MODEL_PATH.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / MODEL_PATH.name
    torch.save(best_state, tmp_path)
    tmp_path.rename(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH} (best val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
