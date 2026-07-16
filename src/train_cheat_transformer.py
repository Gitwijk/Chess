"""Train a Transformer cheat detector on per-move feature sequences.

Instead of per-game averages (train_cheat_detector.py), this model sees the
whole sequence of (move rank, policy prob, eval swing) per game and can pick
up patterns like "engine-perfect stretches" that averages wash out.

Input:  data/processed/cheat_features_sequences.parquet (extract with
        `extract_cheat_features.py --sequences`)
Output: models/cheat_transformer.pt + AUC report

Usage:
    python src/train_cheat_transformer.py
    python src/train_cheat_transformer.py --data other.parquet --epochs 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

_BASE = Path(__file__).resolve().parent.parent
DEFAULT_DATA = _BASE / "data" / "processed" / "cheat_features_sequences.parquet"
MODEL_PATH = _BASE / "models" / "cheat_transformer.pt"

MAX_LEN = 96          # truncate longer games (own moves after opening skip)
D_MODEL = 64


class SeqDataset(Dataset):
    """Per (game, player): [L, 4] float tensor of per-move features + label."""

    def __init__(self, df: pd.DataFrame):
        self.items = []
        for _, r in df.iterrows():
            rank = np.asarray(r["seq_rank"], dtype=np.float32)[:MAX_LEN]
            prob = np.asarray(r["seq_prob"], dtype=np.float32)[:MAX_LEN]
            swing = np.asarray(r["seq_swing"], dtype=np.float32)[:MAX_LEN]
            feats = np.stack([
                np.log1p(np.clip(rank, 0, 50)),   # 0 = engine top choice
                (rank == 0).astype(np.float32),   # explicit top-1 flag
                prob,
                np.clip(swing, -0.5, 0.5),
            ], axis=1)
            self.items.append((feats, float(r["label_tos"]), r["player"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    feats, labels, players = zip(*batch)
    lens = [len(f) for f in feats]
    L = max(lens)
    x = torch.zeros(len(batch), L, 4)
    pad = torch.ones(len(batch), L, dtype=torch.bool)   # True = padding
    for i, f in enumerate(feats):
        x[i, :len(f)] = torch.from_numpy(f)
        pad[i, :len(f)] = False
    return x, pad, torch.tensor(labels), list(players)


class CheatTransformer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True)
        # enable_nested_tensor's fast path is not implemented on MPS
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, pad_mask):
        L = x.shape[1]
        pos = torch.arange(L, device=x.device).clamp(max=MAX_LEN - 1)
        h = self.input_proj(x) + self.pos_emb(pos)[None]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # masked mean pool
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def player_split(players, test_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(np.asarray(players, dtype=str))
    rng.shuffle(uniq)
    test = set(uniq[:int(len(uniq) * test_frac)])
    return test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels, players = [], [], []
    for x, pad, y, ps in loader:
        logits = model(x.to(device), pad.to(device))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.tolist())
        players.extend(ps)
    game_auc = roc_auc_score(labels, scores)
    pdf = pd.DataFrame({"player": players, "score": scores, "label": labels})
    pl = pdf.groupby("player").agg(score=("score", "mean"), label=("label", "first"))
    player_auc = roc_auc_score(pl["label"], pl["score"])
    return game_auc, player_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_parquet(args.data)
    df = df[df["title"] != "BOT"].copy()
    print(f"{len(df):,} sequences, {df['player'].nunique():,} players, "
          f"positive rate {df['label_tos'].mean():.3f}")

    test_players = player_split(df["player"].values)
    test_df = df[df["player"].isin(test_players)]
    train_df = df[~df["player"].isin(test_players)]
    print(f"train {len(train_df):,} / test {len(test_df):,} "
          f"(test positives: {test_df['label_tos'].sum():,})")

    train_loader = DataLoader(SeqDataset(train_df), batch_size=args.batch,
                              shuffle=True, collate_fn=collate)
    test_loader = DataLoader(SeqDataset(test_df), batch_size=args.batch,
                             shuffle=False, collate_fn=collate)

    model = CheatTransformer().to(device)
    n_pos = train_df["label_tos"].sum()
    pos_weight = torch.tensor(
        float((len(train_df) - n_pos) / max(n_pos, 1)), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, pad, y, _ in train_loader:
            x, pad, y = x.to(device), pad.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, pad), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        game_auc, player_auc = evaluate(model, test_loader, device)
        marker = ""
        if player_auc > best_auc:
            best_auc = player_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  *"
        print(f"Epoch {epoch}/{args.epochs}  loss={total/len(train_df):.4f}  "
              f"game_auc={game_auc:.4f}  player_auc={player_auc:.4f}{marker}",
              flush=True)

    tmp_dir = MODEL_PATH.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / MODEL_PATH.name
    torch.save(best_state, tmp)
    tmp.rename(MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH} (best player_auc={best_auc:.4f})")


if __name__ == "__main__":
    main()
