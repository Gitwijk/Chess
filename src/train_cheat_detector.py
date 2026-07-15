"""Train a cheat detector on move-quality features.

Two levels:
  1. Game-level:   HistGBT on per-game features, ROC-AUC (split by player!)
  2. Player-level: aggregate each player's games → one row; this is the real
     use case (you flag accounts, not single games).

BOT-titled accounts are excluded from training but scored separately as a
sanity check — bots are known engine players, so a working detector should
score them very high.

Usage:
    python src/train_cheat_detector.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

_BASE = Path(__file__).resolve().parent.parent
FEATURES_PATH = _BASE / "data" / "processed" / "cheat_features.parquet"

FEATURE_COLS = [
    "top1_rate", "top3_rate", "mean_rank", "median_rank",
    "mean_prob", "mean_swing", "worst_swing", "blunder_rate",
    "n_moves", "elo", "opp_elo",
]


def player_split(players: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Split by player so no player appears in both train and test."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(np.asarray(players, dtype=str))
    rng.shuffle(uniq)
    n_test = int(len(uniq) * test_frac)
    test_players = set(uniq[:n_test])
    mask = np.array([p in test_players for p in players])
    return ~mask, mask


def main():
    df = pd.read_parquet(FEATURES_PATH)
    print(f"{len(df):,} game rows, {df['player'].nunique():,} players")

    bots = df[df["title"] == "BOT"]
    df = df[df["title"] != "BOT"].copy()
    print(f"Excluded {len(bots):,} bot game rows ({bots['player'].nunique()} bots) "
          f"— held out as sanity check")
    print(f"Positive rate (games): {df['label_tos'].mean():.3f}")

    # ---------------- Game level ----------------
    X = df[FEATURE_COLS].values
    y = df["label_tos"].values.astype(int)
    train_m, test_m = player_split(df["player"].values)

    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf.fit(X[train_m], y[train_m])
    game_auc = roc_auc_score(y[test_m], clf.predict_proba(X[test_m])[:, 1])
    print(f"\nGame-level  ROC-AUC: {game_auc:.4f}  "
          f"(test: {test_m.sum():,} games, {y[test_m].sum():,} positive)")

    # ---------------- Player level ----------------
    agg = df.groupby("player").agg(
        label=("label_tos", "first"),
        n_games=("top1_rate", "size"),
        **{f"{c}_mean": (c, "mean") for c in FEATURE_COLS},
        **{f"{c}_std": (c, "std") for c in FEATURE_COLS},
        top1_rate_max=("top1_rate", "max"),
    ).fillna(0.0)

    feat_cols_p = [c for c in agg.columns if c not in ("label", )]
    Xp = agg[feat_cols_p].values
    yp = agg["label"].values.astype(int)
    tr_m, te_m = player_split(agg.index.values)

    clf_p = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf_p.fit(Xp[tr_m], yp[tr_m])
    proba = clf_p.predict_proba(Xp[te_m])[:, 1]
    player_auc = roc_auc_score(yp[te_m], proba)
    print(f"Player-level ROC-AUC: {player_auc:.4f}  "
          f"(test: {te_m.sum():,} players, {yp[te_m].sum():,} positive)")

    # Precision at high-confidence thresholds (what a mod team cares about)
    prec, rec, thr = precision_recall_curve(yp[te_m], proba)
    for target_rec in (0.10, 0.25, 0.50):
        idx = np.argmin(np.abs(rec - target_rec))
        print(f"  precision @ recall {rec[idx]:.2f}: {prec[idx]:.3f}")

    # ---------------- Bot sanity check ----------------
    if len(bots):
        bot_agg = bots.groupby("player").agg(
            n_games=("top1_rate", "size"),
            **{f"{c}_mean": (c, "mean") for c in FEATURE_COLS},
            **{f"{c}_std": (c, "std") for c in FEATURE_COLS},
            top1_rate_max=("top1_rate", "max"),
        ).fillna(0.0)
        bot_scores = clf_p.predict_proba(bot_agg[feat_cols_p].values)[:, 1]
        human_scores = clf_p.predict_proba(Xp[te_m])[:, 1]
        print(f"\nBot sanity check ({len(bot_agg)} bots):")
        print(f"  mean bot score   : {bot_scores.mean():.3f}")
        print(f"  mean human score : {human_scores[yp[te_m] == 0].mean():.3f}")
        print(f"  mean cheater score: {human_scores[yp[te_m] == 1].mean():.3f}")

    # ---------------- Persist player-level model ----------------
    import joblib
    model_dir = _BASE / "models"
    tmp_dir = model_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / "cheat_detector.joblib"
    joblib.dump({"model": clf_p, "feature_cols": feat_cols_p}, tmp)
    out = model_dir / "cheat_detector.joblib"
    tmp.rename(out)
    print(f"\nSaved player-level model to {out}")

    # ---------------- Feature importance (permutation on player level) ----
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf_p, Xp[te_m], yp[te_m],
                                 scoring="roc_auc", n_repeats=5, random_state=42)
    order = np.argsort(-imp.importances_mean)
    print("\nTop features (permutation importance, player level):")
    for i in order[:8]:
        print(f"  {feat_cols_p[i]:20s} {imp.importances_mean[i]:+.4f}")


if __name__ == "__main__":
    main()
