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

# v2 groups, present only when extracted from a corpus with clock data.
TIMING_COLS = ["mean_time", "std_time", "time_entropy_corr",
               "fast_top1_rate", "time_cv", "mean_entropy"]
# Excess over what players of the same rating normally achieve.
EXCESS_COLS = ["excess_top1", "excess_top3", "excess_rank"]

BAND = 200   # Elo band width for the normalisation


EXCESS_SRC = (("top1_rate", "excess_top1"),
              ("top3_rate", "excess_top3"),
              ("mean_rank", "excess_rank"))


def fit_elo_baseline(df: pd.DataFrame, fit_mask: np.ndarray) -> dict:
    """Per-Elo-band mean of each source feature, from CLEAN TRAINING rows only.

    Using all rows would leak test labels into a feature, and including
    cheaters would let them inflate the baseline they are measured against.
    """
    basis = df[fit_mask & ~df["label_tos"].astype(bool).values]
    band = (basis["elo"] // BAND * BAND).astype(int)
    return {src: (basis.groupby(band)[src].mean(), basis[src].mean())
            for src, _ in EXCESS_SRC}


def apply_elo_excess(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Add 'how far above your rating do you play' features.

    Engine-match rate rises naturally with strength, so an absolute top1_rate
    is not comparable between a 1200 and a 2400 player — v1 could only ever
    work inside one narrow rating range. Each game is scored against the mean
    for its Elo band, using a baseline fitted elsewhere so that every group
    (including the held-out bots) is measured on the same yardstick.
    """
    df = df.copy()
    band = (df["elo"] // BAND * BAND).astype(int)
    for src, dst in EXCESS_SRC:
        per_band, global_mean = baseline[src]
        df[dst] = df[src] - band.map(per_band)
        # Bands with no clean training player fall back to the global mean.
        df[dst] = df[dst].fillna(df[src] - global_mean)
    return df


def player_split(players: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Split by player so no player appears in both train and test."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(np.asarray(players, dtype=str))
    rng.shuffle(uniq)
    n_test = int(len(uniq) * test_frac)
    test_players = set(uniq[:n_test])
    mask = np.array([p in test_players for p in players])
    return ~mask, mask


def aggregate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per player: mean and std of each feature, plus game count."""
    return df.groupby("player").agg(
        label=("label_tos", "first"),
        elo=("elo", "mean"),
        n_games=("top1_rate", "size"),
        **{f"{c}_mean": (c, "mean") for c in cols},
        **{f"{c}_std": (c, "std") for c in cols},
        top1_rate_max=("top1_rate", "max"),
    ).fillna(0.0)


def player_auc_for(df: pd.DataFrame, cols: list[str], seed: int = 42):
    """Fit at player level on `cols`; return (auc, probabilities, agg, masks)."""
    agg = aggregate(df, cols)
    feat = [c for c in agg.columns if c not in ("label",)]
    X, y = agg[feat].values, agg["label"].values.astype(int)
    tr, te = player_split(agg.index.values, seed=seed)
    if y[tr].sum() == 0 or y[te].sum() == 0:
        return float("nan"), None, agg, (tr, te), feat
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42).fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y[te], p), p, agg, (tr, te), feat


def main():
    df = pd.read_parquet(FEATURES_PATH)
    print(f"{len(df):,} game rows, {df['player'].nunique():,} players")

    bots = df[df["title"] == "BOT"]
    df = df[df["title"] != "BOT"].copy()
    print(f"Excluded {len(bots):,} bot game rows ({bots['player'].nunique()} bots) "
          f"— held out as sanity check")
    print(f"Positive rate (games): {df['label_tos'].mean():.3f}")

    # Which v2 groups does this feature file actually carry?
    has_timing = all(c in df.columns for c in TIMING_COLS) and \
        df["mean_time"].notna().any()
    print(f"Timing features present: {has_timing}"
          + ("" if has_timing else "  (clockless corpus — v1 feature set only)"))

    # ---------------- Game level ----------------
    X = df[FEATURE_COLS].values
    y = df["label_tos"].values.astype(int)
    train_m, test_m = player_split(df["player"].values)

    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf.fit(X[train_m], y[train_m])
    game_auc = roc_auc_score(y[test_m], clf.predict_proba(X[test_m])[:, 1])
    print(f"\nGame-level  ROC-AUC: {game_auc:.4f}  "
          f"(test: {test_m.sum():,} games, {y[test_m].sum():,} positive)")

    # Elo-excess features are fitted on training players only (no label leak).
    train_rows, _ = player_split(df["player"].values)
    baseline = fit_elo_baseline(df, train_rows)
    df = apply_elo_excess(df, baseline)

    # ---------------- Ablation: which group earns its place? ----------------
    groups = {"v1 baseline": FEATURE_COLS}
    if has_timing:
        groups["+ timing"] = FEATURE_COLS + TIMING_COLS
    groups["+ Elo-excess"] = FEATURE_COLS + EXCESS_COLS
    if has_timing:
        groups["+ both (v2)"] = FEATURE_COLS + TIMING_COLS + EXCESS_COLS

    print("\n=== Ablation (player-level ROC-AUC) ===")
    results = {}
    for name, cols in groups.items():
        auc, *_ = player_auc_for(df, cols)
        results[name] = auc
        print(f"  {name:<16} {auc:.4f}")
    best_name = max(results, key=lambda k: results[k])
    print(f"  best: {best_name}")

    # ---------------- Full model ----------------
    feature_cols = groups[best_name]
    player_auc, proba, agg, (tr_m, te_m), feat_cols_p = player_auc_for(df, feature_cols)
    Xp, yp = agg[feat_cols_p].values, agg["label"].values.astype(int)
    clf_p = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf_p.fit(Xp[tr_m], yp[tr_m])
    print(f"\nPlayer-level ROC-AUC: {player_auc:.4f}  "
          f"(test: {te_m.sum():,} players, {yp[te_m].sum():,} positive)")

    # ---------------- Per-Elo-band performance ----------------
    # "Works outside the elite" is the whole point of v2, so report it.
    te_agg = agg[te_m].copy()
    te_agg["score"] = proba
    te_agg["band"] = (te_agg["elo"] // 400 * 400).astype(int)
    print("\nPer Elo band (test players):")
    for band, g in te_agg.groupby("band"):
        if g["label"].sum() >= 3 and (~g["label"].astype(bool)).sum() >= 3:
            auc_b = roc_auc_score(g["label"].astype(int), g["score"])
            print(f"  {band}-{band + 399}: AUC {auc_b:.3f}  "
                  f"({int(g['label'].sum())} banned / {len(g)} players)")
        else:
            print(f"  {band}-{band + 399}: too few labels "
                  f"({int(g['label'].sum())} banned / {len(g)} players)")

    # Precision at high-confidence thresholds (what a mod team cares about)
    prec, rec, thr = precision_recall_curve(yp[te_m], proba)
    for target_rec in (0.10, 0.25, 0.50):
        idx = np.argmin(np.abs(rec - target_rec))
        print(f"  precision @ recall {rec[idx]:.2f}: {prec[idx]:.3f}")

    # ---------------- Bot sanity check ----------------
    if len(bots):
        # Score the bots through the same pipeline, measured against the SAME
        # baseline as the humans (fitted on clean training humans) — giving
        # bots their own baseline would cancel exactly what we want to see.
        bot_agg = aggregate(apply_elo_excess(bots, baseline), feature_cols)
        bot_agg = bot_agg.reindex(columns=agg.columns, fill_value=0.0).fillna(0.0)
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
