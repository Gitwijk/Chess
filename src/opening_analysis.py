"""Win-rate analysis per opening from games.parquet (24.9M elite games).

Answers: does the chosen opening (White's system + Black's reply, as captured
by the ECO code / opening name) actually predict the result — or does it only
look that way because stronger players prefer certain openings?

Method:
  - Raw rates per opening (all games).
  - Elo-balanced rates: only games with |Elo difference| <= --elo-window,
    which removes the "strong players pick opening X" confound.
  - Predictive-power check: logistic regression on Elo difference alone vs.
    Elo difference + opening, compared by log-loss and accuracy on a holdout.

Outputs data/processed/opening_stats.parquet and prints a summary.

Usage:
    python src/opening_analysis.py
    python src/opening_analysis.py --min-games 5000 --elo-window 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
GAMES_PATH = _BASE / "data" / "processed" / "games.parquet"
OUT_PATH = _BASE / "data" / "processed" / "opening_stats.parquet"

ELO_BANDS = [(2200, 2400), (2400, 2600), (2600, 3500)]


def score_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """W/D/L rates and White's score per opening key."""
    g = df.groupby(key, observed=True)["Result"]
    out = pd.DataFrame({
        "games": g.size(),
        "white_win": g.apply(lambda s: (s == "1-0").mean()),
        "draw": g.apply(lambda s: (s == "1/2-1/2").mean()),
        "black_win": g.apply(lambda s: (s == "0-1").mean()),
    })
    # White's score: 1 per win, 0.5 per draw.
    out["white_score"] = out["white_win"] + 0.5 * out["draw"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, default=2000,
                    help="Minimum games for an opening to be reported")
    ap.add_argument("--elo-window", type=int, default=50,
                    help="Max |WhiteElo - BlackElo| for the balanced sample")
    ap.add_argument("--sample", type=int, default=2_000_000,
                    help="Games sampled for the predictive-power model")
    ap.add_argument("--min-moves", type=int, default=10,
                    help="Drop games shorter than this (abandoned/forfeited games "
                         "otherwise poison the rates: e.g. 'King's Pawn' has a "
                         "median length of 1 move and an 0.86 White score)")
    args = ap.parse_args()

    print(f"Reading {GAMES_PATH.name}...")
    df = pd.read_parquet(GAMES_PATH, columns=[
        "Result", "WhiteElo", "BlackElo", "ECO", "Opening", "NumMoves"])
    df = df[df["Result"].isin(["1-0", "0-1", "1/2-1/2"])]
    df = df[(df["WhiteElo"] > 0) & (df["BlackElo"] > 0)]
    n_before = len(df)
    df = df[df["NumMoves"] >= args.min_moves]
    print(f"{len(df):,} usable games "
          f"({n_before - len(df):,} dropped as shorter than {args.min_moves} moves)")

    overall = df["Result"].value_counts(normalize=True)
    white_score_all = overall.get("1-0", 0) + 0.5 * overall.get("1/2-1/2", 0)
    print(f"\nBaseline: White scores {white_score_all:.4f} overall "
          f"(W {overall.get('1-0', 0):.3f} / D {overall.get('1/2-1/2', 0):.3f} "
          f"/ L {overall.get('0-1', 0):.3f})")

    # ---------------- Elo-balanced sample ----------------
    df["elo_diff"] = df["WhiteElo"] - df["BlackElo"]
    bal = df[df["elo_diff"].abs() <= args.elo_window]
    print(f"Elo-balanced sample (|diff| <= {args.elo_window}): {len(bal):,} games")

    raw = score_table(df, "Opening")
    bal_t = score_table(bal, "Opening")

    stats = raw.join(bal_t, rsuffix="_bal", how="left")
    stats = stats[stats["games"] >= args.min_games].copy()
    stats["white_edge_bal"] = stats["white_score_bal"] - 0.5
    stats = stats.sort_values("games", ascending=False)

    # Per-Elo-band White score, for openings that clear the threshold.
    for lo, hi in ELO_BANDS:
        band = df[(df["WhiteElo"].between(lo, hi)) & (df["BlackElo"].between(lo, hi))]
        bt = score_table(band, "Opening")["white_score"]
        stats[f"white_score_{lo}_{hi}"] = bt.reindex(stats.index)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.parent / (OUT_PATH.name + ".tmp")
    stats.to_parquet(tmp)
    tmp.rename(OUT_PATH)
    print(f"Saved {len(stats):,} openings -> {OUT_PATH}")

    # ---------------- Reports ----------------
    top = stats.head(15)
    print(f"\n=== 15 most played openings (>= {args.min_games:,} games) ===")
    print(f"{'Opening':<52}{'games':>10}{'W score':>9}{'balanced':>10}")
    for name, r in top.iterrows():
        print(f"{str(name)[:50]:<52}{int(r['games']):>10,}"
              f"{r['white_score']:>9.3f}{r['white_score_bal']:>10.3f}")

    ranked = stats.dropna(subset=["white_score_bal"]).sort_values("white_score_bal")
    print(f"\n=== Best for Black (Elo-balanced, >= {args.min_games:,} games) ===")
    for name, r in ranked.head(10).iterrows():
        print(f"{str(name)[:50]:<52}{int(r['games']):>10,}{r['white_score_bal']:>10.3f}")
    print(f"\n=== Best for White (Elo-balanced) ===")
    for name, r in ranked.tail(10)[::-1].iterrows():
        print(f"{str(name)[:50]:<52}{int(r['games']):>10,}{r['white_score_bal']:>10.3f}")

    # ---------------- Does the opening add predictive power? ----------------
    print("\n=== Predictive power: Elo alone vs Elo + opening ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    m = df.sample(min(args.sample, len(df)), random_state=42)
    y = (m["Result"] == "1-0").astype(int).values          # White wins vs not
    elo = m[["elo_diff"]].values.astype(np.float32)
    eco = m[["ECO"]].astype(str).values

    idx_tr, idx_te = train_test_split(np.arange(len(m)), test_size=0.2,
                                      random_state=42, stratify=y)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    eco_tr = enc.fit_transform(eco[idx_tr])
    eco_te = enc.transform(eco[idx_te])

    def fit_report(Xtr, Xte, label):
        clf = LogisticRegression(max_iter=1000).fit(Xtr, y[idx_tr])
        p = clf.predict_proba(Xte)[:, 1]
        ll = log_loss(y[idx_te], p)
        acc = accuracy_score(y[idx_te], p > 0.5)
        print(f"  {label:<22} log-loss {ll:.5f}   acc {acc:.4f}")
        return ll

    from scipy.sparse import hstack, csr_matrix
    ll_elo = fit_report(elo[idx_tr], elo[idx_te], "Elo diff only")
    ll_both = fit_report(hstack([csr_matrix(elo[idx_tr]), eco_tr]).tocsr(),
                         hstack([csr_matrix(elo[idx_te]), eco_te]).tocsr(),
                         "Elo diff + ECO")
    print(f"  -> opening adds {ll_elo - ll_both:+.5f} log-loss improvement "
          f"({(ll_elo - ll_both) / ll_elo * 100:+.2f}%)")


if __name__ == "__main__":
    main()
