"""Baseline model: predict game outcome (White win / Black win / Draw) from pre-game features."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

GAMES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "games.parquet"

RESULT_MAP = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(GAMES_PATH, columns=["WhiteElo", "BlackElo", "ECO", "Result", "TimeControl"])
    df = df[df["Result"].isin(RESULT_MAP)].dropna(subset=["WhiteElo", "BlackElo", "ECO"])

    df["outcome"] = df["Result"].map(RESULT_MAP)
    df["elo_diff"] = df["WhiteElo"] - df["BlackElo"]
    df["white_elo"] = df["WhiteElo"]

    return df[["elo_diff", "white_elo", "ECO", "outcome"]]


def main():
    print("Loading data...")
    df = load_features()
    print(f"{len(df):,} games after filtering")
    print(df["outcome"].value_counts(normalize=True))

    eco_encoder = LabelEncoder()
    df["eco_code"] = eco_encoder.fit_transform(df["ECO"])

    X = df[["elo_diff", "white_elo", "eco_code"]]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Baseline: always predict majority class
    majority_acc = (y_test == y_test.mode()[0]).mean()
    print(f"Majority-class baseline accuracy: {majority_acc:.4f}")


if __name__ == "__main__":
    main()
