"""Common helpers shared across builders and split protocols."""

from pathlib import Path

import pandas as pd


def save_csv(df: pd.DataFrame, out_csv: str):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")
    print(df.head())
    if "split" in df.columns:
        print(df["split"].value_counts())
    print("Samples:", len(df))
    print("Speakers:", df["speaker_id"].nunique())


def check_no_speaker_leakage(train_df, val_df=None, test_df=None):
    train_spk = set(train_df["speaker_id"])

    if val_df is not None:
        overlap = train_spk & set(val_df["speaker_id"])
        if overlap:
            raise AssertionError(f"Speaker leakage train-val: {overlap}")

    if test_df is not None:
        overlap = train_spk & set(test_df["speaker_id"])
        if overlap:
            raise AssertionError(f"Speaker leakage train-test: {overlap}")

    if val_df is not None and test_df is not None:
        overlap = set(val_df["speaker_id"]) & set(test_df["speaker_id"])
        if overlap:
            raise AssertionError(f"Speaker leakage val-test: {overlap}")


def add_split_column(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)
