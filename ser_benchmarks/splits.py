"""Benchmark split functions."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    GroupKFold,
    LeaveOneGroupOut,
)

from ser_benchmarks.common import (
    add_split_column,
    check_no_speaker_leakage,
)


def speaker_dependent_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    """
    Speaker-dependent:
    - random utterance-level split
    - speaker can appear in train/val/test
    """

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_ratio = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    return add_split_column(train_df, val_df, test_df)


def speaker_independent_holdout_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    """
    Speaker-independent holdout:
    - split by speaker_id
    - no speaker overlap
    """

    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_val_idx, test_idx = next(
        gss_test.split(df, groups=df["speaker_id"])
    )

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_ratio = val_size / (1.0 - test_size)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )

    train_idx, val_idx = next(
        gss_val.split(train_val_df, groups=train_val_df["speaker_id"])
    )

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    check_no_speaker_leakage(train_df, val_df, test_df)

    return add_split_column(train_df, val_df, test_df)


def session_independent_loso(
    df: pd.DataFrame,
):
    """
    Leave-One-Session-Out:
    - each fold uses one session as test
    - suitable for IEMOCAP and MSP-IMPROV
    """

    if "session_id" not in df.columns:
        raise ValueError("Need column session_id.")

    folds = []

    for fold_id, test_session in enumerate(sorted(df["session_id"].unique()), start=1):
        test_df = df[df["session_id"] == test_session].copy()
        train_val_df = df[df["session_id"] != test_session].copy()

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=fold_id,
        )

        train_idx, val_idx = next(
            gss_val.split(train_val_df, groups=train_val_df["speaker_id"])
        )

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        check_no_speaker_leakage(train_df, val_df, test_df)

        fold_df = add_split_column(train_df, val_df, test_df)
        fold_df["fold"] = fold_id
        fold_df["test_session"] = test_session

        folds.append((test_session, fold_df))

    return folds


def speaker_independent_groupkfold(
    df: pd.DataFrame,
    n_splits: int = 5,
):
    """
    GroupKFold by speaker_id.
    """

    gkf = GroupKFold(n_splits=n_splits)
    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(
        gkf.split(df, y=df["label"], groups=df["speaker_id"]),
        start=1,
    ):
        train_val_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=fold_id,
        )

        tr_idx, val_idx = next(
            gss_val.split(train_val_df, groups=train_val_df["speaker_id"])
        )

        train_df = train_val_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        check_no_speaker_leakage(train_df, val_df, test_df)

        fold_df = add_split_column(train_df, val_df, test_df)
        fold_df["fold"] = fold_id

        folds.append((fold_id, fold_df))

    return folds


def save_folds(folds, out_dir: str, prefix: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold_key, fold_df in folds:
        out_csv = out_dir / f"{prefix}_fold_{fold_key}.csv"
        fold_df.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")
