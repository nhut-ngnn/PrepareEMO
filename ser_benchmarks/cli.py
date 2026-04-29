"""Command line interface for split-file SER benchmark preparation."""

import argparse
from pathlib import Path

import pandas as pd

from ser_benchmarks.common import save_csv
from ser_benchmarks.esd import build_esd_manifest
from ser_benchmarks.iemocap import build_iemocap_manifest
from ser_benchmarks.meld import build_meld_manifest
from ser_benchmarks.msp_improv import build_msp_improv_manifest
from ser_benchmarks.splits import (
    save_folds,
    session_independent_loso,
    speaker_dependent_split,
    speaker_independent_groupkfold,
    speaker_independent_holdout_split,
)


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Build manifest
    p = sub.add_parser("build_iemocap")
    p.add_argument("--root", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--emotion_set", default="4class", choices=["4class", "5class"])

    p = sub.add_parser("build_meld")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--dev_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--media_root", default="")

    p = sub.add_parser("build_esd")
    p.add_argument("--root", required=True)
    p.add_argument("--out_csv", required=True)

    p = sub.add_parser("build_msp_improv")
    p.add_argument("--root", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--label_csv", default="")

    # Split
    p = sub.add_parser("split")
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "sd",
            "si_holdout",
            "si_groupkfold",
            "loso_session",
            "official",
        ],
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="ser")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "build_iemocap":
        build_iemocap_manifest(args.root, args.out_csv, args.emotion_set)

    elif args.cmd == "build_meld":
        build_meld_manifest(
            args.train_csv,
            args.dev_csv,
            args.test_csv,
            args.out_csv,
            args.media_root,
        )

    elif args.cmd == "build_esd":
        build_esd_manifest(args.root, args.out_csv)

    elif args.cmd == "build_msp_improv":
        build_msp_improv_manifest(args.root, args.out_csv, args.label_csv)

    elif args.cmd == "split":
        df = pd.read_csv(args.manifest)

        if args.mode == "official":
            if "split" not in df.columns:
                raise ValueError("Official mode requires existing split column.")
            save_csv(df, Path(args.out_dir) / f"{args.prefix}_official.csv")

        elif args.mode == "sd":
            split_df = speaker_dependent_split(df, seed=args.seed)
            save_csv(split_df, Path(args.out_dir) / f"{args.prefix}_sd.csv")

        elif args.mode == "si_holdout":
            split_df = speaker_independent_holdout_split(df, seed=args.seed)
            save_csv(split_df, Path(args.out_dir) / f"{args.prefix}_si_holdout.csv")

        elif args.mode == "si_groupkfold":
            folds = speaker_independent_groupkfold(df, n_splits=args.n_splits)
            save_folds(folds, args.out_dir, f"{args.prefix}_si_groupkfold")

        elif args.mode == "loso_session":
            folds = session_independent_loso(df)
            save_folds(folds, args.out_dir, f"{args.prefix}_loso_session")


if __name__ == "__main__":
    main()
