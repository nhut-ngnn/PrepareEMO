"""IEMOCAP manifest builder."""

import re
from pathlib import Path

import pandas as pd

from ser_benchmarks.common import save_csv
from ser_benchmarks.labels import IEMOCAP_4CLS, IEMOCAP_5CLS


def parse_iemocap_speaker_id(utt_id: str) -> str:
    """
    Example:
    Ses01F_impro01_F000 -> Ses01F
    Ses01F_impro01_M000 -> Ses01M
    """
    m = re.match(r"^(Ses\d{2})[FM]_.+_([FM])\d+$", utt_id)
    if not m:
        return "unknown"
    return f"{m.group(1)}{m.group(2)}"


def parse_iemocap_session_id(utt_id: str) -> str:
    m = re.match(r"^(Ses\d{2})", utt_id)
    return m.group(1) if m else "unknown"


def build_iemocap_manifest(
    root_dir: str,
    out_csv: str,
    emotion_set: str = "4class",
):
    """
    Expected IEMOCAP structure:
    IEMOCAP_full_release/
      Session1/
        dialog/EmoEvaluation/*.txt
        sentences/wav/<dialog_id>/<utt_id>.wav
    """

    root = Path(root_dir)

    if emotion_set == "4class":
        label_map = IEMOCAP_4CLS
    elif emotion_set == "5class":
        label_map = IEMOCAP_5CLS
    else:
        raise ValueError("emotion_set must be either '4class' or '5class'.")

    rows = []

    eval_files = sorted(root.glob("Session*/dialog/EmoEvaluation/*.txt"))

    for eval_file in eval_files:
        with open(eval_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Example line:
                # [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5, 2.5, 2.5]
                if not line.startswith("["):
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue

                utt_id = parts[1]
                raw_label = parts[2]

                if raw_label not in label_map:
                    continue

                label = label_map[raw_label]
                session_id = parse_iemocap_session_id(utt_id)
                speaker_id = parse_iemocap_speaker_id(utt_id)

                dialog_id = "_".join(utt_id.split("_")[:-1])
                wav_path = (
                    root
                    / f"Session{int(session_id[-2:])}"
                    / "sentences"
                    / "wav"
                    / dialog_id
                    / f"{utt_id}.wav"
                )

                if not wav_path.exists():
                    continue

                rows.append({
                    "path": str(wav_path),
                    "utt_id": utt_id,
                    "label": label,
                    "speaker_id": speaker_id,
                    "session_id": session_id,
                    "dataset": "IEMOCAP",
                })

    df = pd.DataFrame(rows)
    save_csv(df, out_csv)
