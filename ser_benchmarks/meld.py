"""MELD manifest builder."""

from pathlib import Path

import pandas as pd

from ser_benchmarks.common import save_csv
from ser_benchmarks.labels import MELD_EMOTION_MAP


def find_meld_media_path(media_root: Path, split: str, dialogue_id, utterance_id):
    """
    Supports common MELD clip names:
    dia{Dialogue_ID}_utt{Utterance_ID}.mp4
    dia{Dialogue_ID}_utt{Utterance_ID}.wav
    """

    candidates = [
        media_root / split / f"dia{dialogue_id}_utt{utterance_id}.wav",
        media_root / split / f"dia{dialogue_id}_utt{utterance_id}.mp4",
        media_root / f"{split}_splits" / f"dia{dialogue_id}_utt{utterance_id}.mp4",
        media_root / f"{split}_splits" / f"dia{dialogue_id}_utt{utterance_id}.wav",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    return ""


def build_meld_manifest(
    train_csv: str,
    dev_csv: str,
    test_csv: str,
    out_csv: str,
    media_root: str = "",
):
    """
    MELD should normally keep official train/dev/test split.
    CSV columns usually include:
    Sr No., Utterance, Speaker, Emotion, Sentiment, Dialogue_ID, Utterance_ID, ...
    """

    split_files = {
        "train": train_csv,
        "val": dev_csv,
        "test": test_csv,
    }

    media_root = Path(media_root) if media_root else None
    rows = []

    for split, csv_path in split_files.items():
        df = pd.read_csv(csv_path)

        for _, r in df.iterrows():
            raw_emotion = str(r["Emotion"]).strip().lower()
            label = MELD_EMOTION_MAP.get(raw_emotion, raw_emotion)

            dialogue_id = r["Dialogue_ID"]
            utterance_id = r["Utterance_ID"]
            speaker = str(r["Speaker"]).strip()

            path = ""
            if media_root is not None:
                path = find_meld_media_path(media_root, split, dialogue_id, utterance_id)

            rows.append({
                "path": path,
                "utt_id": f"dia{dialogue_id}_utt{utterance_id}",
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "text": str(r["Utterance"]),
                "label": label,
                "speaker_id": speaker,
                "session_id": f"dialogue_{dialogue_id}",
                "split": split,
                "dataset": "MELD",
            })

    out_df = pd.DataFrame(rows)
    save_csv(out_df, out_csv)
