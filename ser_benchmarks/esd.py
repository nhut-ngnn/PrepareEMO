"""ESD manifest builder."""

import re
from pathlib import Path

import pandas as pd

from ser_benchmarks.common import save_csv
from ser_benchmarks.labels import ESD_EMOTION_MAP


def build_esd_manifest(
    root_dir: str,
    out_csv: str,
):
    """
    Expected ESD structure:
    Data/
      0011/
        Angry/
          Evaluation Set/*.wav
          Test Set/*.wav
          Training Set/*.wav
        Happy/
        Neutral/
        Sad/
        Surprise/
    """

    root = Path(root_dir)
    rows = []

    emotion_folders = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    split_folders = {
        "Training Set": "train",
        "Evaluation Set": "val",
        "Test Set": "test",
    }

    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        if not re.match(r"^\d{4}$", speaker_id):
            continue

        language = "zh" if int(speaker_id) <= 10 else "en"

        for emotion_name in emotion_folders:
            emotion_dir = speaker_dir / emotion_name
            if not emotion_dir.exists():
                continue

            for split_folder, split in split_folders.items():
                wav_dir = emotion_dir / split_folder
                if not wav_dir.exists():
                    continue

                for wav_path in sorted(wav_dir.glob("*.wav")):
                    rows.append({
                        "path": str(wav_path),
                        "utt_id": wav_path.stem,
                        "label": ESD_EMOTION_MAP[emotion_name],
                        "speaker_id": speaker_id,
                        "session_id": speaker_id,
                        "language": language,
                        "split": split,
                        "dataset": "ESD",
                    })

    df = pd.DataFrame(rows)
    save_csv(df, out_csv)
