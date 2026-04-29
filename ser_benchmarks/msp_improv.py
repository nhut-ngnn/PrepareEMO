"""MSP-IMPROV manifest builder."""

import re
from pathlib import Path

import pandas as pd

from ser_benchmarks.common import save_csv
from ser_benchmarks.labels import MSP_LABEL_MAP


def parse_msp_improv_from_filename(wav_path: Path):
    """
    Common filename style:
    MSP-IMPROV-S01A-M01-P-FM01.wav

    Practical parsing:
    - session_id: S01, S02, ...
    - speaker_id: S01A / S01B if available
    - intended emotion sometimes appears as a single code:
      A, H, S, N.
    """

    stem = wav_path.stem

    session_match = re.search(r"S(\d{2})", stem)
    session_id = f"S{session_match.group(1)}" if session_match else "unknown"

    speaker_match = re.search(r"S(\d{2})([AB])", stem)
    speaker_id = (
        f"S{speaker_match.group(1)}{speaker_match.group(2)}"
        if speaker_match
        else session_id
    )

    # Try to find one of the emotion codes between separators
    parts = re.split(r"[-_]", stem)
    label = None
    for token in parts:
        if token in MSP_LABEL_MAP:
            label = MSP_LABEL_MAP[token]
            break

    return session_id, speaker_id, label


def build_msp_improv_manifest(
    root_dir: str,
    out_csv: str,
    label_csv: str = "",
):
    """
    If label_csv is available, prefer it.
    Expected columns:
    file,label,speaker_id,session_id

    Otherwise, fallback to filename parsing.
    """

    root = Path(root_dir)
    rows = []

    if label_csv:
        label_df = pd.read_csv(label_csv)
        required = {"file", "label"}
        missing = required - set(label_df.columns)
        if missing:
            raise ValueError(f"label_csv missing columns: {missing}")

        for _, r in label_df.iterrows():
            file_name = str(r["file"])
            wav_path = root / file_name

            if not wav_path.exists():
                matches = list(root.rglob(file_name))
                if not matches:
                    continue
                wav_path = matches[0]

            session_id = str(r["session_id"]) if "session_id" in label_df.columns else parse_msp_improv_from_filename(wav_path)[0]
            speaker_id = str(r["speaker_id"]) if "speaker_id" in label_df.columns else parse_msp_improv_from_filename(wav_path)[1]

            rows.append({
                "path": str(wav_path),
                "utt_id": wav_path.stem,
                "label": str(r["label"]).lower(),
                "speaker_id": speaker_id,
                "session_id": session_id,
                "dataset": "MSP-IMPROV",
            })

    else:
        for wav_path in sorted(root.rglob("*.wav")):
            session_id, speaker_id, label = parse_msp_improv_from_filename(wav_path)

            if label is None:
                continue

            rows.append({
                "path": str(wav_path),
                "utt_id": wav_path.stem,
                "label": label,
                "speaker_id": speaker_id,
                "session_id": session_id,
                "dataset": "MSP-IMPROV",
            })

    df = pd.DataFrame(rows)
    save_csv(df, out_csv)
