"""Backward-compatible entrypoint and re-export module.

The original implementation was a single file. The functions are now split
across ``ser_benchmarks/`` modules, while this file keeps the old script name.
"""

from ser_benchmarks.common import (
    add_split_column,
    check_no_speaker_leakage,
    save_csv,
)
from ser_benchmarks.esd import build_esd_manifest
from ser_benchmarks.iemocap import (
    build_iemocap_manifest,
    parse_iemocap_session_id,
    parse_iemocap_speaker_id,
)
from ser_benchmarks.labels import (
    ESD_EMOTION_MAP,
    IEMOCAP_4CLS,
    IEMOCAP_5CLS,
    MELD_EMOTION_MAP,
    MSP_LABEL_MAP,
)
from ser_benchmarks.meld import build_meld_manifest, find_meld_media_path
from ser_benchmarks.msp_improv import (
    build_msp_improv_manifest,
    parse_msp_improv_from_filename,
)
from ser_benchmarks.splits import (
    save_folds,
    session_independent_loso,
    speaker_dependent_split,
    speaker_independent_groupkfold,
    speaker_independent_holdout_split,
)
from ser_benchmarks.cli import main


if __name__ == "__main__":
    main()
