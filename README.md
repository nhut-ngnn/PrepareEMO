# PrepareEMO

PrepareEMO is a split-file version of the original `prepare_ser_benchmarks.py`
script for preparing Speech Emotion Recognition (SER) benchmark manifests and
splits.

The function logic is preserved from the original script. The code is only
reorganized into separate files so each dataset and split protocol can be
maintained independently.

## File Layout

```text
PrepareEMO/
  prepare_ser_benchmarks.py      # backward-compatible CLI entrypoint
  ser_benchmarks/
    labels.py                    # emotion label maps
    common.py                    # save_csv, leakage checks, split column helper
    iemocap.py                   # IEMOCAP manifest builder
    meld.py                      # MELD manifest builder
    esd.py                       # ESD manifest builder
    msp_improv.py                # MSP-IMPROV manifest builder
    splits.py                    # benchmark split functions
    cli.py                       # argparse CLI
```

## Academic Benchmark Recommendation

| Dataset | Recommended benchmark | Note |
| --- | --- | --- |
| IEMOCAP | `loso_session` | Main speaker-independent benchmark: leave-one-session-out across five sessions. |
| IEMOCAP | `sd` | Speaker-dependent baseline: random utterance-level train/validation/test split. |
| MELD | `official` | Preserve official train/dev/test partitions. |
| ESD | `official` or `si_groupkfold` | Use official folders for the main split; use speaker-independent GroupKFold as an additional SER protocol. |
| MSP-IMPROV | `loso_session` or `si_groupkfold` | Session- or speaker-disjoint evaluation is appropriate for dyadic sessions. |

## Installation

```bash
python -m pip install -r requirements.txt
```

For editable CLI installation:

```bash
python -m pip install -e ".[dev]"
```

## Usage

The original command style remains valid:

```bash
python prepare_ser_benchmarks.py --help
```

If installed as a package, the console command is also available:

```bash
prepare-emo --help
```

See [examples/commands.md](examples/commands.md) for dataset-specific commands.
# PrepareEMO
