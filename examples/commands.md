# Example Commands

## IEMOCAP

```bash
python prepare_ser_benchmarks.py build_iemocap \
  --root /path/to/IEMOCAP_full_release \
  --out_csv manifests/iemocap_4cls.csv \
  --emotion_set 4class

python prepare_ser_benchmarks.py split \
  --manifest manifests/iemocap_4cls.csv \
  --mode loso_session \
  --out_dir splits/iemocap \
  --prefix iemocap_4cls
```

Speaker-dependent baseline:

```bash
python prepare_ser_benchmarks.py split \
  --manifest manifests/iemocap_4cls.csv \
  --mode sd \
  --out_dir splits/iemocap \
  --prefix iemocap_4cls
```

## MELD

```bash
python prepare_ser_benchmarks.py build_meld \
  --train_csv /path/to/MELD/train_sent_emo.csv \
  --dev_csv /path/to/MELD/dev_sent_emo.csv \
  --test_csv /path/to/MELD/test_sent_emo.csv \
  --media_root /path/to/MELD/audio_or_video_clips \
  --out_csv manifests/meld.csv

python prepare_ser_benchmarks.py split \
  --manifest manifests/meld.csv \
  --mode official \
  --out_dir splits/meld \
  --prefix meld
```

## ESD

```bash
python prepare_ser_benchmarks.py build_esd \
  --root /path/to/ESD/Data \
  --out_csv manifests/esd.csv

python prepare_ser_benchmarks.py split \
  --manifest manifests/esd.csv \
  --mode official \
  --out_dir splits/esd \
  --prefix esd
```

## MSP-IMPROV

```bash
python prepare_ser_benchmarks.py build_msp_improv \
  --root /path/to/MSP-IMPROV \
  --label_csv /path/to/msp_improv_labels.csv \
  --out_csv manifests/msp_improv.csv

python prepare_ser_benchmarks.py split \
  --manifest manifests/msp_improv.csv \
  --mode loso_session \
  --out_dir splits/msp_improv \
  --prefix msp_improv
```
