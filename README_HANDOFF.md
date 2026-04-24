# Vietnamese ASR — Multi-Speaker Zipformer (handoff 2026-04-24)

## Environment

```bash
conda activate voice_to_text          # /home/vietnam/.conda/envs/voice_to_text
export PYTHONPATH=$PWD/icefall:$PWD/zipformer:${PYTHONPATH:-}
```

All paths relative to repo root `/home/vietnam/voice_to_text/`.

## Repo map

| Path | What |
|---|---|
| `zipformer_kd/finetune.py` | Training entry |
| `zipformer_kd/train_hybrid.sh` | Main launcher (env-var driven) |
| `zipformer_kd/multi_speaker_model.py` | `MultiSpeakerRnntModel` with `SpeakerKernel` |
| `zipformer/asr_datamodule.py` | Cut loading; `--augment-types clean,noise,...` |
| `zipformer/tokenizer.py`, `zipformer/tokens.txt` | Whisper-BPE vocab (auto-uppercases input) |
| `zipformer/beam_search.py` | RNNT beam (icefall-style) |
| `eval_dir_with_kernel.py` | Eval wav+txt dir with speaker-kernel path (CTC or RNNT-beam) |
| `eval_vlsp_with_kernel.py` | Eval lhotse cuts |
| `test_asr_quick.py` | Model builder + CTC/RNNT greedy decoders |
| `mms_msg/scripts/generate_multidataset_overlap.py` | Overlap generator (single/ov2/ov3) |
| `mms_msg/scripts/build_lhotse_cutset_augmented.py` | Lhotse cuts + MUSAN/RIR aug (full rebuild) |
| `mms_msg/scripts/augment_train_cuts.py` | **Apply noise/RIR to EXISTING train cuts** (used for v12) |
| `mms_msg/scripts/merge_yt1000h_into_train.py` | Merge new YT into existing train cuts |
| `mms_msg/scripts/convert_yt1000h_csv_to_manifest.py` | CSV → JSONL input manifest |
| `mms_msg/scripts/unify_yt1000h_to_accepted.py` | Consolidate split audio dirs |

## Data

- Train cuts: `data/lhotse_cutsets_new_train/train.{single,overlap2,overlap3}.cuts_{clean,noise}.jsonl.gz`
  - 312,771 single + 134,286 overlap2 + 32,051 overlap3 = ~1,674h total
- Val/test cuts: `data/lhotse_cutsets/{val,test}.*.cuts_*.jsonl.gz`
- Eval dirs: `data/vlsp2020/VLSP2020Test1` (7,500 wav+txt), `VLSP2020Test2` (18,401)
- MUSAN: `data/musan/{noise,music,speech}/` ; RIR: `data/RIRS_NOISES/simulated_rirs/` (pointsource_noises missing — blocker for RIR variant)

## Model history

| Ckpt | Config | Test1 (ctc_greedy / beam=8) | Test2 (ctc / beam=8) | Notes |
|---|---|---|---|---|
| v9 epoch-30 | 1x YT, clean only | 29.29 / — | 56.49 / — | Pre-merge baseline |
| v10 epoch-5 | +YT 1000h merged | 27.63 / — | 54.59 / — | Dataset 1674h |
| v11 best-valid | Resume v10 ep15, LR=5e-5 | 26.68 / **24.33** | 53.32 / **50.40** | Plateau with clean-only |
| v12 **epoch-12** | +noise aug (mix_prob=0.5, snr 5-20) | **25.64** / — | **50.78** / — | Noise-aug wins |

Winning hyperparams (since v9): `ENCODER_LR_SCALE=1.0`, `SPK_KERNEL_DROPOUT=0.0`.

## How to run

### Train (resume v13 template)
```bash
EXP_DIR=zipformer_kd/exp_hybrid_v13 \
PRETRAINED_CKPT=zipformer_kd/exp_hybrid_v12/epoch-12.pt \
NUM_EPOCHS=12 \
BASE_LR=0.00003 \
LR_EPOCHS=5 LR_BATCHES=5000 \
ENCODER_LR_SCALE=1.0 \
SPK_KERNEL_DROPOUT=0.0 \
WARMUP_BATCHES=200 \
AUGMENT_TYPES=noise \
bash zipformer_kd/train_hybrid.sh
```

### Regenerate noise variants (one-shot, lazy metadata)
```bash
python mms_msg/scripts/augment_train_cuts.py \
  --variants noise --mix-types single overlap2 overlap3 \
  --aug-prob 0.5 --snr-low 5 --snr-high 20 \
  --musan-parts noise
```

### Eval CTC greedy (fast, baseline decode)
```bash
python eval_dir_with_kernel.py \
  --checkpoint zipformer_kd/exp_hybrid_v12/epoch-12.pt \
  --dir data/vlsp2020/VLSP2020Test1 \
  --out-dir eval_out/v12_ep12_ctc_test1 \
  --decode-chunk-size 32 --left-context-frames 128 \
  --use-model \
  --decode-method ctc_greedy
```

### Eval RNNT beam (best accuracy)
```bash
# NOTE: rnnt_beam requires icefall on PYTHONPATH
export PYTHONPATH=/home/vietnam/voice_to_text/icefall:/home/vietnam/voice_to_text/zipformer

python eval_dir_with_kernel.py \
  --checkpoint zipformer_kd/exp_hybrid_v12/epoch-12.pt \
  --dir data/vlsp2020/VLSP2020Test1 \
  --out-dir eval_out/v12_ep12_beam8_test1 \
  --decode-chunk-size 32 --left-context-frames 128 \
  --use-model \
  --decode-method rnnt_beam --beam-size 8
```

`--use-model` = load raw checkpoint (skip model-averaging). `--beam-size` 4→8→16 shown to monotonically improve (v11 ctc 26.68 → beam8 24.33 on Test1).

## Key results (VLSP2020)

```
Test1 (7,500 utt, 108k ref words)            Test2 (18,401 utt, 62k ref words)
v9  ep30           ctc_greedy  29.29%       v9  ep30           ctc_greedy  56.49%
v10 ep5            ctc_greedy  27.63%       v10 ep5            ctc_greedy  54.59%
v11 best-valid     ctc_greedy  26.68%       v11 best-valid     ctc_greedy  53.32%
v11 best-valid     rnnt_beam=8 24.33%       v11 best-valid     rnnt_beam=8 50.40%
v12 ep12 (best)    ctc_greedy  25.64%       v12 ep12 (best)    ctc_greedy  50.78%
```

## Known issues / next levers

- RIR variant blocked: `data/RIRS_NOISES/pointsource_noises` missing. Either download it or patch `augment_train_cuts.py` to scan `simulated_rirs/*.wav` directly.
- Test2 at 50% WER remains high — likely domain mismatch (conversational/phone?). Data of similar domain would help more than more augmentation.
- CTC prefix beam + N-gram LM not yet tried (`zipformer/kenlm_scorer.py` present but unused).
- `train.{mix}.cuts_clean.bak_pre_yt1000h.jsonl.gz` = pre-YT-merge backups, safe to delete if stable.

## Backup on Google Drive

Remote: `gdrive:backupdatavoice/`  
- Data shards (tar.zst, 5G each): `*.tar.zst.part0000...` + `<name>.tar.zst.sha256` sidecars
- Manifest: `backup_manifest.txt` (tsv: timestamp, name, source_dir, bytes, sha256, parts, status)
- Models (after step below): `models/*.pt`

Restore one dataset:
```bash
rclone copy gdrive:backupdatavoice/ /restore/ --include "yt1900h_accepted.tar.zst.part*"
cat yt1900h_accepted.tar.zst.part* | zstd -d -T0 | tar -xf - -C /home/vietnam/voice_to_text
```
