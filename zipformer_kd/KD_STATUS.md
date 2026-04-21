# KD Training — Status & Audit

## Cấu hình hiện tại (train_kd.sh)

| Param | Giá trị | Ghi chú |
|-------|---------|---------|
| `KD_OBJECTIVE` | `rnnt` | LM-only KD + joiner diagonal KD |
| `KD_LOSS_SCALE` | `0.05` | Scale LM-KD; skip_am_branch=True → chỉ LM branch |
| `JOINER_KD_SCALE` | `0.3` | KD trực tiếp trên joiner diagonal path |
| `KD_TEMPERATURE` | `4.0` | Softmax temperature cho KL |
| `DECODER_CE_SCALE` | `0.0` | Tắt, chỉ monitor |
| `JOINER_CE_SCALE` | `0.0` | Tắt, chỉ monitor |
| `--bpe-model` | `zipformer_kd/teacher_tokens.txt` | Dùng teacher vocab (100% overlap với teacher) |
| `NUM_EPOCHS` | `1` | Tăng lên ≥5 cho full training |
| `MAX_DURATION` | `600` | frames/batch |
| `CTC_ONLY_STEPS` | `2000` | simple/pruned RNNT tắt; CTC + KD vẫn backprop |

---

## Các thay đổi đã thực hiện

### 1. Teacher AM branch bị degenerate → skip
- **Vấn đề**: `simple_am_proj.bias[<unk>] = 80` → luôn predict `<unk>` với confidence ~100%
- **Fix**: `skip_am_branch=True` trong cả 2 lần gọi `rnnt_kd_loss()` ở `finetune.py`
- **File**: `zipformer_kd/kd_loss.py` (thêm tham số `skip_am_branch`), `zipformer_kd/finetune.py`

### 2. Vocab mismatch: 72% overlap → đổi sang teacher vocab
- **Vấn đề**: Student vocab (cũ) và teacher vocab chỉ trùng 1440/2000 tokens (72%)
  - 560 student tokens không có trong teacher → bị map vào teacher `<unk>` logit
  - 28% KD signal bị sai — tất cả vị trí unmapped nhận cùng 1 giá trị (teacher `<unk>` logit)
- **Fix**: Dùng `zipformer_kd/teacher_tokens.txt` (copy từ `config.json` của teacher JIT)
- **File**: `zipformer_kd/train_kd.sh` dòng `--bpe-model`
- **Hệ quả**: Decoder/joiner/ctc head load từ `student_teacher_init.pt` (teacher vocab weights). Encoder load riêng (strip prefix `asr_model.encoder.`), heads load qua full-model state dict (`asr_model.*` keys khớp trực tiếp)

### 3. Bug critical: encoder không load được Phase 2 checkpoint
- **Vấn đề**: `finetune.py:get_encoder_model()` gọi `encoder.load_state_dict(pretrained["model"], strict=False)`
  - Checkpoint Phase 2 lưu encoder keys với prefix `asr_model.encoder.*` (827 keys)
  - `HubertModel` state_dict không có prefix này → intersection = 0 keys
  - Kết quả: **encoder train từ HubertModel init, không dùng Phase 2 weights**
- **Fix**: Strip prefix `asr_model.encoder.` trước khi load (`finetune.py:get_encoder_model`)
  - Sau fix: 827/827 encoder keys matched
- **File**: `zipformer_kd/finetune.py:1216`

### 4. Bug latent: CTC logits method sai
- **Vấn đề**: `multi_speaker_model.py:661` gọi `get_rnnt_kd_logits()` thay vì `get_ctc_logits()`
  - `get_rnnt_kd_logits` = AM + LM (RNNT combined at null decoder state)
  - `get_ctc_logits` = encoder → dropout → linear (CTC only)
- **Fix**: Đổi sang `get_ctc_logits(H_loss)`
- **Tác động**: Latent — không active với `KD_OBJECTIVE=rnnt` (`use_kd_ctc=False`)
- **File**: `zipformer_kd/multi_speaker_model.py:661`

---

## Kết quả audit

### Vocab consistency ✅
| Check | Kết quả |
|-------|---------|
| `vocab_size` đọc dynamic từ BPE file (`finetune.py:2396`) | OK |
| Model creation dùng `params.vocab_size` (không hardcode) | OK |
| `teacher.py._build_vocab_mapping()` với same vocab → identity mapping | OK |
| `teacher.py.map_to_student_vocab()` | OK |

### Linear layer shapes ✅
| Check | Kết quả |
|-------|---------|
| FitNet projector: student 512→256 (teacher dim) | OK (scale=0.0, disabled) |
| MGD projector: 256→256 | OK (scale=0.0, disabled) |
| Joiner diagonal: student encoder [B,T,512] + decoder [B,T,512] → [B,T,V] | OK |
| `simple_am_proj`, `simple_lm_proj`, `ctc_output`, `joiner.output_linear` | (V, encoder_dim) — với config hiện tại encoder_dim=512 |

### Backpropagation flow ✅
| Check | Kết quả |
|-------|---------|
| `skip_am_branch=True` dừng gradient AM branch | OK |
| `kd_rnnt_nonblank` và `kd_joiner_nonblank_raw` trong `no_grad()` | OK (log-only) |
| `ctc_only_steps=2000`: simple/pruned RNNT không backprop | OK |
| KD losses (rnnt_kd, joiner_kd) trong CTC-only phase | **vẫn backprop** — CTC-only chỉ tắt simple/pruned loss, không tắt KD |
| `loss.detach()` cho metrics | OK |

### Loss computation ✅
| Check | Kết quả |
|-------|---------|
| `rnnt_kd_loss(skip_am_branch=True)` → chỉ LM-KL | OK |
| `joiner_diag_kd_loss()` → KL trên diagonal path | OK |
| `kd_rnnt_nonblank` diagnostic call (no_grad, không affect loss) | OK |
| Scale: `kd_loss_scale=0.05` cho rnnt_kd, `joiner_kd_scale=0.3` cho joiner | OK |
| `use_kd_ctc=False` với `kd_objective=rnnt` | OK |

### Encoder + Head loading ✅ (sau fix)
| Layer group | Source | Keys | Notes |
|-------------|--------|------|-------|
| Encoder (HuBERT+Zipformer) | Phase 2, strip `asr_model.encoder.` prefix | 827 | load trong `get_encoder_model()` |
| decoder.embedding | Teacher JIT | 1 | same vocab → token semantics đúng |
| joiner.output_linear + decoder_proj | Teacher JIT | 4 | blank bias đúng từ teacher |
| simple_lm_proj | Teacher JIT | 2 | LM distribution từ teacher |
| simple_am_proj | **random init** (trunc_normal std=0.02) | 2 | Phase 2 có old vocab ordering → reinit |
| ctc_output | **random init** (trunc_normal std=0.02) | 2 | Phase 2 có old vocab ordering → reinit |

Source of truth: `init_from_teacher.py` log (7 copied + 4 reinit). Log từ `finetune.py` không dùng để xác nhận số keys vì `missing` trong `load_state_dict(head_state, strict=False)` bao gồm cả encoder keys không nằm trong `head_state`.

---

## Quan sát từ training epoch 1 (old vocab + encoder từ scratch — reference only)

> ⚠️ Run này không có giá trị reference đúng: encoder không load Phase 2, vocab sai 28%. Chỉ dùng để so sánh xu hướng.

| Metric | Start | End epoch 1 |
|--------|-------|-------------|
| `tot_loss` | 10.83 | ~2.27 |
| `kd_joiner_raw` (tot) | 20.45 | ~14.44 → plateau sau batch 350 |
| `kd_rnnt_raw` (tot) | 11.59 | ~11.0 (gần flat — do 28% vocab sai) |
| `simple_loss` (tot) | 7.95 | ~0.30 |
| LR | 1e-3 | 9.98e-4 (Eden gần flat với lr_batches=100k) |

---

## Checklist training mới

- [x] `--bpe-model` đổi sang `zipformer_kd/teacher_tokens.txt` (100% vocab overlap)
- [x] Fix `multi_speaker_model.py:661` → `get_ctc_logits` (latent bug)
- [x] Fix `finetune.py:get_encoder_model` → strip `asr_model.encoder.` prefix (critical)
- [ ] Kill old training, dùng `EXP_DIR=zipformer_kd/exp_kd_v2` để tránh load checkpoint cũ
- [ ] Tăng `NUM_EPOCHS` ≥ 5 cho full training
- [ ] Xem xét `LR_BATCHES=5000 LR_EPOCHS=5` cho LR decay nhanh hơn
- [ ] Sau 200 batch đầu: verify `kd_rnnt_raw` giảm mạnh hơn lần trước (target <8 sau 1k batch)
- [ ] Sau 200–500 batch: chạy `python debug_teacher_student.py --tokens zipformer_kd/teacher_tokens.txt ...`
  - **Pass**: RNNT decode ra text (không phải `(empty)`), non-blank max >0.05 ở ít nhất vài frame
  - **Fail/Collapse**: blank ~100% VÀ decode rỗng — script sẽ in `*** COLLAPSE ***`
  - Blank dominance mà decode rỗng = collapse, không phải healthy blank distribution

---

## Cấu trúc KD loss (flow)

```
Student forward
├── HubertModel (Phase 2 weights, 827 keys) → encoder_out [B,T,512]
├── SpeakerKernel (giữ nguyên)
├── CTC head → ctc_loss
└── RNNT:
    ├── [ctc_only_steps=2000] simple/pruned RNNT tắt trong 2000 bước đầu
    ├── simple_loss + pruned_loss (main RNNT, sau bước 2000)
    ├── rnnt_kd_loss(skip_am_branch=True)
    │   └── LM branch only: KL(student_lm || teacher_lm) × scale=0.05
    │       [active từ bước 0, kể cả trong CTC-only phase]
    └── joiner_diag_kd_loss
        └── diagonal path: KL(student_joiner || teacher_joiner) × scale=0.3
            [active từ bước 0, kể cả trong CTC-only phase]

Teacher (frozen JIT, 100% vocab overlap với student):
├── Mel → teacher_encoder [B,T',512]
├── teacher_joiner logits [B,T',V] (blank ~80%)  ← joiner KD
└── teacher LM logits [B,U,V]                    ← LM KD
    AM branch: SKIP (bias[<unk>]=80, degenerate)
```
