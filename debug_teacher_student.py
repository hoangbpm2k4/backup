"""Debug: compare Teacher JIT vs Student model on single-speaker audio.

Runs both models on the same audio file and shows:
  - Decoded text (teacher CTC, student CTC, student RNNT)
  - Frame-level logit distribution statistics
  - KL divergence between aligned teacher/student distributions

Usage:
    cd /home/vietnam/voice_to_text
    python debug_teacher_student.py \
        --checkpoint zipformer/exp_phase2_supervised/best-valid-loss.pt \
        --teacher hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt \
        --audio data/augmentation_samples/single_clean/07_test_00005702_single-5697.wav
"""

import argparse
import sys
import os

# zipformer_kd must come AFTER zipformer in insert order so it ends up at index 0
# (each insert(0, x) pushes previous to index 1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer_kd"))

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import torchaudio

from tokenizer import TokenEncoder
from icefall.utils import AttributeDict


# ---- Helpers ---------------------------------------------------------------

def load_audio(path: str, device: str) -> torch.Tensor:
    """Load wav, resample to 16kHz, mono, layer-norm → [1, T]."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        length_target = int(len(data) * 16000 / sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, length_target),
            np.arange(len(data)), data,
        )
    wav = torch.from_numpy(data).unsqueeze(0)          # [1, T]
    wav = F.layer_norm(wav, (wav.shape[-1],))           # match training
    return wav.to(device)


def greedy_ctc(logits: torch.Tensor, valid_T: int, blank_id: int = 0):
    """Greedy CTC decode from raw logits [1, T, V]."""
    ids = logits[0, :valid_T].argmax(dim=-1).tolist()
    collapsed, prev = [], -1
    for idx in ids:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return [t for t in collapsed if t != blank_id]


def greedy_rnnt(enc_out, enc_lens, decoder, joiner, blank_id,
                context_size, blank_penalty=0.0, max_sym_per_frame=3):
    """RNNT greedy decode from student encoder output."""
    device = enc_out.device
    T = int(enc_lens[0].item())
    hyp = [blank_id] * context_size
    for t in range(T):
        enc_frame = enc_out[0, t:t+1, :].unsqueeze(0)   # [1, 1, D]
        for _ in range(max_sym_per_frame):
            dec_in = torch.tensor([hyp[-context_size:]], device=device, dtype=torch.long)
            dec_out = decoder(dec_in, need_pad=False)    # [1, 1, D_dec]
            logits = joiner(enc_frame, dec_out)          # [1, 1, vocab]
            if blank_penalty > 0:
                logits[0, 0, blank_id] -= blank_penalty
            y = logits.argmax(dim=-1).item()
            if y == blank_id:
                break
            hyp.append(y)
    return hyp[context_size:]


# ---- Teacher forward -------------------------------------------------------

def run_teacher(jit_path: str, wav: torch.Tensor, device: str, tokenizer,
                tokens_path: str = "zipformer_kd/teacher_tokens.txt"):
    """Run teacher JIT, return (joiner_logits_mapped, out_lens).

    NOTE: teacher.simple_am_proj is degenerate (bias[<unk>]=80.0) and always
    predicts <unk> with near-100% confidence regardless of input.
    Use the JOINER for all teacher comparisons — it gives proper distributions
    (80% blank, entropy=0.12) matching the teacher's actual WER 8.1% quality.
    """
    from teacher import TeacherJIT

    teacher = TeacherJIT(
        jit_path,
        device=torch.device(device),
        student_tokens_path=tokens_path,
        teacher_tokens_path="",
    )
    teacher.eval()

    padding_mask = torch.zeros(wav.shape, dtype=torch.bool, device=device)  # no padding

    # ---- Joiner null-state logits (the correct teacher signal) ----
    jit = object.__getattribute__(teacher, "_jit")
    dev = torch.device(device)
    audio_lens = (~padding_mask).long().sum(dim=1)
    mel, mel_lens = teacher._audio_to_mel(wav, audio_lens)
    mel = mel.to(dev)
    mel_lens = mel_lens.to(dev)

    with torch.no_grad():
        x, x_lens = jit.encoder_embed(mel, mel_lens)
        x_tbd = x.permute(1, 0, 2).contiguous()
        out_tbd, out_lens = jit.encoder.encoder(x_tbd, x_lens)
        enc_out = out_tbd.permute(1, 0, 2).contiguous()          # [B, T, D]

        blank_id = int(jit.decoder.blank_id)
        blank_ids = torch.full((wav.shape[0], 1), blank_id, dtype=torch.long, device=dev)
        dec_out = jit.decoder(blank_ids, need_pad=True)           # [B, 1, D_dec]

        # Joiner on every frame with null decoder state
        joiner_logits = jit.joiner(enc_out, dec_out.expand(-1, enc_out.shape[1], -1))
        # Some joiners need explicit boolean project_input arg:
        if joiner_logits is None or joiner_logits.shape[-1] == 1:
            joiner_logits_list = []
            for t in range(enc_out.shape[1]):
                ef = enc_out[:, t:t+1, :]
                lg = jit.joiner(ef, dec_out)
                joiner_logits_list.append(lg)
            joiner_logits = torch.cat(joiner_logits_list, dim=1)

    T = int(out_lens[0].item())
    joiner_mapped = teacher.map_to_student_vocab(joiner_logits)

    tokens = greedy_ctc(joiner_mapped, T, blank_id=blank_id)
    text = tokenizer.decode(tokens) if tokens else "(empty)"

    print(f"\n[Teacher joiner null-state] '{text}' ({len(tokens)} tokens, T={T} frames)")
    print(f"  Note: simple_am_proj is degenerate (bias[<unk>]=80) — joiner is the correct signal")
    _print_logit_stats("Teacher joiner", joiner_mapped, T)

    return joiner_mapped, out_lens


# ---- Student forward -------------------------------------------------------

def build_student(ckpt_path: str, device: str, use_model: bool = True):
    from hubert_ce import HubertModel
    from decoder import Decoder
    from joiner import Joiner
    from model import AsrModel
    from multi_speaker_model import MultiSpeakerRnntModel

    print(f"\nLoading student checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = AttributeDict({k: v for k, v in ckpt.items() if k != "model"})

    encoder = HubertModel(params)
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    enc_out_dim = max(int(x) for x in str(params.encoder_dim).split(","))
    joiner = Joiner(
        encoder_dim=enc_out_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    asr_model = AsrModel(
        encoder=encoder, decoder=decoder, joiner=joiner,
        encoder_dim=enc_out_dim, decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer, use_ctc=params.use_ctc,
    )
    ms_model = MultiSpeakerRnntModel(
        asr_model=asr_model,
        num_speakers=params.num_speakers,
        use_pyannote_diar=False,
        enable_pre_asr_stream_routing=True,
        spk_kernel_dropout=params.get("spk_kernel_dropout", 0.5),
        bg_threshold=params.get("bg_threshold", 0.5),
    )

    state_key = "model" if use_model else "model_avg"
    if state_key not in ckpt or not ckpt[state_key]:
        state_key = "model"
    missing, unexpected = ms_model.load_state_dict(ckpt[state_key], strict=False)
    if missing:
        print(f"  Missing {len(missing)} keys")
    if unexpected:
        print(f"  Unexpected {len(unexpected)} keys")
    print(f"  step: {params.get('batch_idx_train', '?')}")

    ms_model.eval()
    ms_model = ms_model.to(device)
    return ms_model, params


def run_student(ms_model, params, wav: torch.Tensor, device: str, tokenizer,
                chunk_size: int = 16, left_context: int = 128, blank_penalty: float = 1.0):
    """Run student model, return (ctc_logits, enc_out, enc_lens)."""
    hubert = ms_model.asr_model.encoder
    zipformer = hubert.encoder
    spk_kernels = getattr(ms_model, "spk_kernels", None)

    B = wav.shape[0]
    T_feat_max = wav.shape[1] // 320  # approx

    # Step 1: HuBERT conv frontend
    raw_features = hubert.feature_extractor(wav)     # [1, C, T_feat]
    T_feat = raw_features.shape[2]
    features = raw_features.transpose(1, 2)          # [1, T_feat, D]
    features = hubert.layer_norm(features)
    if hubert.post_extract_proj is not None:
        features = hubert.post_extract_proj(features)
    # no dropout in eval mode

    # Step 2: SpeakerKernel with single speaker = all active
    if spk_kernels is not None and "0" in spk_kernels:
        spk_mask = torch.ones(B, T_feat, device=device)
        bg_mask = torch.zeros(B, T_feat, device=device)
        features = spk_kernels["0"](features, spk_mask, bg_mask)
        print(f"  [Student] SpeakerKernel applied (spk_kernels['0'])")
    else:
        print(f"  [Student] No SpeakerKernel found")

    # Step 3: Zipformer with fixed chunk size
    left_context_chunks = max(left_context // chunk_size, 1)
    original_get_chunk_info = zipformer.get_chunk_info

    def fixed_chunk_info():
        return chunk_size, left_context_chunks

    zipformer.get_chunk_info = fixed_chunk_info

    x_lens = torch.tensor([T_feat], device=device, dtype=torch.long)
    src_key_padding_mask = (
        torch.arange(T_feat, device=device).unsqueeze(0) >= x_lens.unsqueeze(1)
    )
    x = features.transpose(0, 1)                       # [T, 1, D]

    with torch.no_grad():
        enc_out_tbd, enc_lens = zipformer(x, x_lens, src_key_padding_mask=src_key_padding_mask)
        enc_out = enc_out_tbd.transpose(0, 1)           # [1, T_out, D]

    zipformer.get_chunk_info = original_get_chunk_info

    T_enc = int(enc_lens[0].item())
    print(f"  [Student] HuBERT feat frames: {T_feat}, Zipformer out frames: {T_enc}")
    print(f"  [Student] chunk_size={chunk_size}, left_context_chunks={left_context_chunks}")

    # CTC decode
    ctc_logits = None
    if hasattr(ms_model.asr_model, "ctc_output"):
        with torch.no_grad():
            ctc_logits = ms_model.asr_model.ctc_output(enc_out)  # [1, T, V]
        tokens_ctc = greedy_ctc(ctc_logits, T_enc, blank_id=0)
        text_ctc = tokenizer.decode(tokens_ctc) if tokens_ctc else "(empty)"
        non_blank = sum(1 for t in ctc_logits[0, :T_enc].argmax(-1).tolist() if t != 0)
        print(f"  [Student CTC]  '{text_ctc}' ({non_blank}/{T_enc} non-blank)")
        _print_logit_stats("Student CTC", ctc_logits, T_enc)

    # RNNT decode
    with torch.no_grad():
        rnnt_tokens = greedy_rnnt(
            enc_out, enc_lens,
            ms_model.asr_model.decoder,
            ms_model.asr_model.joiner,
            blank_id=params.blank_id,
            context_size=params.context_size,
            blank_penalty=blank_penalty,
        )
    text_rnnt = tokenizer.decode(rnnt_tokens) if rnnt_tokens else "(empty)"
    print(f"  [Student RNNT] '{text_rnnt}' ({len(rnnt_tokens)} tokens)")

    # RNNT null-state logits (for KD comparison, requires zipformer_kd/model.py)
    rnnt_logits = None
    if hasattr(ms_model.asr_model, "get_rnnt_kd_logits"):
        with torch.no_grad():
            rnnt_logits = ms_model.asr_model.get_rnnt_kd_logits(enc_out)
        print(f"  [Student RNNT null-state logits] shape: {rnnt_logits.shape}")
        _print_logit_stats("Student RNNT (null-state)", rnnt_logits, T_enc)
    else:
        print(f"  [Student] get_rnnt_kd_logits not available (zipformer base checkpoint)")

    return ctc_logits, rnnt_logits, enc_out, enc_lens


# ---- Comparison stats -------------------------------------------------------

def _print_logit_stats(label: str, logits: torch.Tensor, valid_T: int):
    """Print entropy, top-1 confidence, blank probability for frame-level logits.

    Also detects collapse: blank dominates BUT top non-blank token never exceeds
    threshold → model emits nothing (false positive for 'blank > content' check).
    """
    lg = logits[0, :valid_T]                            # [T, V]
    probs = F.softmax(lg, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # [T]
    top1_conf = probs.max(-1).values                    # [T]
    blank_prob = probs[:, 0]                            # [T] blank=0

    # Non-blank max: highest non-blank token probability per frame
    non_blank_probs = probs[:, 1:]                      # [T, V-1]
    non_blank_max = non_blank_probs.max(-1).values      # [T]

    print(f"  [{label}] entropy: {entropy.mean():.3f} ± {entropy.std():.3f} nats")
    print(f"  [{label}] top1 confidence: {top1_conf.mean():.3f}")
    print(f"  [{label}] blank prob: {blank_prob.mean():.3f}  non-blank max: {non_blank_max.mean():.3f}")

    # Collapse detection: blank dominates AND non-blank signal too weak
    frames_w_signal = (non_blank_max > 0.05).sum().item()
    if frames_w_signal == 0:
        print(f"  [{label}] *** COLLAPSE: blank dominates, non-blank max never >0.05 ***")
    elif blank_prob.mean() > 0.995:
        print(f"  [{label}] *** WARNING: blank={blank_prob.mean():.4f}, only {frames_w_signal}/{valid_T} frames have non-blank >0.05 ***")


def compare_kl(teacher_logits, teacher_lens,
               student_logits, student_lens,
               temperature: float = 1.0, label: str = ""):
    """Compute KL(student || teacher) after time alignment."""
    T_s = int(student_lens[0].item())
    T_t = int(teacher_lens[0].item())

    # Align teacher to student frame count
    if T_t != T_s:
        tl = teacher_logits[:, :T_t, :].permute(0, 2, 1)     # [1, V, T_t]
        tl = F.interpolate(tl, size=T_s, mode="linear", align_corners=False)
        tl = tl.permute(0, 2, 1)                              # [1, T_s, V]
    else:
        tl = teacher_logits[:, :T_t, :]

    sl = student_logits[:, :T_s, :]

    s_log = F.log_softmax(sl / temperature, dim=-1)
    t_prob = F.softmax(tl / temperature, dim=-1)

    kl_per_frame = F.kl_div(
        s_log.reshape(-1, sl.shape[-1]),
        t_prob.reshape(-1, tl.shape[-1]),
        reduction="none",
    ).sum(-1).reshape(T_s)  # [T_s]

    print(f"  [KL {label}] mean: {kl_per_frame.mean():.4f} nats/frame  "
          f"max: {kl_per_frame.max():.4f}  "
          f"(T_teacher={T_t}, T_student={T_s}, temp={temperature})")

    # Top-5 frame indices with highest KL
    topk = kl_per_frame.topk(min(5, T_s))
    top_times = [f"t={i.item()}(kl={v:.3f})" for v, i in zip(topk.values, topk.indices)]
    print(f"  [KL {label}] highest frames: {', '.join(top_times)}")


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug teacher vs student")
    parser.add_argument("--checkpoint", required=True, help="Student checkpoint .pt")
    parser.add_argument("--teacher", required=True, help="Teacher JIT .pt path")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--tokens", default="zipformer_kd/teacher_tokens.txt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--left-context", type=int, default=128)
    parser.add_argument("--blank-penalty", type=float, default=1.0)
    parser.add_argument("--use-model", action="store_true",
                        help="Use model (not model_avg) from checkpoint")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="KL temperature (matches KD training temperature)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = TokenEncoder(args.tokens)

    wav = load_audio(args.audio, device)
    duration = wav.shape[1] / 16000
    print(f"\nAudio: {args.audio} ({duration:.2f}s)")

    # Ground truth
    txt_path = os.path.splitext(args.audio)[0] + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            gt = f.read().strip()
        print(f"[GT]    '{gt}'")

    print("\n" + "="*70)
    print("TEACHER")
    print("="*70)
    # teacher_logits = joiner null-state logits (mapped to student vocab)
    teacher_logits, teacher_lens = run_teacher(args.teacher, wav, device, tokenizer,
                                                tokens_path=args.tokens)

    print("\n" + "="*70)
    print("STUDENT")
    print("="*70)
    student_ctc_logits, student_rnnt_logits, enc_out, student_lens = run_student(
        *((lambda ms, p: (ms, p))(*build_student(args.checkpoint, device, args.use_model))),
        wav, device, tokenizer,
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        blank_penalty=args.blank_penalty,
    )

    print("\n" + "="*70)
    print("KL COMPARISON (teacher joiner vs student)")
    print("="*70)
    if student_ctc_logits is not None:
        compare_kl(
            teacher_logits, teacher_lens,
            student_ctc_logits, student_lens,
            temperature=1.0, label="teacher-joiner vs student-CTC (T=1)"
        )
        compare_kl(
            teacher_logits, teacher_lens,
            student_ctc_logits, student_lens,
            temperature=args.temperature, label=f"teacher-joiner vs student-CTC (T={args.temperature})"
        )

    if student_rnnt_logits is not None:
        compare_kl(
            teacher_logits, teacher_lens,
            student_rnnt_logits, student_lens,
            temperature=1.0, label="teacher-joiner vs student-RNNT-null (T=1)"
        )
        compare_kl(
            teacher_logits, teacher_lens,
            student_rnnt_logits, student_lens,
            temperature=args.temperature, label=f"teacher-joiner vs student-RNNT-null (T={args.temperature})"
        )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    T_t = int(teacher_lens[0].item())
    T_s = int(student_lens[0].item())
    print(f"Teacher frames: {T_t} (~{T_t/duration:.1f} fps)")
    print(f"Student frames: {T_s} (~{T_s/duration:.1f} fps)")
    print(f"Frame-rate ratio: {T_s/T_t:.2f}x")


if __name__ == "__main__":
    main()
