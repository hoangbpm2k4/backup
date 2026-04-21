"""Quick ASR sanity check — does the model produce readable text?

Uses the SAME forward path as training (forward_train_routing steps 1-4)
with oracle activity (speaker 0 = all ones). Tests CTC greedy + RNNT greedy.

Usage:
    cd /home/vietnam/voice_to_text
    python test_asr_quick.py \
        --checkpoint zipformer/exp_phase2_supervised/checkpoint-12500.pt \
        --audio data/augmentation_samples/single_clean/00_*.wav
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer"))

import torch
import soundfile as sf
import numpy as np

from tokenizer import TokenEncoder
from icefall.utils import AttributeDict


def build_model(ckpt_path: str, device: str = "cpu", use_model_avg: bool = True):
    """Reconstruct MultiSpeakerRnntModel from checkpoint."""
    from hubert_ce import HubertModel
    from decoder import Decoder
    from joiner import Joiner
    from model import AsrModel
    from multi_speaker_model import MultiSpeakerRnntModel

    print(f"Loading checkpoint: {ckpt_path}")
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

    # Load weights
    if use_model_avg and "model_avg" in ckpt and ckpt["model_avg"]:
        model_state = ckpt["model_avg"]
        print("  Using model_avg (EMA)")
    else:
        model_state = ckpt["model"]
        print("  Using model (latest step)")
    missing, unexpected = ms_model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  Missing {len(missing)} keys")
    if unexpected:
        print(f"  Unexpected {len(unexpected)} keys")
    print(f"  batch_idx_train: {params.get('batch_idx_train', '?')}")

    ms_model.eval()
    ms_model = ms_model.to(device)
    return ms_model, params


def encode_full_sequence(ms_model, waveform, device, decode_chunk_size=16,
                         left_context_frames=64):
    """Encode using EXACT same path as forward_train_routing (steps 1-4).

    This matches training: HuBERT conv → layer_norm → proj → dropout →
    speaker_kernel(spk=1, bg=0) → Zipformer with causal chunk mask.

    IMPORTANT: Override Zipformer's random chunk sampling with fixed
    decode_chunk_size to ensure deterministic inference.
    """
    hubert = ms_model.asr_model.encoder
    zipformer = hubert.encoder
    speaker_kernel = getattr(ms_model, "speaker_kernel", None)

    B = waveform.shape[0]

    # Step 1: HuBERT conv frontend
    raw_features = hubert.feature_extractor(waveform)  # [B, C, T_feat]
    T_feat = raw_features.shape[2]

    features = raw_features.transpose(1, 2)  # [B, T_feat, D]
    features = hubert.layer_norm(features)
    if hubert.post_extract_proj is not None:
        features = hubert.post_extract_proj(features)
    features = hubert.dropout_input(features)

    # Step 2-3: speaker kernel (single speaker, all active)
    if speaker_kernel is not None:
        spk_mask = torch.ones(B, T_feat, device=device)
        bg_mask = torch.zeros(B, T_feat, device=device)
        features = speaker_kernel(features, spk_mask, bg_mask)

    # Step 4: Zipformer forward with FIXED chunk size for deterministic inference
    # Override get_chunk_info to use decode_chunk_size instead of random sampling
    original_get_chunk_info = zipformer.get_chunk_info
    left_context_chunks = max(left_context_frames // decode_chunk_size, 1)

    def fixed_chunk_info():
        return decode_chunk_size, left_context_chunks

    zipformer.get_chunk_info = fixed_chunk_info
    print(f"  Inference: chunk_size={decode_chunk_size}, left_context_chunks={left_context_chunks}")

    x_lens = torch.tensor([T_feat], device=device, dtype=torch.long)
    src_key_padding_mask = (
        torch.arange(T_feat, device=device).unsqueeze(0) >= x_lens.unsqueeze(1)
    )
    x = features.transpose(0, 1)  # [T, B, D]
    enc_out_tbd, enc_lens = zipformer(x, x_lens, src_key_padding_mask=src_key_padding_mask)
    enc_out = enc_out_tbd.transpose(0, 1)  # [B, T_out, D]

    # Restore original
    zipformer.get_chunk_info = original_get_chunk_info

    return enc_out, enc_lens


def ctc_greedy_decode(ms_model, enc_out, enc_lens):
    """CTC greedy: argmax → collapse repeats → remove blanks."""
    ctc_output = ms_model.asr_model.ctc_output(enc_out)  # [B, T, vocab]
    ids = ctc_output.argmax(dim=-1)[0].tolist()
    T = int(enc_lens[0].item())

    non_blank = sum(1 for i in ids[:T] if i != 0)

    # Collapse repeats + remove blanks
    collapsed = []
    prev = -1
    for idx in ids[:T]:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    tokens = [t for t in collapsed if t != 0]
    return tokens, non_blank, T


def rnnt_greedy_decode(ms_model, enc_out, enc_lens, max_sym_per_frame=3,
                       blank_penalty=0.0):
    """RNNT greedy: frame-by-frame decoder + joiner.

    Args:
        blank_penalty: Subtract this from blank logit to discourage blank emission.
            Typical values: 0.0 (off), 0.5-2.0 (mild to aggressive).
    """
    decoder = ms_model.asr_model.decoder
    joiner = ms_model.asr_model.joiner
    blank_id = decoder.blank_id
    context_size = decoder.context_size
    device = enc_out.device

    T = int(enc_lens[0].item())
    hyp = [blank_id] * context_size

    for t in range(T):
        enc_frame = enc_out[0, t: t + 1, :].unsqueeze(0)  # [1, 1, D]

        for _ in range(max_sym_per_frame):
            decoder_input = torch.tensor(
                [hyp[-context_size:]], device=device, dtype=torch.long
            )
            dec_out = decoder(decoder_input, need_pad=False)  # [1, 1, D_dec]
            logits = joiner(enc_frame, dec_out)  # [1, 1, vocab]

            if blank_penalty > 0:
                logits[0, 0, blank_id] -= blank_penalty

            y = logits.argmax(dim=-1).item()

            if y == blank_id:
                break
            hyp.append(y)

    return hyp[context_size:]


def main():
    parser = argparse.ArgumentParser(description="Quick ASR test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, nargs="+", required=True)
    parser.add_argument(
        "--tokens",
        "--token",
        dest="tokens",
        type=str,
        default="zipformer_kd/teacher_tokens.txt",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-model", action="store_true",
                        help="Use model weights instead of model_avg")
    parser.add_argument("--decode-chunk-size", type=int, default=16,
                        help="Fixed chunk size for inference (must match training config)")
    parser.add_argument("--left-context-frames", type=int, default=64,
                        help="Left context frames for inference")
    parser.add_argument("--blank-penalty", type=float, default=0.0,
                        help="Penalty subtracted from blank logit (0=off, try 0.5-2.0)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = TokenEncoder(args.tokens)
    print(f"Vocab: {tokenizer.get_piece_size()} tokens")

    ms_model, params = build_model(
        args.checkpoint, device, use_model_avg=not args.use_model
    )

    for audio_path in args.audio:
        print(f"\n{'=' * 60}")
        print(f"Audio: {audio_path}")

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            length_target = int(len(data) * 16000 / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, length_target),
                np.arange(len(data)), data,
            )
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]

        # Layer-norm on waveform — matches training (dataset.py)
        waveform = torch.nn.functional.layer_norm(waveform, (waveform.shape[-1],))

        duration = waveform.shape[1] / 16000
        print(f"Duration: {duration:.2f}s")

        waveform = waveform.to(device)

        with torch.no_grad():
            enc_out, enc_lens = encode_full_sequence(
                ms_model, waveform, device,
                decode_chunk_size=args.decode_chunk_size,
                left_context_frames=args.left_context_frames,
            )

            # CTC decode
            ctc_tokens, ctc_non_blank, ctc_total = ctc_greedy_decode(
                ms_model, enc_out, enc_lens
            )
            ctc_text = tokenizer.decode(ctc_tokens) if ctc_tokens else "(empty)"
            print(f"  [CTC]  {ctc_text}  ({ctc_non_blank}/{ctc_total} non-blank)")

            # RNNT decode
            rnnt_tokens = rnnt_greedy_decode(
                ms_model, enc_out, enc_lens, blank_penalty=args.blank_penalty
            )
            rnnt_text = tokenizer.decode(rnnt_tokens) if rnnt_tokens else "(empty)"
            print(f"  [RNNT] {rnnt_text}  ({len(rnnt_tokens)} tokens)")

        # Ground truth
        txt_path = os.path.splitext(audio_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                gt = f.read().strip()
            print(f"  [GT]   {gt}")


if __name__ == "__main__":
    main()
