"""Test inference using the EXACT training forward path.

Loads a real batch from the training dataset and runs forward_train_routing
to verify the model produces non-blank CTC/RNNT predictions.

Usage:
    cd /home/vietnam/voice_to_text
    python test_asr_training_path.py \
        --checkpoint zipformer/exp_phase2_supervised/epoch-1.pt
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer"))

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

from tokenizer import TokenEncoder


def load_model(ckpt_path, device):
    from icefall.utils import AttributeDict
    from hubert_ce import HubertModel
    from decoder import Decoder
    from joiner import Joiner
    from model import AsrModel
    from multi_speaker_model import MultiSpeakerRnntModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = AttributeDict({k: v for k, v in ckpt.items()
                            if k not in ("model", "model_avg", "optimizer")})

    encoder = HubertModel(params)
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    joiner = Joiner(
        encoder_dim=max(int(x) for x in str(params.encoder_dim).split(",")),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    asr_model = AsrModel(
        encoder=encoder, decoder=decoder, joiner=joiner,
        encoder_dim=max(int(x) for x in str(params.encoder_dim).split(",")),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=True,
        use_ctc=getattr(params, "use_ctc", True),
    )
    ms_model = MultiSpeakerRnntModel(
        asr_model=asr_model,
        num_speakers=3,
        use_pyannote_diar=False,
        enable_pre_asr_stream_routing=True,
    )

    # Load model_avg if available, else model
    state = ckpt.get("model_avg") or ckpt["model"]
    missing, unexpected = ms_model.load_state_dict(state, strict=False)
    print(f"Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    ms_model.eval()
    ms_model = ms_model.to(device)
    return ms_model, params


def test_with_training_path(ms_model, audio_path, tokenizer, device):
    """Run the EXACT forward_train_routing path and check CTC predictions."""

    # Load audio
    data, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        length_target = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data) - 1, length_target),
                         np.arange(len(data)), data)

    waveform = torch.from_numpy(data).unsqueeze(0).to(device)  # [1, T]
    waveform = F.layer_norm(waveform, (waveform.shape[-1],))

    dur = waveform.shape[1] / 16000
    print(f"\nAudio: {audio_path} ({dur:.2f}s)")

    hubert = ms_model.asr_model.encoder
    zipformer = hubert.encoder
    speaker_kernel = ms_model.speaker_kernel

    B = 1
    S = int(ms_model.S)

    with torch.no_grad():
        # === EXACT training path (forward_train_routing) ===

        # Step 1: HuBERT conv frontend
        padding_mask = torch.zeros_like(waveform, dtype=torch.bool)
        valid_samples = (~padding_mask).sum(dim=1).long()

        raw_features = hubert.feature_extractor(waveform)  # [1, C, T_feat]
        T_feat = raw_features.shape[2]

        x_lens_batch = ms_model._samples_to_feature_lengths(
            valid_samples, hubert.feature_extractor
        ).clamp(min=0, max=T_feat)

        features = raw_features.transpose(1, 2)  # [1, T_feat, D]
        features = hubert.layer_norm(features)
        if hubert.post_extract_proj is not None:
            features = hubert.post_extract_proj(features)
        features = hubert.dropout_input(features)

        print(f"  Features: shape={features.shape}, mean={features.mean():.4f}, std={features.std():.4f}")
        print(f"  x_lens_batch={x_lens_batch.tolist()}, T_feat={T_feat}")

        # Step 2: oracle activity at 100fps, then resample to T_feat
        frames_100fps = max(1, int(dur * 100))
        spk_activity = torch.zeros(B, frames_100fps, S, device=device)
        spk_activity[:, :, 0] = 1.0  # speaker 0 active everywhere

        activity_full = ms_model._resample_activity(spk_activity, T_feat)
        print(f"  Activity: shape={activity_full.shape}, spk0_mean={activity_full[:,:,0].mean():.4f}")

        # Step 3: resolve stream pairs (SSA target = speaker 0)
        target_speaker_ids = torch.zeros(B, dtype=torch.long, device=device)

        stream_batch_idx, stream_slot_idx = ms_model._resolve_pre_asr_stream_pairs(
            activity_full=activity_full,
            target_speaker_ids=target_speaker_ids,
            active_threshold=ms_model.pre_asr_active_threshold,
        )
        N = stream_batch_idx.numel()
        print(f"  Streams: N={N}, batch_idx={stream_batch_idx.tolist()}, slot_idx={stream_slot_idx.tolist()}")

        if N == 0:
            print("  ERROR: No active streams resolved!")
            return

        # Step 3b: speaker kernel masks
        routed_features = features.index_select(0, stream_batch_idx)
        routed_activity = activity_full.index_select(0, stream_batch_idx)

        slot_index = stream_slot_idx.view(N, 1, 1).expand(-1, T_feat, 1)
        spk_targets = routed_activity.gather(2, slot_index).squeeze(-1)

        bg_targets = torch.stack([
            speaker_kernel.build_bg_mask(activity_full[b:b+1], int(s))[0]
            for b, s in zip(stream_batch_idx.tolist(), stream_slot_idx.tolist())
        ], dim=0)

        print(f"  spk_targets: mean={spk_targets.mean():.4f}, bg_targets: mean={bg_targets.mean():.4f}")

        routed_features = speaker_kernel(routed_features, spk_targets, bg_targets)
        print(f"  After speaker_kernel: mean={routed_features.mean():.4f}, std={routed_features.std():.4f}")

        # Step 4: Zipformer forward
        x_lens_stream = x_lens_batch.index_select(0, stream_batch_idx)
        src_key_padding_mask = (
            torch.arange(T_feat, device=device).unsqueeze(0) >= x_lens_stream.unsqueeze(1)
        )

        x = routed_features.transpose(0, 1)  # [T, N, D]
        enc_out_tbd, enc_lens = zipformer(x, x_lens_stream, src_key_padding_mask=src_key_padding_mask)
        enc_out = enc_out_tbd.transpose(0, 1)  # [N, T_out, D]

        print(f"  Encoder out: shape={enc_out.shape}, mean={enc_out.mean():.4f}, std={enc_out.std():.4f}")
        print(f"  enc_lens={enc_lens.tolist()}")

        # === CTC decode ===
        ctc_logits = ms_model.asr_model.ctc_output(enc_out)  # [N, T, vocab]
        T_out = ctc_logits.shape[1]

        # Check blank vs non-blank
        blank_logit_mean = ctc_logits[:, :, 0].mean().item()
        non_blank_max = ctc_logits[:, :, 1:].max().item()
        non_blank_frames = (ctc_logits.argmax(dim=-1) != 0).sum().item()
        print(f"  CTC: blank_logit_mean={blank_logit_mean:.4f}, non_blank_max={non_blank_max:.4f}")
        print(f"  CTC: {non_blank_frames}/{T_out} non-blank frames")

        # Show top predictions per frame
        ids = ctc_logits.argmax(dim=-1)[0].tolist()
        probs = ctc_logits.softmax(dim=-1)
        print(f"  Sample frames (id, prob):")
        for t in [0, T_out//4, T_out//2, 3*T_out//4, T_out-1]:
            if t < T_out:
                top3_ids = probs[0, t].topk(3)
                tops = [(int(top3_ids.indices[j]), f"{top3_ids.values[j]:.4f}") for j in range(3)]
                print(f"    frame {t}: {tops}")

        # Greedy CTC
        collapsed = []
        prev = -1
        for idx in ids[:int(enc_lens[0].item())]:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        toks = [t for t in collapsed if t != 0]
        text = tokenizer.decode(toks) if toks else "(empty)"
        print(f"\n  [CTC]  {text}")
        print(f"         ({len(toks)} tokens: {toks[:20]})")

        # === RNNT greedy decode ===
        decoder = ms_model.asr_model.decoder
        joiner = ms_model.asr_model.joiner
        context_size = decoder.context_size
        blank_id = 0
        hyp = [blank_id] * context_size
        max_sym_per_frame = 3

        for t in range(int(enc_lens[0].item())):
            enc_frame = enc_out[0, t:t+1, :].unsqueeze(0)  # [1, 1, D]
            for _ in range(max_sym_per_frame):
                decoder_input = torch.tensor(
                    [hyp[-context_size:]], device=device, dtype=torch.long
                )
                dec_out = decoder(decoder_input, need_pad=False)
                logits = joiner(enc_frame, dec_out)
                y = logits.argmax(dim=-1).item()
                if y == blank_id:
                    break
                hyp.append(y)

        rnnt_toks = [t for t in hyp[context_size:] if t != blank_id]
        rnnt_text = tokenizer.decode(rnnt_toks) if rnnt_toks else "(empty)"
        print(f"  [RNNT] {rnnt_text}")
        print(f"         ({len(rnnt_toks)} tokens: {rnnt_toks[:20]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", default="data/augmentation_samples/single_clean/00_test_00007302_single-7296.wav")
    parser.add_argument("--tokens", default="zipformer/tokens.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TokenEncoder(args.tokens)
    ms_model, params = load_model(args.checkpoint, device)

    # Test with individual audio files
    import glob
    audio_files = sorted(glob.glob(args.audio)) if "*" in args.audio else [args.audio]
    for af in audio_files[:3]:
        test_with_training_path(ms_model, af, tokenizer, device)


if __name__ == "__main__":
    main()
