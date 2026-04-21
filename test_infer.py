"""Quick inference test for multi-speaker streaming RNNT.

Usage:
    python test_inference.py \
        --checkpoint zipformer22/exp_phase2_supervised/checkpoint-9000.pt \
        --audio sample_test/00_test_00001328_overlap2-1291.wav \
        --tokens zipformer22/tokens.txt \
        --pyannote-weight-dir weight \
        --num-speakers 3
"""

import argparse
import sys
import os
import pathlib

# Fix Windows console encoding for Vietnamese
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Fix PosixPath on Windows (checkpoint saved on Linux)
if not hasattr(pathlib, "PosixPath") or sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Add zipformer22 to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer22"))

import torch
import torchaudio

from tokenizer import TokenEncoder


def load_model(args):
    """Reconstruct model from checkpoint."""
    from icefall.utils import AttributeDict
    from model import AsrModel
    from hubert_ce import HubertModel
    from decoder import Decoder
    from joiner import Joiner
    from multi_speaker_model import MultiSpeakerRnntModel
    from streaming_multi_speaker import StreamingMultiSpeakerEncoder

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract params from checkpoint
    if "params" in ckpt:
        params = AttributeDict(ckpt["params"])
    else:
        # Fallback: use default params matching training config
        params = AttributeDict({
            "vocab_size": 2000,
            "blank_id": 0,
            "context_size": 2,
            "encoder_dim": "192,256,384,512,384,256",
            "decoder_dim": 512,
            "joiner_dim": 512,
            "num_encoder_layers": "2,2,3,4,3,2",
            "feedforward_dim": "512,768,1024,1536,1024,768",
            "num_heads": "4,4,4,8,4,4",
            "downsampling_factor": "1,2,4,8,4,2",
            "encoder_unmasked_dim": "192,192,256,256,256,192",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "pos_dim": 48,
            "pos_head_dim": "4",
            "query_head_dim": "32",
            "value_head_dim": "12",
            "pos_enc_type": "abs",
            "attn_type": "",
            "use_transducer": True,
            "use_ctc": True,
            "causal": True,
            "chunk_size": "32",
            "left_context_frames": "128",
            "short_chunk_size": 50,
            "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            "extractor_mode": "default",
            "feature_grad_mult": 1.0,
            "loss_weights": [10],
            "num_classes": [504],
            "logit_temp": 0.1,
            "dropout_input": 0.0,
            "dropout_features": 0.0,
            "mask_prob": 0.65,
            "mask_length": 10,
            "mask_selection": "static",
            "mask_other": 0,
            "no_mask_overlap": False,
            "mask_min_space": 1,
            "mask_channel_prob": 0.0,
            "mask_channel_length": 10,
            "mask_channel_selection": "static",
            "mask_channel_other": 0,
            "no_mask_channel_overlap": False,
            "mask_channel_min_space": 1,
            "skip_masked": False,
            "skip_nomask": False,
            "pred_masked_weight": 1,
            "pred_nomask_weight": 0,
            "do_normalize": True,
            "sample_rate": 16000,
            "label_rate": 50,
            "conv_bias": False,
            "required_seq_len_multiple": 2,
            "untie_final_proj": False,
            "checkpoint_activations": False,
            "inf_check": False,
        })

    # Build model architecture
    encoder = HubertModel(params)
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    joiner = Joiner(
        encoder_dim=max(
            int(x) for x in str(params.encoder_dim).split(",")
        ),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    asr_model = AsrModel(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=max(
            int(x) for x in str(params.encoder_dim).split(",")
        ),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=True,
        use_ctc=getattr(params, "use_ctc", True),
    )

    ms_model = MultiSpeakerRnntModel(
        asr_model=asr_model,
        num_speakers=args.num_speakers,
        use_pyannote_diar=False,  # PyAnnote runs externally
        enable_pre_asr_stream_routing=True,
    )

    # Load weights
    model_state = ckpt.get("model", ckpt)
    missing, unexpected = ms_model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  Missing {len(missing)} keys: {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected {len(unexpected)} keys: {unexpected[:5]}...")

    ms_model.eval()
    return ms_model, params


def load_pyannote(weight_dir, num_speakers, device):
    """Load PyAnnote as separate diarization model."""
    from speaker_modules import PyAnnoteSegmentationWrapper
    print(f"Loading PyAnnote from: {weight_dir}")
    pya = PyAnnoteSegmentationWrapper(
        weight_dir=weight_dir,
        num_speakers=num_speakers,
        freeze=True,
    )
    pya.eval()
    pya = pya.to(device)
    print(f"  PyAnnote loaded ({sum(p.numel() for p in pya.parameters())} params)")
    return pya


def run_streaming_inference(ms_model, audio_path, tokenizer, args):
    """Run streaming inference on a single audio file."""
    from streaming_multi_speaker import StreamingMultiSpeakerEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build streaming encoder wrapper BEFORE moving to device
    # (_compute_causal_trim runs dummy forward on CPU)
    streaming_encoder = StreamingMultiSpeakerEncoder(ms_model)
    streaming_encoder = streaming_encoder.to(device)

    # Load PyAnnote separately
    pyannote = load_pyannote(args.pyannote_weight_dir, args.num_speakers, device)

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)  # [1, T_samples]

    print(f"\nAudio: {audio_path}")
    print(f"Duration: {waveform.shape[1] / 16000:.2f}s")
    print(f"Device: {device}")
    print(f"Num speakers: {args.num_speakers}")

    # Step 1: Run PyAnnote on full audio → get full activity
    print("Running PyAnnote diarization...")
    with torch.no_grad():
        # Get activity at native frame rate, then we'll slice per chunk
        full_activity_native = pyannote.forward_native(waveform)  # [1, T_pya, S]
    print(f"  PyAnnote output: {full_activity_native.shape} "
          f"({full_activity_native.shape[1]} frames)")

    # Init state
    chunk_samples = args.chunk_samples
    state = streaming_encoder.init_state(
        batch_size=1,
        device=device,
        left_context_frames=args.left_context_frames,
    )

    T_total = waveform.shape[1]
    num_chunks = (T_total + chunk_samples - 1) // chunk_samples

    print(f"Chunk size: {chunk_samples} samples ({chunk_samples / 16000 * 1000:.0f}ms)")
    print(f"Total chunks: {num_chunks}")
    print(f"Left context frames: {args.left_context_frames}")
    print()

    # Step 2: Stream chunk by chunk, slicing activity per chunk
    all_tokens = [[] for _ in range(args.num_speakers)]
    samples_per_frame = 320  # HuBERT frontend total stride

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, T_total)
            audio_chunk = waveform[:, start:end]  # [1, chunk_len]

            # Pad last chunk if needed
            if audio_chunk.shape[1] < chunk_samples:
                pad_len = chunk_samples - audio_chunk.shape[1]
                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad_len))
                sample_lens = torch.tensor([end - start], device=device)
            else:
                sample_lens = torch.tensor([chunk_samples], device=device)

            # Estimate how many ASR feature frames this chunk produces
            valid_samples = int(sample_lens[0].item())
            est_feat_frames = max(1, valid_samples // samples_per_frame)

            # Resample full activity to target length for this chunk
            chunk_activity = pyannote.forward(
                audio_chunk, target_len=est_feat_frames
            )  # [1, est_feat_frames, S]

            h_all, h_lens, token_deltas, state = streaming_encoder.streaming_step(
                audio_chunk=audio_chunk,
                state=state,
                spk_activity_chunk=chunk_activity,
                sample_lens=sample_lens,
                decode=True,
                beam=1,
                max_sym_per_frame=args.max_sym_per_frame,
            )

            # Collect tokens
            if token_deltas is not None:
                for s in range(args.num_speakers):
                    stream_idx = s  # B=1, so stream s = index s
                    if stream_idx < len(token_deltas):
                        new_toks = token_deltas[stream_idx]
                        if new_toks:
                            all_tokens[s].extend(new_toks)

            if (i + 1) % 10 == 0 or i == num_chunks - 1:
                print(f"  Chunk {i + 1}/{num_chunks} processed")

    # Decode tokens to text
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for s in range(args.num_speakers):
        # Filter out blank (0) and special tokens
        real_tokens = [t for t in all_tokens[s] if t > 0]
        text = tokenizer.decode(real_tokens) if real_tokens else "(silence)"
        print(f"  SPK{s}: {text}")

    # Show ground truth if available
    txt_path = audio_path.replace(".wav", ".txt")
    if os.path.exists(txt_path):
        print("\n" + "-" * 60)
        print("GROUND TRUTH")
        print("-" * 60)
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    print(f"  {line.rstrip()}")
                except UnicodeEncodeError:
                    print(f"  {line.rstrip().encode('utf-8', errors='replace').decode('utf-8')}")

    return all_tokens


def main():
    parser = argparse.ArgumentParser(description="Multi-speaker streaming RNNT inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio .wav file (16kHz mono)")
    parser.add_argument("--tokens", type=str, default="zipformer22/tokens.txt",
                        help="Path to tokens.txt")
    parser.add_argument("--pyannote-weight-dir", type=str, default="weight",
                        help="Path to pyannote model weights")
    parser.add_argument("--num-speakers", type=int, default=3,
                        help="Max number of speakers")
    parser.add_argument("--chunk-samples", type=int, default=10240,
                        help="Chunk size in samples (default: 10240 = 32 frames × 320)")
    parser.add_argument("--left-context-frames", type=int, default=128,
                        help="Left context frames for Zipformer streaming")
    parser.add_argument("--max-sym-per-frame", type=int, default=1,
                        help="Max symbols per encoder frame (1 for most, 2+ for tonal)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda)")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokens}")
    tokenizer = TokenEncoder(args.tokens)
    print(f"  Vocab size: {tokenizer.get_piece_size()}")

    # Load model
    ms_model, params = load_model(args)

    # Run inference
    run_streaming_inference(ms_model, args.audio, tokenizer, args)


if __name__ == "__main__":
    main()
