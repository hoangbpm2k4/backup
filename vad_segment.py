#!/usr/bin/env python3
"""
TEN-VAD Streaming Audio Segmentation Script
=============================================
Fully streaming VAD: reads audio in small chunks, processes frame-by-frame
via state machine, and writes segments on-the-fly without loading entire
files into memory.

Usage:
    python vad_segment.py \
        --input-dir ./data/augmentation_samples \
        --output-dir ./data/vad_segments \
        --threshold 0.45 \
        --hop-size 256 \
        --min-speech-duration 0.32 \
        --max-speech-duration 0.8 \
        --min-silence-duration 0.2 \
        --padding 0.02
"""

import sys
import os
import argparse
import json
import glob
import struct
import wave

import numpy as np
from ten_vad import TenVad


SR = 16000  # TEN-VAD requires 16kHz


def parse_args():
    parser = argparse.ArgumentParser(description="TEN-VAD Streaming Audio Segmentation")
    parser.add_argument("--input-dir", required=True,
                        help="Thư mục chứa audio files (duyệt đệ quy)")
    parser.add_argument("--output-dir", required=True,
                        help="Thư mục lưu segments đã cắt")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Ngưỡng VAD probability (0.0-1.0). Default: 0.45")
    parser.add_argument("--hop-size", type=int, default=256,
                        help="Frame size in samples. 256=16ms, 160=10ms. Default: 256")
    parser.add_argument("--min-speech-duration", type=float, default=0.32,
                        help="Thời gian tối thiểu của segment speech (giây). Default: 0.32")
    parser.add_argument("--min-silence-duration", type=float, default=0.2,
                        help="Thời gian silence tối thiểu để tách segments (giây). Default: 0.2")
    parser.add_argument("--max-speech-duration", type=float, default=0.8,
                        help="Thời gian tối đa của segment speech (giây). Segment dài hơn sẽ bị chia nhỏ. Default: 0.8. Set 0 để tắt.")
    parser.add_argument("--padding", type=float, default=0.02,
                        help="Padding thêm vào đầu/cuối mỗi segment (giây). Default: 0.02")
    return parser.parse_args()


class StreamingVADSegmenter:
    """
    Streaming VAD state machine.
    Feeds audio frame-by-frame, detects speech segments on-the-fly,
    and emits completed segments without needing the full file in memory.

    States:
        SILENCE  -> waiting for speech
        SPEECH   -> accumulating speech frames
        TRAILING -> speech ended, counting silence frames to decide merge or emit
    """

    STATE_SILENCE = 0
    STATE_SPEECH = 1
    STATE_TRAILING = 2

    def __init__(self, hop_size, threshold, min_speech_duration,
                 max_speech_duration, min_silence_duration, padding):
        self.hop_size = hop_size
        self.threshold = threshold
        self.sr = SR

        # Convert durations to frame counts
        self.frame_duration = hop_size / self.sr
        self.min_speech_frames = int(min_speech_duration / self.frame_duration)
        self.max_speech_frames = int(max_speech_duration / self.frame_duration) if max_speech_duration > 0 else 0
        self.min_silence_frames = int(min_silence_duration / self.frame_duration)
        self.padding_samples = int(padding * self.sr)

        # VAD engine
        self.vad = TenVad(hop_size, threshold)

        # State
        self.state = self.STATE_SILENCE
        self.frame_index = 0
        self.speech_start_frame = 0
        self.silence_counter = 0

        # Ring buffer for padding: keep last N samples for pre-padding
        self.padding_buffer = np.zeros(self.padding_samples, dtype=np.int16)

        # Audio accumulator for current speech segment
        self.speech_audio = []  # list of np.array chunks
        self.pre_padding = None  # snapshot of padding_buffer at speech start

        # Completed segments
        self.completed_segments = []

    def _emit_segment(self, include_trailing_silence=False):
        """Finalize current speech segment and add to completed list."""
        if not self.speech_audio:
            return

        speech_frames_count = sum(len(a) for a in self.speech_audio) // self.hop_size

        # Check minimum duration
        if speech_frames_count < self.min_speech_frames:
            self.speech_audio = []
            self.pre_padding = None
            return

        # Build segment audio
        raw_audio = np.concatenate(self.speech_audio)

        # Add pre-padding
        if self.pre_padding is not None and len(self.pre_padding) > 0:
            raw_audio = np.concatenate([self.pre_padding, raw_audio])

        # Add post-padding (will be filled by next frames, for now use zeros)
        # Post-padding is handled by keeping extra trailing frames
        post_pad = np.zeros(self.padding_samples, dtype=np.int16)
        raw_audio = np.concatenate([raw_audio, post_pad])

        start_sample = max(0, self.speech_start_frame * self.hop_size - self.padding_samples)
        end_sample = start_sample + len(raw_audio)

        self.completed_segments.append({
            "audio": raw_audio,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_time": round(start_sample / self.sr, 3),
            "end_time": round(end_sample / self.sr, 3),
            "duration": round(len(raw_audio) / self.sr, 3),
        })

        self.speech_audio = []
        self.pre_padding = None

    def _maybe_split_segment(self):
        """If current accumulated speech exceeds max duration, emit and start new."""
        if self.max_speech_frames <= 0:
            return
        speech_samples = sum(len(a) for a in self.speech_audio)
        max_samples = self.max_speech_frames * self.hop_size
        if speech_samples >= max_samples:
            self._emit_segment()
            # Start new segment from current frame
            self.speech_start_frame = self.frame_index
            self.pre_padding = self.padding_buffer.copy()

    def feed_frame(self, frame_data):
        """
        Feed one frame (hop_size samples, int16) into the streaming segmenter.
        Returns list of newly completed segments (may be empty).
        """
        prob, flag = self.vad.process(frame_data)
        is_speech = (flag == 1)

        new_segments = []

        if self.state == self.STATE_SILENCE:
            if is_speech:
                # Transition: SILENCE -> SPEECH
                self.state = self.STATE_SPEECH
                self.speech_start_frame = self.frame_index
                self.pre_padding = self.padding_buffer.copy()
                self.speech_audio = [frame_data.copy()]
            # else: stay in silence

        elif self.state == self.STATE_SPEECH:
            if is_speech:
                self.speech_audio.append(frame_data.copy())
                self._maybe_split_segment()
            else:
                # Start counting silence
                self.state = self.STATE_TRAILING
                self.silence_counter = 1
                self.speech_audio.append(frame_data.copy())  # include in segment for now

        elif self.state == self.STATE_TRAILING:
            if is_speech:
                # Speech resumed - merge gap (stay in speech)
                self.state = self.STATE_SPEECH
                self.speech_audio.append(frame_data.copy())
                self.silence_counter = 0
                self._maybe_split_segment()
            else:
                self.silence_counter += 1
                self.speech_audio.append(frame_data.copy())  # buffer trailing
                if self.silence_counter >= self.min_silence_frames:
                    # Silence long enough - emit segment
                    # Remove trailing silence frames from audio
                    trim_samples = self.silence_counter * self.hop_size
                    if trim_samples > 0 and self.speech_audio:
                        full = np.concatenate(self.speech_audio)
                        full = full[:-trim_samples] if trim_samples < len(full) else full
                        self.speech_audio = [full]
                    self._emit_segment()
                    new_segments = self._drain_completed()
                    self.state = self.STATE_SILENCE
                    self.silence_counter = 0

        # Update padding ring buffer
        pad_len = self.padding_samples
        if pad_len > 0:
            if len(frame_data) >= pad_len:
                self.padding_buffer[:] = frame_data[-pad_len:]
            else:
                shift = pad_len - len(frame_data)
                self.padding_buffer[:shift] = self.padding_buffer[len(frame_data):]
                self.padding_buffer[shift:] = frame_data

        self.frame_index += 1
        return new_segments

    def flush(self):
        """Call after all frames are fed. Emits any remaining speech segment."""
        if self.state in (self.STATE_SPEECH, self.STATE_TRAILING):
            # Trim trailing silence if in TRAILING state
            if self.state == self.STATE_TRAILING and self.silence_counter > 0:
                trim_samples = self.silence_counter * self.hop_size
                if trim_samples > 0 and self.speech_audio:
                    full = np.concatenate(self.speech_audio)
                    full = full[:-trim_samples] if trim_samples < len(full) else full
                    self.speech_audio = [full]
            self._emit_segment()
        return self._drain_completed()

    def _drain_completed(self):
        """Return and clear completed segments."""
        segs = self.completed_segments
        self.completed_segments = []
        return segs


def write_wav_segment(filepath, audio_data, sr=SR):
    """Write int16 mono audio to WAV file."""
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())


def stream_wav_frames(wav_path, hop_size):
    """
    Generator: stream WAV file frame-by-frame without loading entire file.
    Yields np.ndarray of int16 with shape (hop_size,) per frame.
    """
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

        if framerate != SR:
            raise ValueError(f"Expected {SR}Hz, got {framerate}Hz")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit, got {sampwidth * 8}-bit")

        bytes_per_sample = sampwidth * n_channels
        chunk_bytes = hop_size * bytes_per_sample

        while True:
            raw = wf.readframes(hop_size)
            if len(raw) < chunk_bytes:
                break  # not enough for a full frame
            samples = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                samples = samples[::n_channels]  # take first channel
            yield samples


def process_file_streaming(wav_path, output_dir, hop_size, threshold,
                           min_speech_duration, max_speech_duration,
                           min_silence_duration, padding):
    """Process a single WAV file using fully streaming VAD."""

    # Validate format first (quick header check)
    try:
        with wave.open(wav_path, 'rb') as wf:
            if wf.getframerate() != SR:
                print(f"  [SKIP] {wav_path}: sample rate={wf.getframerate()}, expected {SR}")
                return None
            if wf.getsampwidth() != 2:
                print(f"  [SKIP] {wav_path}: {wf.getsampwidth() * 8}-bit, expected 16-bit")
                return None
            total_frames = wf.getnframes()
            source_duration = total_frames / SR
    except Exception as e:
        print(f"  [SKIP] {wav_path}: {e}")
        return None

    # Create streaming segmenter
    segmenter = StreamingVADSegmenter(
        hop_size, threshold,
        min_speech_duration, max_speech_duration,
        min_silence_duration, padding
    )

    # Stream through file frame-by-frame
    all_segments = []
    for frame in stream_wav_frames(wav_path, hop_size):
        new_segs = segmenter.feed_frame(frame)
        all_segments.extend(new_segs)

    # Flush remaining
    final_segs = segmenter.flush()
    all_segments.extend(final_segs)

    if not all_segments:
        return {"source": wav_path, "source_duration": round(source_duration, 3),
                "segments": [], "num_segments": 0, "merged_file": None}

    # Save segments
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    segment_files = []
    merged_audio_parts = []

    for idx, seg in enumerate(all_segments):
        audio = seg.pop("audio")
        out_filename = f"{basename}_seg{idx:03d}.wav"
        out_path = os.path.join(output_dir, out_filename)
        write_wav_segment(out_path, audio)
        seg["filename"] = out_filename
        segment_files.append(seg)
        merged_audio_parts.append(audio)

    # Write merged verification file
    merged_filename = None
    if merged_audio_parts:
        merged_audio = np.concatenate(merged_audio_parts)
        merged_filename = f"{basename}_merged.wav"
        merged_path = os.path.join(output_dir, merged_filename)
        write_wav_segment(merged_path, merged_audio)

    return {
        "source": wav_path,
        "source_duration": round(source_duration, 3),
        "num_segments": len(segment_files),
        "segments": segment_files,
        "merged_file": merged_filename,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("TEN-VAD Streaming Audio Segmentation")
    print("=" * 60)
    print(f"  Input dir:            {args.input_dir}")
    print(f"  Output dir:           {args.output_dir}")
    print(f"  Threshold:            {args.threshold}")
    print(f"  Hop size:             {args.hop_size} samples ({args.hop_size / 16000 * 1000:.1f}ms)")
    print(f"  Min speech duration:  {args.min_speech_duration}s")
    print(f"  Max speech duration:  {args.max_speech_duration}s{'  (disabled)' if args.max_speech_duration <= 0 else ''}")
    print(f"  Min silence duration: {args.min_silence_duration}s")
    print(f"  Padding:              {args.padding}s")
    print(f"  Mode:                 STREAMING")
    print("=" * 60)

    # Find all WAV files recursively
    wav_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True))
    print(f"\nFound {len(wav_files)} WAV files\n")

    if not wav_files:
        print("No WAV files found!")
        return

    all_results = []
    total_segments = 0
    total_files_processed = 0
    total_files_skipped = 0

    # Group files by parent directory
    dir_files = {}
    for f in wav_files:
        parent = os.path.basename(os.path.dirname(f))
        if parent not in dir_files:
            dir_files[parent] = []
        dir_files[parent].append(f)

    for subdir_name in sorted(dir_files.keys()):
        files = dir_files[subdir_name]
        out_subdir = os.path.join(args.output_dir, subdir_name)
        os.makedirs(out_subdir, exist_ok=True)

        print(f"--- Processing [{subdir_name}] ({len(files)} files) ---")

        for wav_path in files:
            basename = os.path.basename(wav_path)
            result = process_file_streaming(
                wav_path, out_subdir,
                args.hop_size, args.threshold,
                args.min_speech_duration, args.max_speech_duration,
                args.min_silence_duration, args.padding
            )

            if result is None:
                total_files_skipped += 1
                continue

            total_files_processed += 1
            total_segments += result["num_segments"]
            all_results.append(result)

            seg_info = ", ".join(
                f"{s['start_time']:.2f}-{s['end_time']:.2f}s"
                for s in result["segments"]
            ) if result["segments"] else "no speech"
            print(f"  {basename}: {result['num_segments']} segments [{seg_info}]")

    # Save metadata JSON
    metadata_path = os.path.join(args.output_dir, "vad_segments_metadata.json")
    metadata = {
        "config": {
            "threshold": args.threshold,
            "hop_size": args.hop_size,
            "min_speech_duration": args.min_speech_duration,
            "max_speech_duration": args.max_speech_duration,
            "min_silence_duration": args.min_silence_duration,
            "padding": args.padding,
            "mode": "streaming",
        },
        "summary": {
            "total_files": len(wav_files),
            "files_processed": total_files_processed,
            "files_skipped": total_files_skipped,
            "total_segments": total_segments,
        },
        "results": all_results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Files processed: {total_files_processed}")
    print(f"  Files skipped:   {total_files_skipped}")
    print(f"  Total segments:  {total_segments}")
    print(f"  Metadata saved:  {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
