# Streaming Cache-Aware Trong Multi-Speaker Zipformer

Tài liệu này dùng để thuyết trình phần "model đang streaming kiểu gì" trong code hiện tại.

## 1) Kết luận ngắn gọn

Model hiện tại là **streaming cache-aware thật**.

Nó không chỉ:
- cắt audio thành chunk ngắn
- giới hạn right-context

Mà còn:
- giữ state qua chunk cho conv frontend
- giữ state qua chunk cho Zipformer/Sortformer/Adapter/Decoder
- cập nhật lens/counters chính xác theo từng chunk

## 2) Cache-aware khác gì pseudo-streaming?

`Pseudo-streaming`:
- chỉ chạy theo chunk độc lập
- chunk sau không dùng thông tin cache từ chunk trước

`Cache-aware streaming` (code hiện tại):
- mỗi chunk nhận `state` từ chunk trước
- xử lý xong trả `new_state`
- chunk sau dùng lại `new_state`

Vì vậy mô hình "nhớ" lịch sử ngắn hạn cần thiết mà không phải re-run toàn bộ audio.

## 3) State nào đang được giữ?

### 3.1 `StreamingState` (wrapper tổng)

Trong `streaming_multi_speaker.py`, `StreamingState` giữ:
- `conv_frontend_state`: cache frontend waveform/normalization
- `zipformer_states`: cache nội bộ encoder theo layer
- `sortformer_states`: cache nhánh diarization
- `speaker_selector_states`: cache adapter theo speaker
- `processed_lens_base`: cumulative lens để build mask left-context đúng
- `early_sortformer_states`, `injector_cond_states`, `slot_state`: cho mid-injection
- `decoder_state`, `beam_search_state`: state decode online
- `cum_valid_samples`, `cum_valid_feat_frames`: counters cho length chính xác

### 3.2 Conv frontend cache

Trong `wav2vec2_module.py`, `StreamingConvFrontend.get_init_state()` trả:
- `wav_cache`: suffix waveform chưa "tiêu thụ" hết
- `norm_state`: trạng thái cumulative normalization
- `block0_feat_cache`: cache feature block0 cho phần overlap
- `exact_len_state`: đếm `cum_valid_samples`, `cum_valid_feat_frames`

### 3.3 Adapter/Sortformer cache

Trong `speaker_modules.py`:
- mỗi `_AdapterLayer` có `attn_cache` + `conv_cache`
- `SpeakerSelectorAdapter` giữ `act_cache` + `layer_states`
- `SortformerBranch` giữ `layer_states`

### 3.4 Slot memory cache (nếu bật mid-injection)

`SpeakerSlotMemory` giữ:
- `E_sum`
- `E_count`

Đây là running slot prototypes theo speaker.

## 4) Luồng inference streaming (audio-level)

Pipeline thực tế:

1. `streaming_forward_from_audio(audio_chunk, ..., state, sample_lens)`
2. frontend xử lý chunk mới + cache cũ
3. Zipformer streaming dùng `zipformer_states`
4. diar/adapter streaming dùng state riêng
5. trả `h_all_chunk`, `h_lens_chunk`, `new_state`
6. decode chunk rồi tiếp tục với `new_state`

Ví dụ tối giản:

```python
import torch
from streaming_multi_speaker import StreamingMultiSpeakerEncoder

# ms_model: MultiSpeakerRnntModel đã load trọng số
streamer = StreamingMultiSpeakerEncoder(ms_model).to(device)

B = 1
state = streamer.init_state(batch_size=B, device=device, init_decode=True)

for audio_chunk, sample_lens in stream_source:  # audio_chunk: [B, T_samples]
    h_all_chunk, h_lens_chunk, state = streamer.streaming_forward_from_audio(
        audio_chunk=audio_chunk.to(device),
        spk_activity_chunk=None,     # dùng sortformer nội bộ nếu có
        state=state,
        sample_lens=sample_lens.to(device),
    )

    # Greedy decode online
    state.decoder_state = streamer.streaming_greedy_decode_chunk(
        h_all_chunk=h_all_chunk,
        h_lens_chunk=h_lens_chunk,
        decoder_state=state.decoder_state,
        max_sym_per_frame=1,
    )
```

## 5) Đoạn code demo để "chứng minh cache đang chạy"

Bạn có thể in state sau mỗi chunk khi demo:

```python
wav_cache, norm_state, block0_feat_cache, exact_len_state = state.conv_frontend_state

print("wav_cache:", tuple(wav_cache.shape))
print("zipformer_state_tensors:", len(state.zipformer_states))
print("processed_lens_base:", state.processed_lens_base.tolist())
print("cum_valid_samples:", exact_len_state["cum_valid_samples"].tolist())
print("cum_valid_feat_frames:", exact_len_state["cum_valid_feat_frames"].tolist())
```

Nếu cache-aware hoạt động đúng, các giá trị trên sẽ thay đổi liên tục theo chunk, không reset về 0.

## 6) Trả lời câu hỏi thường gặp

### "Cache này có giữ full encoder output không?"

**Không.**

Nó giữ **state tối thiểu** để tiếp tục chunk sau:
- cache attention/conv per-layer
- lens counters
- decoder context/hypothesis

Không giữ toàn bộ hidden sequence của toàn utterance như offline full-context.

### "Vậy right-context có còn không?"

Với cấu hình causal hiện tại:
- right-context tương lai bị chặn
- chỉ dùng chunk hiện tại + left-context từ cache

Đây chính là streaming inference chuẩn.

## 7) Gợi ý trình bày 1 slide

Tiêu đề:
`Cache-aware Streaming = Chunk + Stateful Memory`

3 ý chính:
- `Chunk input` để giảm latency
- `State carry-over` để giữ ngữ cảnh lịch sử
- `Causal masking` để không nhìn tương lai

1 câu kết:
`Hệ thống hiện tại là streaming cache-aware thật, không phải pseudo-streaming theo chunk độc lập.`

