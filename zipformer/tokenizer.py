"""Token-based text encoder compatible with sentencepiece API.

Loads tokens.txt (Whisper-style BPE vocabulary) and provides encode/decode
with the same interface as sentencepiece.SentencePieceProcessor so that
finetune.py can use it as a drop-in replacement.

Token format in tokens.txt:
    <blk> 0
    <sos/eos> 1
    <unk> 2
    ▁MỘT 3
    ▁VÀ 4
    ...
    N 83
    ...

Encoding strategy (greedy longest-match):
    1. Split text on whitespace → words
    2. For each word, try ▁WORD (full word token). If found, emit it.
    3. Otherwise, greedily match longest subword/char tokens from left to right.
    4. Unknown characters → <unk> id.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union


class TokenEncoder:
    """Drop-in replacement for spm.SentencePieceProcessor for tokens.txt."""

    def __init__(self, tokens_path: Union[str, Path]):
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}
        self._unk_id = 2
        self._blk_id = 0

        with open(tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                token, idx_str = parts
                idx = int(idx_str)
                self._token2id[token] = idx
                self._id2token[idx] = token

        # Build sorted subword list (longest first) for greedy matching
        # Exclude special tokens from matching
        self._subwords = sorted(
            [t for t in self._token2id if not t.startswith("<")],
            key=lambda t: len(t),
            reverse=True,
        )

        # Word-level tokens (▁-prefixed) for fast lookup
        self._word_tokens = {
            t: self._token2id[t]
            for t in self._token2id
            if t.startswith("▁") and not t.startswith("<")
        }

    def get_piece_size(self) -> int:
        return len(self._token2id)

    def piece_to_id(self, piece: str) -> int:
        return self._token2id.get(piece, self._unk_id)

    def id_to_piece(self, idx: int) -> str:
        return self._id2token.get(idx, "<unk>")

    def encode(
        self,
        input: Union[str, List[str]],
        out_type: type = int,
    ) -> Union[List[int], List[List[int]]]:
        """Encode text(s) to token ids.

        Args:
            input: single string or list of strings.
            out_type: int (return ids) or str (return token strings).

        Returns:
            List of ids (single input) or list of list of ids (batch input).
        """
        if isinstance(input, str):
            return self._encode_one(input, out_type)
        return [self._encode_one(s, out_type) for s in input]

    def _encode_one(self, text: str, out_type: type) -> list:
        text = text.strip()
        if not text:
            return []
        # tokens.txt in this project is mostly uppercase (e.g., ▁ĐÓ, ▁LÀ, ...).
        # Normalize input text to uppercase to avoid excessive <unk> for lowercase labels.
        text = text.upper()

        tokens = []
        words = text.split()

        for word in words:
            # Try full word token
            word_token = "▁" + word
            if word_token in self._word_tokens:
                tokens.append(word_token)
            else:
                # Prefix token for word boundary
                remaining = "▁" + word
                while remaining:
                    matched = False
                    for sub in self._subwords:
                        if remaining.startswith(sub):
                            tokens.append(sub)
                            remaining = remaining[len(sub):]
                            matched = True
                            break
                    if not matched:
                        # Unknown char, skip it with <unk>
                        tokens.append("<unk>")
                        remaining = remaining[1:]

        if out_type == int:
            return [self._token2id.get(t, self._unk_id) for t in tokens]
        return tokens

    def decode(self, ids):
        """Decode token ids back to text.

        Supports both a single token-id list and a batch list-of-lists,
        matching sentencepiece decode() behavior used in decode.py.
        """
        if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            return [self.decode(x) for x in ids]

        pieces = [self._id2token.get(int(i), "") for i in ids if int(i) != self._blk_id]
        text = "".join(pieces).replace("▁", " ").strip()
        return text

    # Alias for sentencepiece compatibility
    def load(self, path: str):
        """No-op for API compatibility — already loaded in __init__."""
        pass
