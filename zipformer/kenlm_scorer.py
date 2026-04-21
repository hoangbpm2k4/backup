"""
KenLM-based n-gram scorer for inline shallow fusion in streaming RNNT beam search.

Follows the same incremental-state pattern as icefall's ``NgramLmStateCost``
(icefall/ngram_lm.py) but uses KenLM Python bindings directly instead of
kaldifst FST.  Each hypothesis carries a ``kenlm.State`` object; on token
emission the state is advanced and the log-probability delta is returned.

Usage — training the n-gram::

    # 1. Tokenize text corpus into BPE pieces
    python -c "
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load('bpe.model')
    with open('corpus.txt') as f, open('corpus_bpe.txt', 'w') as out:
        for line in f:
            pieces = sp.EncodeAsPieces(line.strip())
            out.write(' '.join(pieces) + '\\n')
    "

    # 2. Train n-gram with KenLM (install: pip install
    #    https://github.com/kpu/kenlm/archive/master.zip )
    lmplz -o 3 --prune 0 1 2 < corpus_bpe.txt > lm_3gram.arpa

    # 3. (Optional) convert to binary for faster loading
    build_binary lm_3gram.arpa lm_3gram.bin

Usage — decoding::

    from kenlm_scorer import KenLmScorer
    scorer = KenLmScorer("lm_3gram.arpa", sp_model_path="bpe.model")

    # pass to streaming_step / streaming_beam_search_chunk:
    #   streaming_step(..., beam=4, lm_scorer=scorer, lm_scale=0.3)
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


class KenLmScorer:
    """Incremental KenLM n-gram scorer wrapping ``kenlm.State``.

    Each beam-search hypothesis carries a ``kenlm.State`` object.
    When a non-blank token is emitted, call :meth:`score_token` to advance
    the state and obtain the log-probability delta (in **natural log**, matching
    RNNT log-softmax).

    Args:
        kenlm_path: path to ARPA or binary (``.bin``) KenLM file.
        sp_model_path: path to SentencePiece ``.model`` for id → piece mapping.
            If *None*, ``token_list`` is used; if both are *None*, tokens are
            converted to ``str(token_id)``.
        token_list: list where ``token_list[id]`` gives the string piece.
            Takes precedence over *sp_model_path*.
    """

    # log(10), used to convert KenLM log10 scores → natural log
    _LOG10: float = math.log(10.0)

    def __init__(
        self,
        kenlm_path: str,
        sp_model_path: Optional[str] = None,
        token_list: Optional[List[str]] = None,
    ):
        try:
            import kenlm as _kenlm
        except ImportError:
            raise ImportError(
                "KenLM Python bindings not found.  Install with:\n"
                "  pip install https://github.com/kpu/kenlm/archive/master.zip"
            )

        self.model: Any = _kenlm.Model(kenlm_path)
        self._kenlm = _kenlm  # keep module ref for State() creation

        # Build id → piece mapping
        if token_list is not None:
            self._pieces: Optional[List[str]] = list(token_list)
        elif sp_model_path is not None:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(sp_model_path)
            self._pieces = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
        else:
            self._pieces = None  # fallback: str(token_id)

    def _tok_to_piece(self, token_id: int) -> str:
        if self._pieces is not None and 0 <= token_id < len(self._pieces):
            return self._pieces[token_id]
        return str(token_id)

    def get_init_state(self) -> Any:
        """Return the initial KenLM state (beginning-of-sentence)."""
        state = self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    def score_token(
        self, token_id: int, prev_state: Any
    ) -> Tuple[float, Any]:
        """Advance KenLM by one BPE token.

        Args:
            token_id: integer token ID (from RNNT vocab).
            prev_state: ``kenlm.State`` from the parent hypothesis.

        Returns:
            ``(ln_score, new_state)`` where *ln_score* is the **natural-log**
            probability of this token given the n-gram history, and
            *new_state* is the updated ``kenlm.State`` for the next step.
        """
        piece = self._tok_to_piece(token_id)
        new_state = self._kenlm.State()
        log10_score: float = self.model.BaseScore(prev_state, piece, new_state)
        return log10_score * self._LOG10, new_state

    def score_sequence(self, token_ids: List[int]) -> float:
        """Score an entire token sequence (convenience for n-best rescoring).

        Same as ``kenlm.Model.score()`` but using token IDs.

        Returns:
            Total **natural-log** probability of the sequence.
        """
        pieces = " ".join(self._tok_to_piece(t) for t in token_ids)
        log10_total: float = self.model.score(pieces)
        return log10_total * self._LOG10
