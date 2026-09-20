"""Phase 2 — 태국어/미얀마어 분절기 통합 인터페이스.

Phase 3(용어 치환)는 이 모듈의 `segment_words()`만 알면 된다: 언어 코드 하나로
어느 백엔드를 쓸지 신경 쓰지 않고 오프셋 붙은 토큰 리스트를 받을 수 있다.
"""
from __future__ import annotations

from app.segment.myanmar import (
    CLAUSE_SEP,
    SENTENCE_END,
    segment_myanmar_words,
    split_myanmar_sentences,
)
from app.segment.thai import THAI_WORD_ENGINE, segment_thai_words
from app.segment.types import Token, tokens_with_offsets

_WORD_SEGMENTERS = {
    "tha_Thai": segment_thai_words,
    "mya_Mymr": segment_myanmar_words,
}


def segment_words(text: str, lang: str) -> list[Token]:
    """lang: FLORES 코드 ('tha_Thai' 또는 'mya_Mymr')."""
    try:
        fn = _WORD_SEGMENTERS[lang]
    except KeyError:
        raise ValueError(f"지원하지 않는 언어: {lang!r} (지원: {list(_WORD_SEGMENTERS)})") from None
    return fn(text)


__all__ = [
    "Token",
    "tokens_with_offsets",
    "segment_words",
    "segment_thai_words",
    "segment_myanmar_words",
    "split_myanmar_sentences",
    "THAI_WORD_ENGINE",
    "SENTENCE_END",
    "CLAUSE_SEP",
]
