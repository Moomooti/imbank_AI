"""Phase 2 — 미얀마어 단어/문장 분절.

단어 분절은 pyidaungsu(CRF 기반)를 그대로 쓴다.

문장 분절은 pyidaungsu에 기능 자체가 없어서(사전검증 단계에서 확인됨) 직접
규칙 기반으로 구현한다:
    - ။ (U+104B, 문장 종결부호)로만 문장 경계를 잡는다.
    - ၊ (U+104A, 절 구분부호)는 문장 안의 쉼표 같은 역할이라 경계로 쓰지 않는다.
    - 숫자 바로 뒤에 오는 ။는 문장 끝이 아니라 "၁။", "(၂)။" 같은 번호매김(리스트
      항목 표시)인 경우가 많다 — 특히 정부/금융 공문서에서 흔한 패턴이라 이걸
      실수로 문장 경계로 잡으면 조항 번호마다 문장이 끊어지는 오탐이 난다.
      그래서 숫자(미얀마 숫자 0x1040-0x1049, 아라비아 숫자 모두) 바로 뒤의
      ။는 경계로 보지 않는다.
"""
from __future__ import annotations

import pyidaungsu as pds

from app.segment.types import Token, tokens_with_offsets

SENTENCE_END = "။"     
CLAUSE_SEP = "၊"                                  

_MYANMAR_DIGITS = "".join(chr(c) for c in range(0x1040, 0x104A))
_DIGITS = set("0123456789" + _MYANMAR_DIGITS)


def segment_myanmar_words(text: str) -> list[Token]:
    pieces = pds.tokenize(text, form="word")
    return tokens_with_offsets(text, pieces)


def _is_numbering_marker(text: str, end_idx: int) -> bool:
    """text[end_idx] == SENTENCE_END 라고 가정. 그 ။가 번호매김(예: '၁။',
    '(၂)။')인지 판단.

    핵심 기준: 숫자 덩어리 자체가 "줄/구절의 맨 처음"에 있어야 번호로 인정한다
    (문자열 시작, 줄바꿈, 또는 직전 문장/절 경계 바로 다음). 문장 중간에 공백
    하나 두고 나온 숫자(예: "...총 100 ။" 처럼 그냥 숫자로 끝나는 문장)까지
    번호매김으로 오인하면 진짜 문장 경계를 삼켜버리므로, 일반 공백은 경계로
    인정하지 않는다 — 실제 관공서 문서의 번호매김은 거의 항상 줄 맨 앞에 온다.
    """
    j = end_idx - 1
    if j >= 0 and text[j] == ")":
        j -= 1
    saw_digit = False
    while j >= 0 and text[j] in _DIGITS:
        saw_digit = True
        j -= 1
    if not saw_digit:
        return False
    return j < 0 or text[j] in "\n" + SENTENCE_END + CLAUSE_SEP


def split_myanmar_sentences(text: str) -> list[str]:
    """။ 기준으로만 문장을 나눈다. 번호매김으로 판단되는 ။는 건너뛴다.
    ။로 끝나지 않는 마지막 조각(문장부호 없는 문서 끝)도 비어있지 않으면 포함한다.
    각 문장에는 자신을 끝맺는 ။가 그대로 붙어 있다.
    """
    sentences: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch == SENTENCE_END and not _is_numbering_marker(text, i):
            sentences.append(text[start : i + 1])
            start = i + 1
    remainder = text[start:]
    if remainder.strip():
        sentences.append(remainder)
    return sentences
