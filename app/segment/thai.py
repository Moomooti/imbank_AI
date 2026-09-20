"""Phase 2 — 태국어 단어 분절.

태국어는 우리 5개 언어 중 공백 없이 쓰는 문자라 단어 경계를 알아야 Phase 3에서
용어를 안전하게(단어 중간을 잘라먹지 않고) 찾아 치환할 수 있다.

엔진은 "newmm"(사전 기반 최장일치, pythainlp 기본값)으로 고정한다. pythainlp는
newmm 외에 attacut(딥러닝), longest, icu 등 여러 엔진을 지원하는데, 엔진마다
분절 결과가 달라진다 — 재현성을 위해 코드에서 명시적으로 고정해두고, 나중에
바꾸려면 여기 한 곳만 고치면 되게 한다.
"""
from __future__ import annotations

from pythainlp.tokenize import Trie, word_tokenize

from app.segment.types import Token, tokens_with_offsets

THAI_WORD_ENGINE = "newmm"


def segment_thai_words(text: str, custom_dict: Trie | None = None) -> list[Token]:
    """custom_dict: 알고 있는 용어(예: 용어사전 번역문)를 Trie로 넘기면 newmm이
    그 경계를 우선해서 잡는다. Phase 3 용어 매칭에서, 공백 없이 다른 말과 바로
    붙어버린 용어도 안정적으로 하나의 경계로 잡히게 하려고 추가했다 (실측: 커스텀
    사전 없이는 문맥에 따라 용어 시작 지점이 옆 단어와 뭉쳐버리는 경우가 있었음).
    기본값 None이면 기존과 동일하게 동작한다.
    """
    kwargs = {"custom_dict": custom_dict} if custom_dict is not None else {}
    pieces = word_tokenize(text, engine=THAI_WORD_ENGINE, keep_whitespace=True, **kwargs)
    return tokens_with_offsets(text, pieces)
