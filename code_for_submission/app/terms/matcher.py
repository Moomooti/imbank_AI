"""Phase 3 — 용어 치환기 (A/D 그룹).

핵심 원칙(Phase 2 인터페이스 기반):
    - 치환은 원문 오프셋 방식으로만 한다: text[:start] + 대체값 + text[end:]
    - 토큰을 join해서 문장을 재구성하지 않는다 — 미얀마어는 공백이 토큰으로
      안 나와서(Phase 2에서 확인) join하면 원문이 깨진다.
    - 매치가 여러 개면 뒤에서 앞으로(start가 큰 것부터) 역순으로 치환해야
      앞쪽 오프셋이 안 밀린다.
    - 겹치는 매치는 (긴 용어 먼저, 그다음 match_priority) 순으로 우선순위를 매겨
      그리디하게 겹치지 않는 것만 채택한다.

언어별 매칭 방식:
    - 라틴 문자(vie_Latn, ind_Latn, tgl_Latn): 단어 경계가 있으니 분절기가 필요
      없고, `(?<!\\w)term(?!\\w)` 정규식으로 충분하다.
    - 공백 없는 문자(tha_Thai, mya_Mymr): Phase 2 `segment_words`로 토큰화한 뒤
      토큰 시퀀스가 연속으로 일치하는 구간을 찾는다 (부분 문자열 매칭은 단어
      중간을 잘라먹을 수 있어 위험).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pythainlp.tokenize import Trie

from app.segment import Token, segment_words
from app.segment.thai import segment_thai_words
from app.terms.glossary_index import LATIN_LANGS, TOKENIZED_LANGS, GlossaryRow


@dataclass(frozen=True)
class TermMatch:
    start: int
    end: int
    matched_text: str                             
    ko_term: str
    match_priority: int
    is_loanword: bool


def find_term_matches(text: str, lang: str, rows: list[GlossaryRow]) -> list[TermMatch]:
    if lang in LATIN_LANGS:
        candidates = _find_regex(text, rows)
    elif lang in TOKENIZED_LANGS:
        candidates = _find_tokenized(text, lang, rows)
    else:
        raise ValueError(f"지원하지 않는 언어: {lang!r}")
    return _resolve_overlaps(candidates)


def _find_regex(text: str, rows: list[GlossaryRow]) -> list[TermMatch]:
    candidates: list[TermMatch] = []
    for row in rows:
        term = row.term_text
        if not term.strip():
            continue
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append(
                TermMatch(m.start(), m.end(), m.group(), row.ko_term, row.match_priority, row.is_loanword)
            )
    return candidates


def _content_tokens(tokens: list[Token]) -> list[Token]:
    """공백뿐인 토큰(태국어 keep_whitespace=True로 섞여 들어옴)은 시퀀스 비교에서
    제외한다 — 매치 구간의 시작/끝은 실제(비공백) 토큰의 오프셋으로 잡히므로,
    두 단어 사이의 공백은 어차피 매치 구간 안에 자연스럽게 포함된다."""
    return [t for t in tokens if t.text.strip()]


def _find_tokenized(text: str, lang: str, rows: list[GlossaryRow]) -> list[TermMatch]:
    term_dict: Trie | None = None
    if lang == "tha_Thai":
                                                  
                                                  
                                                                   
                                                  
                                                           
                                                     
                                                     
                                              
        term_dict = Trie({row.term_text for row in rows if row.term_text.strip()})
        text_tokens = _content_tokens(segment_thai_words(text, custom_dict=term_dict))
    else:
        text_tokens = _content_tokens(segment_words(text, lang))
    text_pieces = [t.text for t in text_tokens]

                            
    term_token_cache: dict[str, list[str]] = {}

    candidates: list[TermMatch] = []
    for row in rows:
        term = row.term_text
        if not term.strip():
            continue
        if term not in term_token_cache:
            if lang == "tha_Thai":
                term_tokens = segment_thai_words(term, custom_dict=term_dict)
            else:
                term_tokens = segment_words(term, lang)
            term_token_cache[term] = [t.text for t in _content_tokens(term_tokens)]
        term_pieces = term_token_cache[term]
        n = len(term_pieces)
        if n == 0:
            continue
        for i in range(len(text_pieces) - n + 1):
            if text_pieces[i : i + n] == term_pieces:
                start = text_tokens[i].start
                end = text_tokens[i + n - 1].end
                candidates.append(
                    TermMatch(start, end, text[start:end], row.ko_term, row.match_priority, row.is_loanword)
                )
    return candidates


def _resolve_overlaps(candidates: list[TermMatch]) -> list[TermMatch]:
    """긴 용어 먼저, 그다음 match_priority 순으로 정렬해 그리디하게 겹치지 않는
    매치만 채택한다."""
    ordered = sorted(candidates, key=lambda m: (-(m.end - m.start), m.match_priority))
    accepted: list[TermMatch] = []
    claimed: list[tuple[int, int]] = []
    for m in ordered:
        if any(not (m.end <= s or m.start >= e) for s, e in claimed):
            continue
        accepted.append(m)
        claimed.append((m.start, m.end))
    return sorted(accepted, key=lambda m: m.start)


def apply_matches(text: str, matches: list[TermMatch], value_fn) -> str:
    """value_fn(TermMatch) -> str. 뒤(끝 오프셋이 큰 것)부터 역순으로 치환해서
    앞쪽 오프셋이 안 밀리게 한다. 토큰을 join하지 않고 원문 문자열을 그대로
    슬라이싱-접합한다."""
    result = text
    for m in sorted(matches, key=lambda m: m.start, reverse=True):
        value = value_fn(m)
        result = result[: m.start] + value + result[m.end :]
    return result
