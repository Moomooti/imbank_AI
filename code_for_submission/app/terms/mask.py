"""Phase 3.5 — mask / restore.

원칙: 이 번역들은 정답이 아니라 초벌 seed라는 프로젝트 원칙과 별개로, 금융
용어는 번역기가 손대면 안 되는 "보호 대상"이다(P2). 그래서:

    1) mask   : 한국어 원문에서 용어(ko_term)를 찾아 식별자로 가린다
    2) translate: 식별자가 박힌 한국어 문장을 그대로 NLLB/DeepL에 넣는다
                  (식별자는 번역기가 못 알아먹고 그대로 통과시켜야 함)
    3) restore: 번역 결과에서 식별자를 찾아 사전의 {lang}_draft 정답 번역으로
                  바꿔치기한다

식별자 형식은 {{FINTERM001}} (중괄호 두 겹) 로 확정 — 실측으로 검증됨:
    - bare/curly/square 세 형식이 다 두 라운드 합쳐 100% 생존
    - square([...])는 우리 실제 데이터(FSC 보도자료 "[1] 조각투자를...")에
      조항 번호로 이미 쓰이고 있어 원문과 충돌 위험 있음 -> 제외
    - underscore(__...__)는 식별자끼리 쉼표로 바로 붙으면 NLLB 미얀마가
      앞쪽 "__"를 벗겨버리는 실패가 나옴(46/48, 100% 아님) -> 제외
    - curly는 i18n 플레이스홀더 관례라 MT가 이미 "안 건드리는 패턴"으로
      학습돼 있을 가능성이 높고, 우리 도메인에 자연발생할 일이 없음 -> 채택
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.terms.glossary_index import GlossaryRow
from app.terms.matcher import TermMatch, _resolve_overlaps, apply_matches
from app.terms.policy import filter_ko_terms_for_lang

MASK_TEMPLATE = "{{{{FINTERM{:03d}}}}}"                       
MASK_RE = re.compile(r"\{\{FINTERM\d+\}\}")


@dataclass(frozen=True)
class MaskEntry:
    identifier: str
    ko_term: str
    match_priority: int
    order: int                              


def find_term_matches_ko(text: str, ko_terms: list[tuple[str, int]]) -> list[TermMatch]:
    """한국어 원문에서 ko_term을 찾는다.

    한국어는 조사(을/는/이/가 등)가 공백 없이 명사에 바로 붙으므로, 라틴 문자용
    (?<!\\w)/(?!\\w) 단어경계 lookaround를 그대로 쓰면 조사가 붙은 자리를 못
    찾는다(예: "선물환을"에서 "선물환" 뒤에 "을"이 와서 경계 판정 실패).
    한국어 복합명사는 임의 음절이 이어붙어 우연히 부분일치할 위험이 라틴 문자
    보다는 낮다고 보고, 순수 부분 문자열 검색으로 찾는다.
    """
    candidates: list[TermMatch] = []
    for ko_term, match_priority in ko_terms:
        if not ko_term.strip():
            continue
        for m in re.finditer(re.escape(ko_term), text):
            candidates.append(TermMatch(m.start(), m.end(), m.group(), ko_term, match_priority, False))
    return _resolve_overlaps(candidates)


def mask_text(text: str, ko_terms: list[tuple[str, int]], lang: str | None = None) -> tuple[str, list[MaskEntry]]:
    """text 안의 ko_term 출현을 왼쪽부터 순서대로 {{FINTERM001}}, 002... 로 치환.
    반환된 masked 텍스트와 MaskEntry 목록(어떤 식별자가 어떤 용어였는지)을 같이
    돌려준다 — restore 단계에서 이 목록이 있어야 무엇으로 되돌릴지 안다.

    lang을 주면 confidence-gated 정책(masking_policy.csv)을 적용한다 -- 필리핀/
    미얀마처럼 마스킹이 평균 순손해로 확인된 언어는 allowlist에 없는 용어를
    애초에 검색 대상에서 뺀다(그 용어는 마스킹 없이 원문 그대로 순정 NLLB로
    번역됨). lang을 안 주면(기존 호출부 호환) 필터링 없이 전부 검색한다.
    """
    if lang is not None:
        ko_terms = filter_ko_terms_for_lang(ko_terms, lang)
    matches = find_term_matches_ko(text, ko_terms)
    ordered = sorted(matches, key=lambda m: m.start)

    entries: list[MaskEntry] = []
    identifier_by_match_id: dict[int, str] = {}
    for i, m in enumerate(ordered, start=1):
        identifier = MASK_TEMPLATE.format(i)
        entries.append(MaskEntry(identifier=identifier, ko_term=m.ko_term, match_priority=m.match_priority, order=i))
        identifier_by_match_id[id(m)] = identifier

    masked = apply_matches(text, matches, lambda m: identifier_by_match_id[id(m)])
    return masked, entries


@dataclass
class RestoreReport:
    total: int = 0
    survived: int = 0                                
    order_preserved_pairs: int = 0                             
    order_total_pairs: int = 0
    restored: int = 0                       
    fallback_used: int = 0                       
    details: list[dict] = field(default_factory=list)

    @property
    def survival_rate(self) -> float:
        return self.survived / self.total if self.total else 1.0

    @property
    def order_preservation_rate(self) -> float:
        return self.order_preserved_pairs / self.order_total_pairs if self.order_total_pairs else 1.0

    @property
    def restore_success_rate(self) -> float:
        return self.restored / self.total if self.total else 1.0


def restore_text(translated: str, entries: list[MaskEntry], lang_rows: list[GlossaryRow]) -> tuple[str, RestoreReport]:
    """translated(식별자가 박힌 번역 결과)에서 식별자를 찾아 lang_rows의
    {lang}_draft 값으로 되돌린다.

    - 위치보존율: 살아남은 식별자들을 "원래 순서(order)" 기준으로 줄 세웠을 때,
      번역 결과에서의 실제 등장 위치(index)도 그 순서대로 커지는지를 본다.
      즉 001이 002보다 앞에 나왔어야 하는데 뒤로 밀렸으면 위반 1건으로 센다.
      (어순이 다른 언어라 어느 정도 순서가 바뀌는 건 정상이지만, 식별자끼리의
      상대순서가 뒤집히면 문장 구조 자체가 크게 흐트러졌을 가능성이 커서
      그런 사고만 골라서 잡아낸다.)
    - 식별자가 아예 유실(문자열 자체가 안 남음)됐으면 위치를 알 수 없으니
      "복원 못 함(수동 검토 필요)"으로만 로그에 남기고, 조용히 넘어가거나
      임의 위치에 억지로 끼워넣지 않는다.
    - 식별자는 살아남았는데 그 언어의 draft 번역이 비어있으면(아직 번역 안 된
      경우 등) 한국어 원어로 폴백한다.
    """
    term_by_ko = {r.ko_term: r.term_text for r in lang_rows}

    found: list[tuple[int, MaskEntry]] = []
    details: list[dict] = []
    for e in entries:
        idx = translated.find(e.identifier)
        if idx == -1:
            details.append({"order": e.order, "ko_term": e.ko_term, "identifier": e.identifier, "status": "lost"})
            continue
        found.append((idx, e))

    found_by_order = sorted(found, key=lambda x: x[1].order)
    order_total_pairs = max(len(found_by_order) - 1, 0)
    order_preserved_pairs = sum(
        1 for (idx_a, _), (idx_b, _) in zip(found_by_order, found_by_order[1:]) if idx_a < idx_b
    )

    result = translated
    restored = 0
    fallback_used = 0
    for idx, e in sorted(found, key=lambda x: x[0], reverse=True):           
        restore_value = term_by_ko.get(e.ko_term, "")
        if restore_value.strip():
            value, status = restore_value, "restored"
            restored += 1
        else:
            value, status = e.ko_term, "fallback_no_translation"                           
            fallback_used += 1
        result = result[:idx] + value + result[idx + len(e.identifier) :]
        details.append({"order": e.order, "ko_term": e.ko_term, "identifier": e.identifier, "status": status})

    report = RestoreReport(
        total=len(entries),
        survived=len(found),
        order_preserved_pairs=order_preserved_pairs,
        order_total_pairs=order_total_pairs,
        restored=restored,
        fallback_used=fallback_used,
        details=sorted(details, key=lambda d: d["order"]),
    )
    return result, report
