"""Phase 4-A 2단계 — confidence-gated 마스킹 정책 로더.

Phase 1(COMET-Kiwi)~2(LaBSE)에서 확인된 사실: 베트남/인니/태국(DeepL 백엔드)은
마스킹이 평균 순이득이지만, 필리핀/미얀마(NLLB 백엔드)는 평균 순손해다. 그래서
필리핀/미얀마는 기본을 "비마스킹"으로 뒤집고, Phase 4-A에서 복구+baseline
능가까지 검증된 용어(mask_allowlist_{lang}.csv)만 예외적으로 마스킹을
허용한다.

masking_policy.csv / mask_allowlist_*.csv가 아직 없으면(구버전 체크아웃 등)
전부 full_mask로 폴백한다 -- 정책 파일 부재가 곧 "마스킹 금지"로 조용히
바뀌면 더 위험하므로, 안전한 기본값은 "기존 동작 유지"다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POLICY_CSV = ROOT / "data" / "masking" / "masking_policy.csv"

FULL_MASK = "full_mask"
CONFIDENCE_GATED = "confidence_gated"


@lru_cache(maxsize=1)
def load_policy() -> dict[str, str]:
    """lang -> "full_mask" | "confidence_gated". 정책 파일 없으면 빈 dict(=전부 full_mask 취급)."""
    if not POLICY_CSV.exists():
        return {}
    df = pd.read_csv(POLICY_CSV, encoding="utf-8-sig")
    return {
        str(r.lang): CONFIDENCE_GATED if str(r.policy).startswith(CONFIDENCE_GATED) else FULL_MASK
        for r in df.itertuples(index=False)
    }


@lru_cache(maxsize=8)
def load_allowlist(lang: str) -> frozenset[str]:
    path = ROOT / "data" / "masking" / f"mask_allowlist_{lang}.csv"
    if not path.exists():
        return frozenset()
    df = pd.read_csv(path, encoding="utf-8-sig")
    return frozenset(df["ko_term"])


def is_gated(lang: str) -> bool:
    return load_policy().get(lang) == CONFIDENCE_GATED


def filter_ko_terms_for_lang(ko_terms: list[tuple[str, int]], lang: str) -> list[tuple[str, int]]:
    """mask_text에 넘기기 전에 언어 정책을 적용해 ko_terms 목록을 걸러낸다.

    full_mask 언어(또는 정책 파일이 아예 없는 경우)는 그대로 반환(기존 동작).
    confidence_gated 언어는 allowlist에 있는 용어만 남긴다 -- 나머지는 애초에
    "찾을 대상"에서 빠지므로 masking 자체가 안 걸리고 원문 그대로 통째로
    NLLB에 들어가 순정 번역된다.
    """
    if not is_gated(lang):
        return ko_terms
    allow = load_allowlist(lang)
    return [(t, p) for t, p in ko_terms if t in allow]
